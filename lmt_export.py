# SPDX-License-Identifier: GPL-3.0-or-later
"""MT Framework LMT replacement exporter.

Replacement Actions can have a different frame count and LoopFrame from the
original destination slot. Split pre-loop/loop Actions are reassembled into one
LMT animation. Existing track storage is reused and only size growth is inserted,
with known absolute pointers updated to avoid duplicating every replaced buffer.
"""
import os
import struct
import math
import bpy
import mathutils


MTFCORMAT_4 = mathutils.Matrix((
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


def _read_source(path, slot, x64=False):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8:
        raise ValueError("Source LMT is too small")
    if data[:4] != b"LMT\0":
        raise ValueError("Only little-endian LMT files are supported by this safe exporter")
    version, count = struct.unpack_from("<HH", data, 4)
    if version > 90:
        raise ValueError("LMT version %d needs the newer pointer layout; export cancelled to avoid a game crash" % version)
    if x64:
        raise ValueError("64-bit pointer LMT is not supported by this exporter yet")
    if slot < 0 or slot >= count:
        raise ValueError("Slot %d is outside the source LMT's %d slots" % (slot, count))
    table_start = 8
    table_end = table_start + count * 4
    if table_end > len(data):
        raise ValueError("LMT offset table is truncated")
    offsets = list(struct.unpack_from("<%dI" % count, data, table_start))
    if not offsets[slot]:
        raise ValueError("Source slot %d is empty; choose a populated slot" % slot)
    # Parse the selected block directly from the physical slot pointer. Do not
    # use LMTHeader.Offsettable here because that parser intentionally skips empty slots.
    block_offset, bone_count, frame_count, loop_frame = struct.unpack_from("<IIIi", data, offsets[slot])
    if not frame_count or frame_count > 100000:
        raise ValueError("Implausible source frame count: %d" % frame_count)
    if bone_count > 4096:
        raise ValueError("Implausible source track count: %d" % bone_count)
    def_size = 36 if version > 55 else 32
    if block_offset < table_end or block_offset + bone_count * def_size > len(data):
        raise ValueError("Source animation block has invalid track-definition offsets")
    defs = []
    cursor = block_offset
    for _ in range(bone_count):
        raw = data[cursor:cursor + def_size]
        if len(raw) != def_size:
            raise ValueError("Truncated source track definition")
        cursor += def_size
        bt, usage, joint, bone = struct.unpack_from("<4B", raw, 0)
        weight, bufsize, bufoff = struct.unpack_from("<fII", raw, 4)
        ref = struct.unpack_from("<4f", raw, 16)
        boundsoff = struct.unpack_from("<I", raw, 32)[0] if version > 55 else 0
        if bufsize and (bufoff > len(data) or bufsize > len(data) - bufoff):
            raise ValueError("Source track buffer is outside the file")
        defs.append({"raw": raw, "type": bt, "usage": usage, "joint": joint,
                     "bone": bone, "weight": weight, "size": bufsize,
                     "offset": bufoff, "ref": ref, "bounds": boundsoff})
    return data, version, count, offsets, slot, frame_count, loop_frame, defs


def _bone_by_mtf_index(arm, bone_id):
    # Preserve the addon's MHGU/DMC4SE remap semantics.
    remap = list(arm.get("mtf_remaptable", []))
    rootid = int(arm.get("mtf_lmt_rootid", 255))
    if bone_id == rootid or bone_id == 255:
        game_root = arm.get("mtf_game_root_bone")
        if game_root and arm.pose.bones.get(game_root):
            return arm.pose.bones.get(game_root)
        root_name = arm.get("mtf_root_bone", "bone255")
        if arm.pose.bones.get(root_name):
            return arm.pose.bones.get(root_name)
    idx = bone_id
    if remap and 0 <= bone_id < len(remap):
        idx = int(remap[bone_id])
        if idx == 255:
            return None
    for pb in arm.pose.bones:
        try:
            if int(pb.bone.get("mtf_index", -1)) == idx:
                return pb
        except (TypeError, ValueError):
            pass
    return None


def _resolve_track_bone(arm, bone_id, usage, action, selected_bones=None):
    """Resolve a track to a pose bone, including renamed root bones in selected-only mode."""
    pb = _bone_by_mtf_index(arm, bone_id)
    if pb is not None and _action_has_curve(action, pb.name, usage):
        return pb
    # Root motion is BoneID 255 / Usage 4. Older .blend files may not retain
    # mtf_root_bone metadata, or the synthetic bone may have been renamed.
    # In selected-only mode, an explicitly selected bone with keyed Location
    # is an unambiguous user override for this root track.
    if bone_id == 255 and usage == 4 and selected_bones:
        candidates = [
            arm.pose.bones.get(name) for name in selected_bones
            if arm.pose.bones.get(name) is not None
            and _action_has_curve(action, name, usage)
        ]
        if len(candidates) == 1:
            return candidates[0]
    return pb


def _action_has_curve(action, bone_name, usage):
    if action is None:
        return False
    if not hasattr(action, "fcurves"):
        return True  # Blender layered actions: use evaluated pose sampling.
    if usage in (0, 3):
        paths = {"rotation_quaternion"}
    elif usage in (1, 4):
        paths = {"location"}
    elif usage == 2:
        paths = {"scale"}
    else:
        return False
    prefix = 'pose.bones["%s"]' % bone_name.replace('\\', '\\\\').replace('"', '\\"')
    return any(fc.data_path.startswith(prefix) and fc.data_path.split(".")[-1] in paths
               for fc in action.fcurves)


def _q_to_mt(q, is_root):
    q = q.copy().normalized()
    if is_root:
        # Exact inverse of the confirmed import-side root alignment:
        # q_blender = C @ inverse(q_mt) @ inverse(C)
        # Therefore q_mt = inverse(C) @ inverse(q_blender) @ C.
        c = MTFCORMAT_4
        inverse = c.inverted() @ q.to_matrix().to_4x4().inverted() @ c
        return inverse.to_quaternion().normalized()
    # Child rotations are imported as parent-local quaternions without a
    # second global-basis conversion, so export preserves them as-is.
    return q


def _vector_to_mt(arm, pb, loc, is_root):
    if is_root:
        target = pb.bone.matrix_local.translation + (pb.bone.matrix_local.to_quaternion() @ loc)
        basis = pb.bone.matrix_local.to_3x3()
        # Imported root tracks use the synthetic bone's rest axes as a reference.
        return basis.inverted() @ target
    rest = pb.bone.matrix_local.copy()
    if pb.bone.parent:
        rest = pb.bone.parent.matrix_local.inverted() @ pb.bone.matrix_local
    target = rest.translation + (rest.to_quaternion() @ loc)
    return target


def _signed14(v):
    n = int(round(max(-1.999, min(1.999, float(v))) / 0.000244156))
    return n if n >= 0 else (0x3FFF + n)


def _encode_quat(q, delta):
    q = q.copy().normalized()
    w, x, y, z = (_signed14(q.w), _signed14(q.x), _signed14(q.y), _signed14(q.z))
    iv1 = (w & 0x3FFF) | ((z & 0x3FFF) << 14) | ((y & 0xF) << 28)
    iv2 = (y >> 4 & 0x3FF) | ((x & 0x3FFF) << 10) | ((max(0, min(255, delta)) & 0xFF) << 24)
    return struct.pack("<II", iv1, iv2)


def _encode_root_motion_position(values, loop_frame):
    """Encode BoneID 255 / Usage 4 using the native v56+ vector codec.

    In the supplied MHGU LMT, animated root-translation tracks use BufferType
    4 (16-bit quantized Vector3) or 5 (8-bit quantized Vector3), with a 32-byte
    Bounds record. The previous exporter wrote BufferType 3 and no Bounds
    pointer for every vector track; keep ordinary bone vectors on that path,
    but write root motion in the native quantized format.

    The ReferenceFrame position is sampled at the loop pivot when LoopFrame is
    positive, otherwise at frame 0. Native clips use ReferenceFrame.w = 0 for
    a positive loop pivot and 1 for LoopFrame 0 / -1.
    """
    if not values:
        raise ValueError("Root motion track has no sampled position values")

    vectors = [tuple(float(v[k]) for k in range(3)) for v in values]
    mins = [min(v[k] for v in vectors) for k in range(3)]
    maxs = [max(v[k] for v in vectors) for k in range(3)]
    # Native Bounds records use (offset + fraction * addin). A small non-zero
    # range is used for axes that do not move, matching existing LMT samples.
    addin = [max(maxs[k] - mins[k], 1.0e-5) for k in range(3)]
    offset = mins[:]
    bounds_data = struct.pack(
        "<8f", addin[0], addin[1], addin[2], 1.0e-5,
        offset[0], offset[1], offset[2], 0.0,
    )
    # Quantize against the float32 values that will actually be written to LMT.
    addin32 = struct.unpack_from("<4f", bounds_data, 0)
    offset32 = struct.unpack_from("<4f", bounds_data, 16)

    buffer = bytearray()
    for index, vector in enumerate(vectors):
        quantized = []
        for axis in range(3):
            frac = (vector[axis] - offset32[axis]) / addin32[axis]
            quantized.append(max(0, min(65535, int(round(frac * 65535.0)))))
        # BufferType 4 stores x/y/z ushort plus ushort frame delta. The final
        # key has delta 0; all preceding samples are one frame apart.
        delta = 1 if index < len(vectors) - 1 else 0
        buffer.extend(struct.pack("<4H", quantized[0], quantized[1], quantized[2], delta))

    try:
        loop_index = int(loop_frame)
    except (TypeError, ValueError):
        loop_index = -1
    loop_index = loop_index if 0 < loop_index < len(vectors) else 0
    reference_w = 0.0 if int(loop_frame) > 0 else 1.0
    reference = (vectors[loop_index][0], vectors[loop_index][1],
                 vectors[loop_index][2], reference_w)
    return bytes(buffer), bounds_data, reference


def _sample_track(scene, arm, pb, usage, frame_count, frame_start,
                  action_override=None):
    """Sample a pose-bone channel from the active Action or an override Action."""
    if arm.animation_data is None:
        raise ValueError("The target armature has no animation data")
    original_action = arm.animation_data.action
    changed_action = action_override is not None and action_override != original_action
    if changed_action:
        arm.animation_data.action = action_override
    vals = []
    prev_q = None
    try:
        try:
            bpy.context.view_layer.update()
        except Exception:
            pass
        for i in range(frame_count):
            scene.frame_set(frame_start + i)
            # Force dependency graph evaluation before reading pose values.
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
            if usage in (0, 3):
                q = pb.rotation_quaternion.copy().normalized()
                is_root = (pb.name == arm.get("mtf_game_root_bone") or
                           int(pb.bone.get("mtf_index", -1)) == 255)
                # The importer applies root-controller conversion to Usage 3
                # tracks as well as BoneID 255, so export must invert it for both.
                q = _q_to_mt(q, is_root or usage == 3)
                if prev_q is not None and prev_q.dot(q) < 0:
                    q.negate()
                prev_q = q.copy()
                vals.append(q)
            elif usage in (1, 4):
                loc = pb.location.copy()
                is_root = (pb.name == arm.get("mtf_game_root_bone") or
                           int(pb.bone.get("mtf_index", -1)) == 255)
                vals.append(_vector_to_mt(arm, pb, loc, is_root))
            elif usage == 2:
                vals.append(pb.scale.copy())
    finally:
        if changed_action:
            arm.animation_data.action = original_action
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
    return vals


def _action_frame_info(action):
    """Return (frame_count, first_frame) for an Action's current keyed range."""
    if action is None:
        return 1, 0
    try:
        frame_range = action.frame_range
        start = int(round(frame_range[0]))
        end = int(round(frame_range[1]))
    except Exception:
        start = int(round(getattr(action, "frame_start", 0)))
        end = int(round(getattr(action, "frame_end", start)))
    return max(1, end - start + 1), start


def _split_loop_target(target_action, animation_id, active_action):
    """Resolve a split pre-loop/loop Action pair without constraining it to the
    destination slot's old frame count or LoopFrame.

    MT Framework stores the pair as one LMT animation. The selected Action is
    the phase being edited; its companion contributes the other phase. The new
    LMT frame count and loop pivot are calculated from these two Actions.
    """
    if target_action is None:
        return None
    try:
        target_id = int(target_action.get("mtf_animation_index", -1))
    except (TypeError, ValueError):
        return None
    if target_id != int(animation_id):
        raise ValueError(
            "Selected Action '%s' maps to animation ID %d, not target ID %d. "
            "Refresh the animation list and choose the correct target."
            % (target_action.name, target_id, int(animation_id)))

    target_is_loop = bool(target_action.get("mtf_is_loop", False))
    target_is_full_loop = bool(target_action.get("mtf_full_loop", False))
    if target_is_full_loop:
        return None

    if target_is_loop:
        try:
            metadata_loop_frame = int(target_action.get("mtf_source_loop_frame", 0) or 0)
        except (TypeError, ValueError):
            metadata_loop_frame = 0
    else:
        try:
            metadata_loop_frame = int(target_action.get("mtf_loop_frame", 0) or 0)
        except (TypeError, ValueError):
            metadata_loop_frame = 0

    is_split_segment = metadata_loop_frame > 0 and (
        target_is_loop or int(target_action.get("mtf_loop_frame", 0) or 0) > 0
    )
    if not is_split_segment:
        return None

    if target_is_loop:
        companion_name = str(target_action.get("mtf_source_action", ""))
        companion = bpy.data.actions.get(companion_name) if companion_name else None
    else:
        companion = None
        for candidate in bpy.data.actions:
            if candidate == target_action:
                continue
            try:
                candidate_id = int(candidate.get("mtf_animation_index", -1))
            except (TypeError, ValueError):
                continue
            if (
                candidate_id == target_id
                and bool(candidate.get("mtf_is_loop", False))
                and not bool(candidate.get("mtf_full_loop", False))
                and str(candidate.get("mtf_source_action", "")) == target_action.name
            ):
                companion = candidate
                break

    if companion is None:
        phase = "loop" if target_is_loop else "pre-loop"
        other = "pre-loop" if target_is_loop else "loop"
        raise ValueError(
            "Cannot export the combined animation for '%s': its companion %s Action is missing. "
            "Keep both Actions imported, then export again."
            % (target_action.name, other))
    try:
        if int(companion.get("mtf_animation_index", -1)) != target_id:
            raise ValueError("The selected Action's companion maps to a different LMT animation ID")
    except (TypeError, ValueError) as exc:
        raise ValueError("The selected Action's companion metadata is invalid") from exc

    active_count, active_start = _action_frame_info(active_action)
    companion_count, companion_start = _action_frame_info(companion)
    if target_is_loop:
        pre_loop_count = companion_count
        loop_count = active_count
    else:
        pre_loop_count = active_count
        loop_count = companion_count
    total_count = pre_loop_count + loop_count
    if total_count <= 0 or total_count > 100000:
        raise ValueError("The combined pre-loop/loop animation has an invalid frame count")
    return {
        "target_is_loop": target_is_loop,
        "companion_action": companion,
        "companion_start": companion_start,
        "frame_count": total_count,
        "loop_frame": pre_loop_count,
        "active_start": active_start,
    }


def _write_lmt_updates(filepath, source, version, slot_count, offsets, target_slot,
                       target_defs_offset, target_track_count, updates,
                       frame_count, loop_frame):
    """Patch updated track buffers while reusing their old storage where possible.

    When a replacement buffer is longer than its original buffer, only the
    additional bytes are inserted at the end of that buffer. All known absolute
    LMT pointers (slot headers, definition tables, track buffers and bounds) are
    adjusted. This prevents each export from appending a second full copy of
    every edited track buffer to the end of the file.
    """
    def_size = 36 if version > 55 else 32
    populated = [(i, off) for i, off in enumerate(offsets) if off]
    slots = []
    track_records = []
    definition_ranges = []
    header_end = max((off + 0x40 for _, off in populated), default=0)

    for physical_slot, slot_offset in populated:
        if slot_offset + 16 > len(source):
            raise ValueError("Slot %d header is outside the source LMT" % physical_slot)
        defs_offset, track_count, slot_frames, slot_loop = struct.unpack_from(
            "<IIIi", source, slot_offset)
        if (not track_count or defs_offset < 8
                or defs_offset + track_count * def_size > len(source)):
            raise ValueError("Slot %d has an invalid track-definition table" % physical_slot)
        definition_ranges.append((defs_offset, defs_offset + track_count * def_size))
        slots.append({
            "physical_slot": physical_slot,
            "slot_offset": slot_offset,
            "defs_offset": defs_offset,
            "track_count": track_count,
            "frames": slot_frames,
            "loop": slot_loop,
        })
        for index in range(track_count):
            def_pos = defs_offset + index * def_size
            raw = source[def_pos:def_pos + def_size]
            if len(raw) != def_size:
                raise ValueError("Track definition is truncated in slot %d" % physical_slot)
            size, buffer_offset = struct.unpack_from("<II", raw, 8)
            bounds_offset = struct.unpack_from("<I", raw, 32)[0] if version > 55 else 0
            if size and (buffer_offset < 8 or buffer_offset + size > len(source)):
                raise ValueError("Track buffer in slot %d points outside the source LMT" % physical_slot)
            if bounds_offset and (bounds_offset < 8 or bounds_offset + 32 > len(source)):
                raise ValueError("Track bounds in slot %d point outside the source LMT" % physical_slot)
            track_records.append({
                "physical_slot": physical_slot,
                "index": index,
                "def_pos": def_pos,
                "size": size,
                "buffer_offset": buffer_offset,
                "bounds_offset": bounds_offset,
            })

    target_slot_info = next((item for item in slots if item["physical_slot"] == target_slot), None)
    if target_slot_info is None:
        raise ValueError("Target LMT slot %d is empty" % target_slot)
    if target_slot_info["defs_offset"] != target_defs_offset:
        raise ValueError("Target LMT layout changed while export was running")

    # Some LMTs reuse one track-definition table for multiple animation slots.
    # In that case a dict keyed only by definition address assigns the record to
    # whichever slot was visited last, causing a false "no longer maps" error.
    # More importantly, modifying that shared table/buffer in place would alter
    # the other animation too. Use copy-on-write for the selected slot: duplicate
    # only its definition table and write the changed track buffers at the end.
    # This preserves all other slots and avoids shifting/duplicating the whole LMT.
    target_defs_start_for_overlap = int(target_slot_info["defs_offset"])
    target_defs_end_for_overlap = (
        target_defs_start_for_overlap + int(target_slot_info["track_count"]) * def_size)
    shared_definition_slots = []
    for item in slots:
        if item["physical_slot"] == target_slot:
            continue
        other_start = int(item["defs_offset"])
        other_end = other_start + int(item["track_count"]) * def_size
        if other_start < target_defs_end_for_overlap and target_defs_start_for_overlap < other_end:
            shared_definition_slots.append(item)
    if shared_definition_slots:
        target_defs_start = int(target_slot_info["defs_offset"])
        target_defs_end = target_defs_start + int(target_slot_info["track_count"]) * def_size
        if target_defs_start < 8 or target_defs_end > len(source):
            raise ValueError("Shared target track-definition table is outside the source LMT")

        cloned_defs = bytearray(source[target_defs_start:target_defs_end])
        used_def_positions = set()
        out = bytearray(source)
        for upd in updates:
            def_pos = int(upd["def_pos"])
            relative = def_pos - target_defs_start
            if relative < 0 or relative >= len(cloned_defs) or relative % def_size:
                raise ValueError(
                    "Updated track definition 0x%X is outside the selected LMT slot's table"
                    % def_pos)
            if relative in used_def_positions:
                raise ValueError("Duplicate track update was generated for one LMT definition")
            used_def_positions.add(relative)

            # Append a private sample buffer; never overwrite a buffer that may
            # also be referenced by another slot using the shared definition.
            while len(out) % 4:
                out.append(0)
            new_buffer_offset = len(out)
            buffer = bytes(upd["buffer"])
            out.extend(buffer)

            bounds_offset = struct.unpack_from("<I", cloned_defs, relative + 32)[0] if version > 55 else 0
            bounds_data = upd.get("bounds_data")
            if bounds_data:
                while len(out) % 4:
                    out.append(0)
                bounds_offset = len(out)
                out.extend(bytes(bounds_data))

            tr = upd["track"]
            ref = upd["ref"]
            if version > 55:
                new_def = struct.pack(
                    "<4BfII4fI", upd["type"], upd["usage"], tr["joint"], tr["bone"],
                    tr["weight"], len(buffer), new_buffer_offset, *ref, bounds_offset)
            else:
                new_def = struct.pack(
                    "<4BfII4f", upd["type"], upd["usage"], tr["joint"], tr["bone"],
                    tr["weight"], len(buffer), new_buffer_offset, *ref)
            if len(new_def) != def_size:
                raise AssertionError("Internal track-definition size mismatch")
            cloned_defs[relative:relative + def_size] = new_def

        while len(out) % 4:
            out.append(0)
        cloned_defs_offset = len(out)
        out.extend(cloned_defs)
        struct.pack_into(
            "<IIIi", out, target_slot_info["slot_offset"],
            cloned_defs_offset, target_slot_info["track_count"], int(frame_count), int(loop_frame))
        with open(filepath, "wb") as fh:
            fh.write(out)
        return len(updates)

    records_by_def = {record["def_pos"]: record for record in track_records}
    insertions = []
    updated_by_def = {}
    for upd in updates:
        def_pos = upd["def_pos"]
        record = records_by_def.get(def_pos)
        if record is None or record["physical_slot"] != target_slot:
            raise ValueError("Updated track no longer maps to the selected LMT slot")
        if def_pos in updated_by_def:
            raise ValueError("Duplicate track update was generated for one LMT definition")
        updated_by_def[def_pos] = upd
        old_size = int(record["size"])
        old_offset = int(record["buffer_offset"])
        new_size = len(upd["buffer"])
        if old_size > 0 and new_size > old_size:
            insertions.append((old_offset + old_size, new_size - old_size))

    insertion_points = [pos for pos, _ in insertions]
    if len(insertion_points) != len(set(insertion_points)):
        raise ValueError("Source LMT has overlapping/aliased track buffers; safe resizing was cancelled")
    for insert_at, _delta in insertions:
        if insert_at < header_end:
            raise ValueError("Track buffer overlaps the LMT slot-header area; safe resizing was cancelled")
        for start, end in definition_ranges:
            if start < insert_at < end:
                raise ValueError("Track buffer overlaps a definition table; safe resizing was cancelled")
        for record in track_records:
            start = record["buffer_offset"]
            end = start + record["size"]
            if record["size"] and start < insert_at < end:
                raise ValueError("Source LMT contains overlapping track buffers; safe resizing was cancelled")
            bounds = record["bounds_offset"]
            if bounds and bounds < insert_at < bounds + 32:
                raise ValueError("Track buffer overlaps a bounds block; safe resizing was cancelled")

    out = bytearray(source)
    # Work from high offsets to low offsets so each insertion coordinate still
    # refers to the original source file.
    for insert_at, delta in sorted(insertions, reverse=True):
        out[insert_at:insert_at] = b"\0" * delta

    def remap_offset(value):
        value = int(value or 0)
        if value <= 0:
            return value
        return value + sum(delta for insert_at, delta in insertions if insert_at <= value)

    # If a source track has no original storage, append only that new buffer.
    appended_buffer_offsets = {}
    for upd in updates:
        record = records_by_def[upd["def_pos"]]
        if record["size"] == 0:
            while len(out) % 4:
                out.append(0)
            appended_buffer_offsets[upd["def_pos"]] = len(out)
            out.extend(upd["buffer"])

    appended_bounds_offsets = {}
    for upd in updates:
        if not upd.get("bounds_data"):
            continue
        record = records_by_def[upd["def_pos"]]
        if not record["bounds_offset"]:
            while len(out) % 4:
                out.append(0)
            appended_bounds_offsets[upd["def_pos"]] = len(out)
            out.extend(upd["bounds_data"])

    # Remap every known absolute pointer so other animation slots continue to
    # point to their original definitions, buffers and quantization bounds.
    for item in slots:
        slot_offset = item["slot_offset"]
        mapped_defs = remap_offset(item["defs_offset"])
        struct.pack_into("<I", out, slot_offset, mapped_defs)

    for record in track_records:
        mapped_def = remap_offset(record["def_pos"])
        mapped_buffer = remap_offset(record["buffer_offset"])
        struct.pack_into("<I", out, mapped_def + 12, mapped_buffer)
        if version > 55:
            mapped_bounds = remap_offset(record["bounds_offset"])
            struct.pack_into("<I", out, mapped_def + 32, mapped_bounds)

    # Replace each updated buffer at its original location (extended in place
    # when needed), then rewrite its definition with the exact new size.
    for upd in updates:
        tr = upd["track"]
        def_pos = upd["def_pos"]
        record = records_by_def[def_pos]
        if record["size"] == 0:
            new_buffer_offset = appended_buffer_offsets[def_pos]
        else:
            new_buffer_offset = remap_offset(record["buffer_offset"])
        buffer = upd["buffer"]
        out[new_buffer_offset:new_buffer_offset + len(buffer)] = buffer

        bounds_offset = 0
        bounds_data = upd.get("bounds_data")
        if bounds_data:
            if record["bounds_offset"]:
                bounds_offset = remap_offset(record["bounds_offset"])
            else:
                bounds_offset = appended_bounds_offsets[def_pos]
            out[bounds_offset:bounds_offset + len(bounds_data)] = bounds_data

        ref = upd["ref"]
        if version > 55:
            new_def = struct.pack(
                "<4BfII4fI", upd["type"], upd["usage"], tr["joint"], tr["bone"],
                tr["weight"], len(buffer), new_buffer_offset, *ref, bounds_offset)
        else:
            new_def = struct.pack(
                "<4BfII4f", upd["type"], upd["usage"], tr["joint"], tr["bone"],
                tr["weight"], len(buffer), new_buffer_offset, *ref)
        if len(new_def) != def_size:
            raise AssertionError("Internal track-definition size mismatch")
        mapped_def_pos = remap_offset(def_pos)
        out[mapped_def_pos:mapped_def_pos + def_size] = new_def

    # Frame count and LoopFrame belong to the exported Action, not the old slot.
    struct.pack_into(
        "<IIIi", out, target_slot_info["slot_offset"],
        remap_offset(target_slot_info["defs_offset"]),
        target_slot_info["track_count"], int(frame_count), int(loop_frame))

    with open(filepath, "wb") as fh:
        fh.write(out)
    return len(updates)



def _resolve_animation_id(source_lmt, animation_id):
    """Map a dense, user-facing animation ID to the physical LMT table slot.

    The importer omits empty offset-table entries, so its animation 0 is the
    first non-zero slot pointer, animation 1 is the second, and so on.
    """
    with open(source_lmt, "rb") as fh:
        header = fh.read(8)
        if len(header) < 8 or header[:4] != b"LMT\0":
            raise ValueError("Source LMT has an invalid signature or truncated header")
        _version, slot_count = struct.unpack_from("<HH", header, 4)
        table_size = slot_count * 4
        table = fh.read(table_size)
    if slot_count == 0 or len(table) != table_size:
        raise ValueError("Source LMT animation offset table is invalid or truncated")
    offsets = struct.unpack("<%dI" % slot_count, table)
    populated = [physical for physical, offset in enumerate(offsets) if offset]
    if animation_id < 0 or animation_id >= len(populated):
        raise ValueError(
            "Animation ID %d is out of range (valid populated animation IDs: 0..%d; %d clips found)."
            % (animation_id, len(populated) - 1, len(populated)))
    return populated[animation_id]


def export_lmt(filepath, arm_obj, scene, source_lmt, slot_index, x64=False,
               selected_bones=None, target_action_name=None):
    """Replace one animation in a copy of a native-layout LMT.

    The replacement may have a different frame count and LoopFrame from the
    destination slot. Split pre-loop/loop Actions are sampled as one animation.
    Compact/repacked containers remain deliberately rejected because relocating
    their overall slot layout has not been validated against the game parser.
    """
    if not source_lmt or not os.path.isfile(source_lmt):
        raise ValueError("Select a valid game-native LMT as Source LMT")
    if os.path.abspath(filepath) == os.path.abspath(source_lmt):
        raise ValueError("Output path must differ from the source LMT; the source is never overwritten")
    action = arm_obj.animation_data.action if arm_obj.animation_data else None
    if action is None:
        raise ValueError("The selected armature has no active Action")

    animation_id = int(slot_index)
    if animation_id < 0:
        raise ValueError("Animation ID must be 0 or greater")
    target_action = bpy.data.actions.get(target_action_name) if target_action_name else None
    if target_action_name and target_action is None:
        raise ValueError("Selected target Action '%s' no longer exists. Reopen Export LMT and choose it again." % target_action_name)
    if target_action is None:
        raise ValueError("Choose a named target animation in the Export LMT list")
    try:
        target_action_id = int(target_action.get("mtf_animation_index", -1))
    except (TypeError, ValueError):
        target_action_id = -1
    if target_action_id != animation_id:
        raise ValueError("Selected Action '%s' does not have valid MT Framework animation metadata" % target_action.name)
    # UI IDs are dense over populated animations; empty physical LMT slots are skipped.
    physical_slot = _resolve_animation_id(source_lmt, animation_id)
    source, version, slot_count, offsets, slot, source_frame_count, source_loop_frame, selected_defs = _read_source(
        source_lmt, physical_slot, x64=x64)
    split_target = _split_loop_target(target_action, animation_id, action)
    if split_target is not None:
        frame_count = int(split_target["frame_count"])
        loop_frame = int(split_target["loop_frame"])
        frame_start = int(split_target["active_start"])
    else:
        frame_count, frame_start = _action_frame_info(action)
        if bool(target_action.get("mtf_full_loop", False)):
            loop_frame = 0
        else:
            # A non-split Action without a loop marker is exported as a normal
            # non-looping clip. The destination slot's previous LoopFrame is not
            # allowed to constrain or silently change the replacement.
            loop_frame = -1
    if frame_count <= 0 or frame_count > 100000:
        raise ValueError("Selected Action has an invalid frame count")
    if version <= 50:
        raise ValueError("Safe export currently requires LMT version 51 or newer")
    def_size = 36 if version > 55 else 32
    populated = [(i, off) for i, off in enumerate(offsets) if off]
    for ordinal, (source_slot, source_offset) in enumerate(populated):
        expected_offset = 0x410 + ordinal * 0x40
        if source_offset != expected_offset:
            raise ValueError(
                "Source LMT uses a compact/repacked slot layout (slot %d points to 0x%X; "
                "expected native offset 0x%X). Do not use GU(6)_repacked_test.lmt as the "
                "physical source container; select the original game LMT or a native-layout "
                "single-track test file. This exporter will not relocate the whole container."
                % (source_slot, source_offset, expected_offset))
        if source_offset + 16 > len(source):
            raise ValueError("Slot %d header is outside the source LMT" % source_slot)
        defs_offset, track_count, slot_frames, slot_loop = struct.unpack_from(
            "<IIIi", source, source_offset)
        if not track_count or defs_offset < 8 or defs_offset + track_count * def_size > len(source):
            raise ValueError("Slot %d has an invalid track-definition table" % source_slot)
        if slot_frames <= 0 or slot_frames > 100000:
            raise ValueError("Slot %d has an implausible frame count" % source_slot)

    if slot not in [i for i, _ in populated]:
        raise ValueError("Target slot %d is empty" % slot)
    slot_offset = offsets[slot]
    defs_offset, track_count, original_slot_frames, original_slot_loop = struct.unpack_from(
        "<IIIi", source, slot_offset)
    tracks = []
    for index in range(track_count):
        pos = defs_offset + index * def_size
        raw = bytes(source[pos:pos + def_size])
        bt, usage, joint, bone = struct.unpack_from("<4B", raw, 0)
        weight, size, buf_off = struct.unpack_from("<fII", raw, 4)
        ref = struct.unpack_from("<4f", raw, 16)
        bounds = struct.unpack_from("<I", raw, 32)[0] if version > 55 else 0
        if size and (buf_off < 8 or buf_off + size > len(source)):
            raise ValueError("Track %d points outside the source LMT buffer" % index)
        if bounds and (bounds < 8 or bounds + 32 > len(source)):
            raise ValueError("Track %d has an invalid bounds pointer" % index)
        tracks.append({"index": index, "raw": raw, "type": bt, "usage": usage,
                       "joint": joint, "bone": bone, "weight": weight,
                       "size": size, "offset": buf_off, "ref": ref, "bounds": bounds})

    frame_count = max(1, min(int(frame_count), 100000))
    frame_start = max(0, int(frame_start))
    selected_bones = None if selected_bones is None else set(selected_bones)

    # Refuse likely accidental whole-rig export. Presence of keyframes alone is
    # not evidence that the user edited a track; imported Actions key most bones.
    if selected_bones is not None:
        supported = []
        for tr in tracks:
            if tr["usage"] not in (0, 1, 2, 3, 4):
                continue
            pb = _resolve_track_bone(
                arm_obj, tr["bone"], tr["usage"], action, selected_bones)
            if pb is not None and _action_has_curve(action, pb.name, tr["usage"]):
                supported.append(pb.name)
        selected_track_count = sum(1 for name in supported if name in selected_bones)
        if len(supported) >= 4 and selected_track_count / len(supported) > 0.75:
            raise ValueError(
                "Safety stop: %d of %d supported tracks are selected. Select only the edited "
                "pose bone(s) before exporting." % (selected_track_count, len(supported)))

    old_frame = scene.frame_current
    updates = []
    try:
        for tr in tracks:
            usage, bone_id = tr["usage"], tr["bone"]
            if usage not in (0, 1, 2, 3, 4):
                continue
            pb = _resolve_track_bone(
                arm_obj, bone_id, usage, action, selected_bones)
            if pb is None:
                continue
            if selected_bones is not None and pb.name not in selected_bones:
                continue
            if not _action_has_curve(action, pb.name, usage):
                continue

            if split_target is not None:
                loop_split = int(split_target["loop_frame"])
                companion_action = split_target["companion_action"]
                if not _action_has_curve(companion_action, pb.name, usage):
                    # Keep the old track only when this channel is absent from
                    # the companion segment; never substitute a partial clip.
                    continue
                active_start = int(split_target["active_start"])
                companion_start = int(split_target["companion_start"])
                if split_target["target_is_loop"]:
                    before = _sample_track(
                        scene, arm_obj, pb, usage, loop_split, companion_start,
                        action_override=companion_action)
                    after = _sample_track(
                        scene, arm_obj, pb, usage, frame_count - loop_split,
                        active_start, action_override=action)
                    vals = before + after
                else:
                    before = _sample_track(
                        scene, arm_obj, pb, usage, loop_split, active_start,
                        action_override=action)
                    after = _sample_track(
                        scene, arm_obj, pb, usage, frame_count - loop_split,
                        companion_start, action_override=companion_action)
                    vals = before + after
                if usage in (0, 3):
                    # Keep quaternion signs continuous across the spliced boundary.
                    for value_index in range(1, len(vals)):
                        if vals[value_index - 1].dot(vals[value_index]) < 0:
                            vals[value_index].negate()
            else:
                vals = _sample_track(scene, arm_obj, pb, usage, frame_count, frame_start)
            if not vals:
                continue
            bounds_data = None
            is_root_translation = (bone_id == 255 and usage == 4 and version > 55)
            if is_root_translation:
                # Root motion in native v67 samples is BoneID 255 / Usage 4,
                # stored as Type 4 + 32-byte Bounds, not generic float Type 3.
                buffer, bounds_data, ref = _encode_root_motion_position(vals, loop_frame)
                bt = 4
            elif usage in (0, 3):
                buffer = b"".join(_encode_quat(q, 1) for q in vals)
                first = vals[0]
                ref = (float(first.x), float(first.y), float(first.z), float(first.w))
                bt = 6
            else:
                buffer = b"".join(struct.pack("<3fI", float(v.x), float(v.y), float(v.z), 1)
                                   for v in vals)
                first = vals[0]
                ref = (float(first.x), float(first.y), float(first.z), 0.0)
                bt = 3
            updates.append({"track": tr, "def_pos": defs_offset + tr["index"] * def_size,
                            "type": bt, "usage": usage, "buffer": buffer,
                            "bounds_data": bounds_data, "ref": ref})
    finally:
        scene.frame_set(old_frame)

    if not updates:
        if selected_bones is not None:
            raise ValueError(
                "No supported keyed tracks found for selected bone(s): %s. "
                "For root motion, select the root bone that has Location keyframes "
                "and confirm the active Action contains those keyframes."
                % ", ".join(sorted(selected_bones)))
        raise ValueError("No supported keyed tracks found in the active Action")

    return _write_lmt_updates(
        filepath, source, version, slot_count, offsets, slot, defs_offset, track_count,
        updates, frame_count, loop_frame)

