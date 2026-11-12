# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservative in-place MT Framework LMT slot patcher.

Only edited track definitions are changed. New track buffers are appended to
an otherwise byte-for-byte copy of the source file. No slot headers, existing
buffers, bounds blocks, or unrelated slot pointers are compacted or relocated.
Supported edited tracks use native-compatible quaternion/vector codecs; root translation uses Type 4 + Bounds.
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
    """Resolve LMT bone IDs, including Bone 255 root tracks."""
    remap = list(arm.get("mtf_remaptable", []))
    rootid = int(arm.get("mtf_lmt_rootid", 255))

    # Root Bone must always be resolved independently of selection.
    # Match the same resolution order used by lmt_import.py.
    if bone_id == rootid or bone_id == 255:
        # 1. Explicit game root mapping.
        game_root = arm.get("mtf_game_root_bone")
        if game_root:
            pb = arm.pose.bones.get(game_root)
            if pb is not None:
                return pb

        # 2. Original synthetic root name.
        root_name = arm.get("mtf_root_bone", "bone255")
        pb = arm.pose.bones.get(root_name)
        if pb is not None:
            return pb

        # 3. Persistent MT Framework bone index.
        for pb in arm.pose.bones:
            try:
                if int(pb.bone.get("mtf_index", -1)) == 255:
                    return pb
            except (TypeError, ValueError):
                continue

        # 4. Compatibility fallback for older armatures.
        for pb in arm.pose.bones:
            if pb.bone.parent is None and not pb.bone.use_deform:
                return pb

        # Match importer's final fallback.
        bones = list(arm.pose.bones)
        if bones:
            return bones[0]

        return None

    # Normal bone remapping.
    idx = bone_id
    if remap and 0 <= bone_id < len(remap):
        idx = int(remap[bone_id])

        # 255 is reserved for the root track.
        if idx == 255:
            return None

    for pb in arm.pose.bones:
        try:
            if int(pb.bone.get("mtf_index", -1)) == idx:
                return pb
        except (TypeError, ValueError):
            continue

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


def _action_fcurves(action):
    """Collect keyframe curves from both legacy and layered Blender Actions."""
    if action is None:
        return []
    curves = []
    seen = set()

    def add_collection(collection):
        try:
            for fc in collection:
                key = (getattr(fc, "data_path", ""),
                       int(getattr(fc, "array_index", 0)), id(fc))
                if key not in seen:
                    seen.add(key)
                    curves.append(fc)
        except (TypeError, RuntimeError, ReferenceError):
            pass

    # Blender's legacy compatibility view (also works for older Blender files).
    try:
        add_collection(action.fcurves)
    except (AttributeError, TypeError, RuntimeError):
        pass

    # Blender 4.4+ layered Actions store F-Curves in a channelbag per slot.
    try:
        for layer in action.layers:
            for strip in layer.strips:
                for slot in action.slots:
                    try:
                        bag = strip.channelbag(slot)
                    except (AttributeError, TypeError, RuntimeError, ReferenceError):
                        bag = None
                    if bag is not None:
                        add_collection(bag.fcurves)
    except (AttributeError, TypeError, RuntimeError, ReferenceError):
        pass
    return curves


def _action_has_curve(action, bone_name, usage):
    if action is None:
        return False
    if usage in (0, 3):
        paths = {"rotation_quaternion"}
    elif usage in (1, 4):
        paths = {"location"}
    elif usage == 2:
        paths = {"scale"}
    else:
        return False
    prefix = 'pose.bones["%s"]' % bone_name.replace('\\', '\\\\').replace('"', '\\"')
    curves = _action_fcurves(action)
    # If this Blender build exposes no F-Curve API for the Action, let evaluated
    # pose sampling decide; an explicitly available but empty curve list is False.
    if not curves:
        has_curve_api = hasattr(action, "fcurves") or hasattr(action, "layers")
        return not has_curve_api
    return any(fc.data_path.startswith(prefix) and fc.data_path.split(".")[-1] in paths
               for fc in curves)


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
    """Sample one pose-bone channel from an explicit Action without leaving it active."""
    if arm.animation_data is None:
        raise ValueError("The target armature has no animation data")
    original_action = arm.animation_data.action
    original_slot = getattr(arm.animation_data, "action_slot", None)
    changed_action = action_override is not None and action_override != original_action
    if changed_action:
        arm.animation_data.action = action_override
        try:
            slots = getattr(action_override, "slots", None)
            if slots and hasattr(arm.animation_data, "action_slot"):
                arm.animation_data.action_slot = slots[0]
        except (AttributeError, TypeError, RuntimeError, ReferenceError):
            pass
    vals = []
    prev_q = None
    try:
        for i in range(frame_count):
            scene.frame_set(frame_start + i)
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
            if usage in (0, 3):
                q = pb.rotation_quaternion.copy().normalized()
                is_root = (pb.name == arm.get("mtf_game_root_bone") or
                           int(pb.bone.get("mtf_index", -1)) == 255)
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
                if hasattr(arm.animation_data, "action_slot"):
                    arm.animation_data.action_slot = original_slot
            except (AttributeError, TypeError, RuntimeError, ReferenceError):
                pass
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
    return vals


def _action_frame_info(action):
    """Return (frame_count, first_frame) spanning the first through last keyed frame."""
    coords = []
    curves = _action_fcurves(action)
    for curve in curves:
        try:
            for point in curve.keyframe_points:
                coords.append(float(point.co.x))
        except (AttributeError, TypeError, RuntimeError, ReferenceError):
            continue
    if coords:
        # LMT is frame-indexed. Floor/ceil ensures fractional Blender keys are
        # included rather than accidentally dropping the first or last frame.
        first = math.floor(min(coords))
        last = math.ceil(max(coords))
    else:
        try:
            frame_range = action.frame_range
            first = math.floor(float(frame_range[0]))
            last = math.ceil(float(frame_range[1]))
        except Exception:
            first = math.floor(float(getattr(action, "frame_start", 0)))
            last = math.ceil(float(getattr(action, "frame_end", first)))
    if last < first:
        raise ValueError("Selected Action has an invalid keyed frame range")
    return max(1, int(last - first + 1)), int(first)


def _find_split_companion_action(target_action, animation_id):
    """Find the other Blender Action that forms the same pre-loop/loop clip."""
    target_is_loop = bool(target_action.get("mtf_is_loop", False))
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
            if (candidate_id == int(animation_id)
                    and bool(candidate.get("mtf_is_loop", False))
                    and not bool(candidate.get("mtf_full_loop", False))
                    and str(candidate.get("mtf_source_action", "")) == target_action.name):
                companion = candidate
                break
    if companion is None:
        raise ValueError(
            "The linked PRE-LOOP / LOOP Action is missing. Keep both imported Actions in Blender and try again.")
    try:
        companion_id = int(companion.get("mtf_animation_index", -1))
    except (TypeError, ValueError):
        companion_id = -1
    if companion_id != int(animation_id):
        raise ValueError("The linked PRE-LOOP / LOOP Action belongs to a different LMT animation ID")
    target_owner = str(target_action.get("mtf_armature_name", "") or "")
    companion_owner = str(companion.get("mtf_armature_name", "") or "")
    if target_owner and companion_owner and target_owner != companion_owner:
        raise ValueError("The linked PRE-LOOP / LOOP Actions belong to different armatures")
    if bool(companion.get("mtf_full_loop", False)):
        raise ValueError("The linked Action is marked as a full loop, not a partial-loop companion")
    return companion




def _write_lmt_updates(filepath, source, version, slot_offset, defs_offset,
                       track_count, def_size, tracks, updates, frame_count,
                       loop_frame):
    """Append new buffers and a private cloned target definition table.

    Existing buffers, bounds, slot headers, and pointers for every other slot
    remain byte-for-byte in place. This avoids risky in-place buffer growth and
    absolute-pointer remapping.
    """
    out = bytearray(source)
    updated_by_index = {int(item["track"]["index"]): item for item in updates}
    if len(updated_by_index) != len(updates):
        raise ValueError("Duplicate track updates target the same definition")

    new_defs = bytearray()
    for tr in tracks:
        upd = updated_by_index.get(int(tr["index"]))
        if upd is None:
            new_defs.extend(tr["raw"])
            continue

        while len(out) % 4:
            out.append(0)
        buffer_offset = len(out)
        buffer = bytes(upd["buffer"])
        if not buffer:
            raise ValueError("Refusing to write an empty replacement track buffer")
        out.extend(buffer)

        bounds_offset = 0
        bounds_data = upd.get("bounds_data")
        if bounds_data:
            while len(out) % 4:
                out.append(0)
            bounds_offset = len(out)
            out.extend(bytes(bounds_data))

        ref = tuple(float(v) for v in upd["ref"])
        if version > 55:
            definition = struct.pack(
                "<4BfII4fI", upd["type"], upd["usage"], tr["joint"], tr["bone"],
                tr["weight"], len(buffer), buffer_offset, *ref, bounds_offset)
        else:
            definition = struct.pack(
                "<4BfII4f", upd["type"], upd["usage"], tr["joint"], tr["bone"],
                tr["weight"], len(buffer), buffer_offset, *ref)
        if len(definition) != def_size:
            raise AssertionError("Track-definition size mismatch")
        new_defs.extend(definition)

    if len(new_defs) != track_count * def_size:
        raise AssertionError("Cloned target definition table has an invalid size")
    while len(out) % 4:
        out.append(0)
    new_defs_offset = len(out)
    out.extend(new_defs)

    struct.pack_into(
        "<IIIi", out, slot_offset,
        new_defs_offset, int(track_count), int(frame_count), int(loop_frame))

    # Validate the new target slot and every rewritten pointer before touching disk.
    if new_defs_offset % 4:
        raise ValueError("Internal error: cloned definition table is not 4-byte aligned")
    for index in updated_by_index:
        p = new_defs_offset + index * def_size
        _bt, _usage, _joint, _bone = struct.unpack_from("<4B", out, p)
        _weight, size, buffer_pointer = struct.unpack_from("<fII", out, p + 4)
        if not size or buffer_pointer % 4 or buffer_pointer + size > len(out):
            raise ValueError("New track buffer pointer failed final validation")
        if version > 55:
            bounds_pointer = struct.unpack_from("<I", out, p + 32)[0]
            if bounds_pointer and (bounds_pointer % 4 or bounds_pointer + 32 > len(out)):
                raise ValueError("New track bounds pointer failed final validation")

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
    """Replace the selected imported Action in a copy of its source LMT.

    A replacement Action may have any keyed frame range. For a split partial-loop
    pair, only the selected phase comes from the selected Action; the companion
    phase is decoded again from Source LMT so edits in the other Blender Action
    cannot leak into this replacement.
    """
    if not source_lmt or not os.path.isfile(source_lmt):
        raise ValueError("Select a valid game-native LMT as Source LMT")
    if os.path.abspath(filepath) == os.path.abspath(source_lmt):
        raise ValueError("Output path must differ from the source LMT; the source is never overwritten")
    if arm_obj.animation_data is None:
        raise ValueError("The selected armature has no animation data")

    target_action = bpy.data.actions.get(target_action_name) if target_action_name else None
    if target_action is None:
        raise ValueError("Choose a named target Action in the Export LMT list")
    try:
        animation_id = int(target_action.get("mtf_animation_index", -1))
    except (TypeError, ValueError):
        animation_id = -1
    if animation_id != int(slot_index) or animation_id < 0:
        raise ValueError("Selected Action has invalid LMT animation metadata")
    action_owner = str(target_action.get("mtf_armature_name", "") or "")
    if action_owner and action_owner != arm_obj.name:
        raise ValueError(
            "Selected Action belongs to armature '%s', not '%s'. Choose an Action imported for the active rig."
            % (action_owner, arm_obj.name))

    physical_slot = _resolve_animation_id(source_lmt, animation_id)
    source, version, slot_count, offsets, slot, source_frame_count, source_loop_frame, _ = _read_source(
        source_lmt, physical_slot, x64=x64)
    if version <= 50:
        raise ValueError("Safe export currently requires LMT version 51 or newer")
    def_size = 36 if version > 55 else 32

    populated = [(i, off) for i, off in enumerate(offsets) if off]
    table_end = 8 + slot_count * 4
    # Validate slot headers and definition tables without assuming the native
    # 0x410 + N*0x40 layout. Some valid LMTs use compact slot headers; this
    # append-only writer does not move them, so their layout can be preserved.
    for source_slot, source_offset in populated:
        if source_offset < table_end or source_offset + 16 > len(source):
            raise ValueError("Slot %d has an invalid or truncated animation header" % source_slot)
        source_defs, source_tracks, source_frames, _source_loop = struct.unpack_from(
            "<IIIi", source, source_offset)
        if (source_tracks <= 0 or source_tracks > 4096
                or source_defs < table_end
                or source_defs + source_tracks * def_size > len(source)):
            raise ValueError("Slot %d has an invalid track-definition table" % source_slot)
        if source_frames <= 0 or source_frames > 100000:
            raise ValueError("Slot %d has an invalid frame count" % source_slot)

    slot_offset = offsets[physical_slot]
    if offsets.count(slot_offset) > 1:
        raise ValueError(
            "Target animation ID %d shares its slot header with another ID; "
            "independent replacement is unsafe." % animation_id)
    defs_offset, track_count, source_frame_count, source_loop_frame = struct.unpack_from(
        "<IIIi", source, slot_offset)
    if not track_count or defs_offset < 8 or defs_offset + track_count * def_size > len(source):
        raise ValueError("Target slot has an invalid track-definition table")
    if source_frame_count <= 0 or source_frame_count > 100000:
        raise ValueError("Target slot has an invalid source frame count")

    target_frame_count, target_frame_start = _action_frame_info(target_action)
    if target_frame_count <= 0 or target_frame_count > 100000:
        raise ValueError("Selected Action has an invalid keyed frame range")

    target_is_loop = bool(target_action.get("mtf_is_loop", False))
    target_is_full_loop = bool(target_action.get("mtf_full_loop", False))
    try:
        metadata_loop = int(target_action.get(
            "mtf_source_loop_frame" if target_is_loop else "mtf_loop_frame", 0) or 0)
    except (TypeError, ValueError):
        metadata_loop = 0
    is_split_segment = (not target_is_full_loop and metadata_loop > 0)

    companion_action = None
    companion_frame_count = 0
    companion_frame_start = 1
    if is_split_segment:
        if not (0 < source_loop_frame < source_frame_count):
            raise ValueError(
                "Selected Action is marked as a split-loop segment, but Source LMT has no valid partial LoopFrame")
        companion_action = _find_split_companion_action(target_action, animation_id)
        companion_frame_count, companion_frame_start = _action_frame_info(companion_action)
        if target_is_loop:
            pre_loop_count = companion_frame_count
            loop_count = target_frame_count
        else:
            pre_loop_count = target_frame_count
            loop_count = companion_frame_count
        frame_count = pre_loop_count + loop_count
        loop_frame = pre_loop_count
        if frame_count <= 0 or frame_count > 100000:
            raise ValueError("Combined PRE-LOOP + LOOP animation has an invalid frame count")
    else:
        frame_count = target_frame_count
        loop_frame = 0 if target_is_full_loop else -1

    tracks = []
    for index in range(track_count):
        pos = defs_offset + index * def_size
        raw = bytes(source[pos:pos + def_size])
        if len(raw) != def_size:
            raise ValueError("Target track definition is truncated")
        bt, usage, joint, bone = struct.unpack_from("<4B", raw, 0)
        weight, size, buffer_offset = struct.unpack_from("<fII", raw, 4)
        ref = struct.unpack_from("<4f", raw, 16)
        bounds = struct.unpack_from("<I", raw, 32)[0] if version > 55 else 0
        if size and (buffer_offset < 8 or buffer_offset + size > len(source)):
            raise ValueError("Track %d points outside the source LMT buffer" % index)
        if bounds and (bounds < 8 or bounds + 32 > len(source)):
            raise ValueError("Track %d has an invalid bounds pointer" % index)
        tracks.append({"index": index, "raw": raw, "type": bt, "usage": usage,
                       "joint": joint, "bone": bone, "weight": weight,
                       "size": size, "offset": buffer_offset, "ref": ref,
                       "bounds": bounds})

    selected_bones = None if selected_bones is None else set(selected_bones)
    timeline_changed = (frame_count != source_frame_count or loop_frame != source_loop_frame)
    if selected_bones is not None and timeline_changed:
        raise ValueError(
            "The clip duration or LoopFrame changes (frames %d -> %d, LoopFrame %d -> %d). "
            "Disable 'Export Selected Bones Only' so the whole clip can be replaced consistently."
            % (source_frame_count, frame_count, source_loop_frame, loop_frame))

    # A global frame-count / loop-pivot change affects every track in the slot.
    # Do not silently leave an old native buffer attached to the new timeline.
    # Require full keyed coverage before allowing variable-length replacement.
    if timeline_changed:
        incomplete = []
        for tr in tracks:
            usage = int(tr["usage"])
            if usage not in (0, 1, 2, 3, 4):
                incomplete.append("track %d (unsupported usage %d)" % (tr["index"], usage))
                continue
            pb = _resolve_track_bone(arm_obj, tr["bone"], usage, target_action)
            if pb is None or not _action_has_curve(target_action, pb.name, usage):
                incomplete.append("track %d / bone %d" % (tr["index"], tr["bone"]))
                continue
            if is_split_segment and not _action_has_curve(companion_action, pb.name, usage):
                incomplete.append("track %d / bone %d (missing companion phase)" %
                                  (tr["index"], tr["bone"]))
        if incomplete:
            preview = ", ".join(incomplete[:8])
            if len(incomplete) > 8:
                preview += ", ..."
            raise ValueError(
                "Cannot safely change this animation's duration because not every LMT track "
                "has editable keyframes in the required Action segment(s): %s. "
                "Keep the original duration or restore the missing track channels before export."
                % preview)

    original_action = arm_obj.animation_data.action
    old_frame = scene.frame_current
    updates = []
    try:
        supported_selected = []
        for tr in tracks:
            usage = int(tr["usage"])
            if usage not in (0, 1, 2, 3, 4):
                continue
            pb = _resolve_track_bone(
                arm_obj, tr["bone"], usage, target_action, selected_bones)
            if pb is None or not _action_has_curve(target_action, pb.name, usage):
                continue
            if selected_bones is None or pb.name in selected_bones:
                supported_selected.append(pb.name)
        if selected_bones is not None and not supported_selected:
            raise ValueError(
                "No supported tracks in Action '%s' match the selected pose bone(s)."
                % target_action.name)

        for tr in tracks:
            usage, bone_id = int(tr["usage"]), int(tr["bone"])
            if usage not in (0, 1, 2, 3, 4):
                continue
            pb = _resolve_track_bone(
                arm_obj, bone_id, usage, target_action, selected_bones)
            if pb is None:
                continue
            if selected_bones is not None and pb.name not in selected_bones:
                continue
            if not _action_has_curve(target_action, pb.name, usage):
                continue

            if is_split_segment:
                # Both Blender Actions are phases of ONE native LMT animation.
                # Export always recombines PRE-LOOP then LOOP, regardless of which
                # half is selected in the UI. Each phase uses its own first keyframe.
                if not _action_has_curve(companion_action, pb.name, usage):
                    # This track has no keyed channel in the companion Action;
                    # keep its original native data rather than inventing values.
                    continue
                companion_values = _sample_track(
                    scene, arm_obj, pb, usage, companion_frame_count,
                    companion_frame_start, action_override=companion_action)
                target_values = _sample_track(
                    scene, arm_obj, pb, usage, target_frame_count,
                    target_frame_start, action_override=target_action)
                if target_is_loop:
                    vals = companion_values + target_values
                else:
                    vals = target_values + companion_values
                if usage in (0, 3):
                    for value_index in range(1, len(vals)):
                        if vals[value_index - 1].dot(vals[value_index]) < 0:
                            vals[value_index].negate()
            else:
                vals = _sample_track(
                    scene, arm_obj, pb, usage, target_frame_count,
                    target_frame_start, action_override=target_action)

            if not vals:
                continue
            bounds_data = None
            is_root_translation = (bone_id == 255 and usage == 4 and version > 55)
            if is_root_translation:
                buffer, bounds_data, ref = _encode_root_motion_position(vals, loop_frame)
                buffer_type = 4
            elif usage in (0, 3):
                buffer = b"".join(_encode_quat(q, 1) for q in vals)
                first = vals[0]
                ref = (float(first.x), float(first.y), float(first.z), float(first.w))
                buffer_type = 6
            else:
                buffer = b"".join(
                    struct.pack("<3fI", float(v.x), float(v.y), float(v.z), 1)
                    for v in vals)
                first = vals[0]
                ref = (float(first.x), float(first.y), float(first.z), 0.0)
                buffer_type = 3
            updates.append({"track": tr, "type": buffer_type,
                            "usage": usage, "buffer": buffer,
                            "bounds_data": bounds_data, "ref": ref})
    finally:
        try:
            if arm_obj.animation_data is not None:
                arm_obj.animation_data.action = original_action
            scene.frame_set(old_frame)
            bpy.context.view_layer.update()
        except Exception:
            pass

    if not updates:
        raise ValueError("No supported keyed tracks found for the selected Action / pose bones")

    return _write_lmt_updates(
        filepath, source, version, slot_offset, defs_offset,
        track_count, def_size, tracks, updates, frame_count, loop_frame)

