# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservative in-place MT Framework LMT slot patcher.

Only edited track definitions are changed. New track buffers are appended to
an otherwise byte-for-byte copy of the source file. No slot headers, existing
buffers, bounds blocks, or unrelated slot pointers are compacted or relocated.
Supported edited tracks use Type 6 quaternion and Type 3 vector codecs.
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


def _align(data, alignment=4):
    return data + (b"\0" * ((-len(data)) % alignment))


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
        # Inverse of lmt_import._root_controller_rotation:
        # blender = inverse(mt_matrix @ MTFCORMAT_4)
        return (MTFCORMAT_4.inverted() @ q.to_matrix().to_4x4().inverted()).to_quaternion().normalized()
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


def _sample_track(scene, arm, pb, usage, frame_count, frame_start):
    action = arm.animation_data.action if arm.animation_data else None
    vals = []
    prev_q = None
    for i in range(frame_count):
        scene.frame_set(frame_start + i)
        # force dependency graph evaluation before reading pose values
        try:
            bpy.context.view_layer.update()
        except Exception:
            pass
        if usage in (0, 3):
            q = pb.rotation_quaternion.copy().normalized()
            is_root = (pb.name == arm.get("mtf_game_root_bone") or
                       int(pb.bone.get("mtf_index", -1)) == 255)
            q = _q_to_mt(q, is_root)
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
    return vals



def _parse_block_tracks(data, slot_offset, version):
    """Read one animation slot and capture every raw buffer/bounds payload."""
    if slot_offset < 8 or slot_offset + 16 > len(data):
        raise ValueError("Animation slot header lies outside the LMT file")
    defs_offset, track_count, frame_count, loop_frame = struct.unpack_from("<IIIi", data, slot_offset)
    def_size = 36 if version > 55 else 32
    if track_count > 4096 or not frame_count or frame_count > 100000:
        raise ValueError("Implausible track/frame count in slot")
    if defs_offset < 8 or defs_offset + track_count * def_size > len(data):
        raise ValueError("Animation track definitions lie outside the LMT file")
    tracks = []
    for i in range(track_count):
        p = defs_offset + i * def_size
        raw = bytes(data[p:p + def_size])
        bt, usage, joint, bone = struct.unpack_from("<4B", raw, 0)
        buf_size, buf_off = struct.unpack_from("<II", raw, 8)
        bounds_off = struct.unpack_from("<I", raw, 32)[0] if version > 55 else 0
        if buf_size and (buf_off < 8 or buf_off + buf_size > len(data)):
            raise ValueError("Track buffer for slot points outside the LMT file")
        buffer = bytes(data[buf_off:buf_off + buf_size]) if buf_size else b""
        bounds = None
        if bounds_off:
            if bounds_off < 8 or bounds_off + 32 > len(data):
                raise ValueError("Track bounds for slot point outside the LMT file")
            bounds = bytes(data[bounds_off:bounds_off + 32])
        tracks.append({
            "raw": raw, "buffer": buffer, "bounds_data": bounds, "new": False,
            "weight": struct.unpack_from("<f", raw, 4)[0],
            "type": bt, "usage": usage, "joint": joint, "bone": bone,
        })
    return {"frame_count": frame_count, "loop_frame": loop_frame, "tracks": tracks}


def _write_repacked_lmt(filepath, source, version, slot_count, block_map):
    """Rebuild the full container with native-style grouped slot headers.

    The header table maps populated animation slots to a 0x40-stride header area.
    Each slot's definitions and payloads follow that area. Identical payload bytes
    are shared to avoid bloating the file and to retain the source's common aliasing.
    """
    def_size = 36 if version > 55 else 32
    out = bytearray(b"LMT\0" + struct.pack("<HH", version, slot_count))
    out.extend(b"\0" * (slot_count * 4))
    while len(out) < 0x410:
        out.append(0)

    slots = sorted(block_map)
    header_offsets = {slot: 0x410 + i * 0x40 for i, slot in enumerate(slots)}
    out.extend(b"\0" * (len(slots) * 0x40))
    for slot in slots:
        tail = block_map[slot].get("header_tail")
        if tail is None or len(tail) != 48:
            raise ValueError("Source slot header metadata is missing; use the repaired/native-layout LMT as source")
        extra = block_map[slot].get("header_extra")
        if not isinstance(extra, bytes) or not extra:
            raise ValueError("Source slot auxiliary metadata is missing; export cancelled to avoid data loss")
        row = header_offsets[slot]
        out[row + 16:row + 64] = tail
    buffer_offsets = {}
    bounds_offsets = {}

    for slot in slots:
        block = block_map[slot]
        tracks = block["tracks"]
        while len(out) % 4:
            out.append(0)
        defs_offset = len(out)
        defs_table_offset = len(out)
        out.extend(b"\0" * (len(tracks) * def_size))
        definitions = []

        for track in tracks:
            raw = track.get("raw")
            buffer = bytes(track.get("buffer", b"") or b"")
            bounds_data = track.get("bounds_data")
            buffer_offset = 0
            bounds_offset = 0

            if buffer:
                if buffer in buffer_offsets:
                    buffer_offset = buffer_offsets[buffer]
                else:
                    while len(out) % 4:
                        out.append(0)
                    buffer_offset = len(out)
                    out.extend(buffer)
                    buffer_offsets[buffer] = buffer_offset

            if bounds_data is not None:
                bounds_data = bytes(bounds_data)
                if bounds_data in bounds_offsets:
                    bounds_offset = bounds_offsets[bounds_data]
                else:
                    while len(out) % 4:
                        out.append(0)
                    bounds_offset = len(out)
                    out.extend(bounds_data)
                    bounds_offsets[bounds_data] = bounds_offset

            if track.get("new"):
                item = track
                ref = item["ref"]
                if version > 55:
                    definition = struct.pack(
                        "<4BfII4fI", int(item["type"]), int(item["usage"]),
                        int(item["joint"]), int(item["bone"]), float(item["weight"]),
                        len(buffer), buffer_offset, *[float(x) for x in ref], bounds_offset)
                else:
                    definition = struct.pack(
                        "<4BfII4f", int(item["type"]), int(item["usage"]),
                        int(item["joint"]), int(item["bone"]), float(item["weight"]),
                        len(buffer), buffer_offset, *[float(x) for x in ref])
            else:
                if raw is None or len(raw) != def_size:
                    raise ValueError("Invalid preserved track definition")
                definition = bytearray(raw)
                struct.pack_into("<II", definition, 8, len(buffer), buffer_offset)
                if version > 55:
                    struct.pack_into("<I", definition, 32, bounds_offset)
                definition = bytes(definition)

            if len(definition) != def_size:
                raise AssertionError("Internal track definition size mismatch")
            definitions.append(definition)

        for i, definition in enumerate(definitions):
            pos = defs_table_offset + i * def_size
            out[pos:pos + def_size] = definition
        struct.pack_into("<IIIi", out, header_offsets[slot], defs_offset, len(tracks),
                         int(block["frame_count"]), int(block["loop_frame"]))
        # Preserve the slot's 4-byte terminator or 21-track auxiliary record.
        # The header's +0x34 field is a pointer to that data, not a generic block end.
        while len(out) % 4:
            out.append(0)
        extra_offset = len(out)
        extra = block["header_extra"]
        out.extend(extra)
        struct.pack_into("<I", out, header_offsets[slot] + 0x34, extra_offset)
        struct.pack_into("<I", out, 8 + slot * 4, header_offsets[slot])

    while len(out) % 16:
        out.append(0)
    with open(filepath, "wb") as fh:
        fh.write(out)
    return len(out)


def export_lmt(filepath, arm_obj, scene, source_lmt, slot_index, x64=False,
               selected_bones=None):
    """Patch edited tracks in place while preserving the source LMT container.

    The source must have the game's native 0x40-stride slot-header layout.
    Compact/repacked containers are deliberately rejected: earlier tests showed
    that moving every slot's definitions and buffers caused severe in-game lag.
    Only changed definition records and newly appended buffers are written.
    """
    if not source_lmt or not os.path.isfile(source_lmt):
        raise ValueError("Select a valid game-native LMT as Source LMT")
    if os.path.abspath(filepath) == os.path.abspath(source_lmt):
        raise ValueError("Output path must differ from the source LMT; the source is never overwritten")
    action = arm_obj.animation_data.action if arm_obj.animation_data else None
    if action is None:
        raise ValueError("The selected armature has no active Action")

    source, version, slot_count, offsets, slot, frame_count, loop_frame, selected_defs = _read_source(
        source_lmt, int(slot_index), x64=x64)
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
        if source_slot == slot:
            frame_count, loop_frame = slot_frames, slot_loop

    if slot not in [i for i, _ in populated]:
        raise ValueError("Target slot %d is empty" % slot)
    slot_offset = offsets[slot]
    defs_offset, track_count, source_frame_count, source_loop_frame = struct.unpack_from(
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

    frame_count = max(1, min(int(source_frame_count), 100000))
    frame_start = int(round(action.frame_range[0])) if hasattr(action, "frame_range") else int(round(action.frame_start))
    frame_start = max(0, frame_start)
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
            updates.append({"track": tr, "type": bt, "usage": usage, "buffer": buffer,
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

    out = bytearray(source)
    for upd in updates:
        tr = upd["track"]
        buffer = upd["buffer"]
        while len(out) % 4:
            out.append(0)
        new_buffer_offset = len(out)
        out.extend(buffer)
        bounds_offset = 0
        bounds_data = upd.get("bounds_data")
        if bounds_data:
            while len(out) % 4:
                out.append(0)
            bounds_offset = len(out)
            out.extend(bounds_data)
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
        def_pos = defs_offset + tr["index"] * def_size
        out[def_pos:def_pos + def_size] = new_def

    with open(filepath, "wb") as fh:
        fh.write(out)
    return len(updates)

