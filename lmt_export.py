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
        raise ValueError("Destination LMT is too small")
    if data[:4] != b"LMT\0":
        raise ValueError("Only little-endian LMT files are supported by this safe exporter")
    version, count = struct.unpack_from("<HH", data, 4)
    if version > 90:
        raise ValueError("LMT version %d needs the newer pointer layout; export cancelled to avoid a game crash" % version)
    if x64:
        raise ValueError("64-bit pointer LMT is not supported by this exporter yet")
    if slot < 0 or slot >= count:
        raise ValueError("Slot %d is outside the destination LMT's %d slots" % (slot, count))
    table_start = 8
    table_end = table_start + count * 4
    if table_end > len(data):
        raise ValueError("LMT offset table is truncated")
    offsets = list(struct.unpack_from("<%dI" % count, data, table_start))
    if not offsets[slot]:
        raise ValueError("Destination slot %d is empty; choose a populated slot" % slot)
    block_offset, bone_count, frame_count, loop_frame = struct.unpack_from("<IIIi", data, offsets[slot])
    if not frame_count or frame_count > 100000:
        raise ValueError("Implausible destination frame count: %d" % frame_count)
    if bone_count > 4096:
        raise ValueError("Implausible destination track count: %d" % bone_count)
    def_size = 36 if version > 55 else 32
    if block_offset < table_end or block_offset + bone_count * def_size > len(data):
        raise ValueError("Destination animation block has invalid track-definition offsets")
    defs = []
    cursor = block_offset
    for _ in range(bone_count):
        raw = data[cursor:cursor + def_size]
        if len(raw) != def_size:
            raise ValueError("Truncated destination track definition")
        cursor += def_size
        bt, usage, joint, bone = struct.unpack_from("<4B", raw, 0)
        weight, bufsize, bufoff = struct.unpack_from("<fII", raw, 4)
        ref = struct.unpack_from("<4f", raw, 16)
        boundsoff = struct.unpack_from("<I", raw, 32)[0] if version > 55 else 0
        if bufsize and (bufoff > len(data) or bufsize > len(data) - bufoff):
            raise ValueError("Destination track buffer is outside the file")
        defs.append({"raw": raw, "type": bt, "usage": usage, "joint": joint,
                     "bone": bone, "weight": weight, "size": bufsize,
                     "offset": bufoff, "ref": ref, "bounds": boundsoff})
    return data, version, count, offsets, slot, frame_count, loop_frame, defs


def list_lmt_animations(filepath, x64=False):
    """Return populated destination slots using dense animation IDs.

    IDs follow the importer's convention: empty physical pointer entries are
    skipped, so ID 0 is the first populated slot.
    """
    if not filepath or not os.path.isfile(filepath):
        raise ValueError("Choose an existing destination .lmt file")
    if x64:
        raise ValueError("64-bit pointer LMT is not supported by this exporter yet")
    with open(filepath, "rb") as fh:
        data = fh.read()
    if len(data) < 8 or data[:4] != b"LMT\0":
        raise ValueError("Destination file has an invalid little-endian LMT header")
    version, slot_count = struct.unpack_from("<HH", data, 4)
    if version > 90:
        raise ValueError("LMT version %d needs the newer pointer layout, which is not supported" % version)
    table_end = 8 + slot_count * 4
    if table_end > len(data):
        raise ValueError("Destination LMT offset table is truncated")
    offsets = struct.unpack_from("<%dI" % slot_count, data, 8) if slot_count else ()
    def_size = 36 if version > 55 else 32
    items = []
    for physical_slot, offset in enumerate(offsets):
        if not offset:
            continue
        if offset < table_end or offset + 16 > len(data):
            raise ValueError("Destination slot %d has an invalid animation header" % physical_slot)
        defs_offset, track_count, frame_count, loop_frame = struct.unpack_from("<IIIi", data, offset)
        if (not track_count or track_count > 4096 or defs_offset < table_end
                or defs_offset + track_count * def_size > len(data)):
            raise ValueError("Destination slot %d has an invalid track-definition table" % physical_slot)
        if not frame_count or frame_count > 100000:
            raise ValueError("Destination slot %d has an invalid frame count" % physical_slot)
        items.append({
            "animation_index": len(items),
            "physical_slot": physical_slot,
            "frame_count": frame_count,
            "loop_frame": loop_frame,
            "is_split_segment": loop_frame > 0,
            "pre_loop_frames": loop_frame if loop_frame > 0 else 0,
            "loop_frames": frame_count - loop_frame if loop_frame > 0 else 0,
            "track_count": track_count,
        })
    return items


def _bone_by_mtf_index(arm, bone_id):
    """Resolve LMT bone IDs, including Bone 255 root tracks."""
    remap = list(arm.get("mtf_remaptable", []))
    rootid = int(arm.get("mtf_lmt_rootid", 255))
    if bone_id == rootid or bone_id == 255:
        game_root = arm.get("mtf_game_root_bone")
        if game_root:
            pb = arm.pose.bones.get(game_root)
            if pb is not None:
                return pb
        root_name = arm.get("mtf_root_bone", "bone255")
        pb = arm.pose.bones.get(root_name)
        if pb is not None:
            return pb
        for pb in arm.pose.bones:
            try:
                if int(pb.bone.get("mtf_index", -1)) == 255:
                    return pb
            except (TypeError, ValueError):
                continue
        for pb in arm.pose.bones:
            if pb.bone.parent is None and not pb.bone.use_deform:
                return pb
        bones = list(arm.pose.bones)
        return bones[0] if bones else None

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
            continue
    return None


def _resolve_track_bone(arm, bone_id, usage, action, selected_bones=None):
    """Resolve a track to a pose bone, including renamed root bones."""
    pb = _bone_by_mtf_index(arm, bone_id)
    if pb is not None and _action_has_curve(action, pb.name, usage):
        return pb
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

    try:
        add_collection(action.fcurves)
    except (AttributeError, TypeError, RuntimeError):
        pass
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
    if not curves:
        has_curve_api = hasattr(action, "fcurves") or hasattr(action, "layers")
        return not has_curve_api
    return any(fc.data_path.startswith(prefix) and fc.data_path.split(".")[-1] in paths
               for fc in curves)


def _q_to_mt(q, is_root):
    q = q.copy().normalized()
    if is_root:
        c = MTFCORMAT_4
        inverse = c.inverted() @ q.to_matrix().to_4x4().inverted() @ c
        return inverse.to_quaternion().normalized()
    return q


def _vector_to_mt(arm, pb, loc, is_root):
    if is_root:
        target = pb.bone.matrix_local.translation + (pb.bone.matrix_local.to_quaternion() @ loc)
        basis = pb.bone.matrix_local.to_3x3()
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
    if not values:
        raise ValueError("Root motion track has no sampled position values")
    vectors = [tuple(float(v[k]) for k in range(3)) for v in values]
    mins = [min(v[k] for v in vectors) for k in range(3)]
    maxs = [max(v[k] for v in vectors) for k in range(3)]
    addin = [max(maxs[k] - mins[k], 1.0e-5) for k in range(3)]
    offset = mins[:]
    bounds_data = struct.pack(
        "<8f", addin[0], addin[1], addin[2], 1.0e-5,
        offset[0], offset[1], offset[2], 0.0,
    )
    addin32 = struct.unpack_from("<4f", bounds_data, 0)
    offset32 = struct.unpack_from("<4f", bounds_data, 16)
    buffer = bytearray()
    for index, vector in enumerate(vectors):
        quantized = []
        for axis in range(3):
            frac = (vector[axis] - offset32[axis]) / addin32[axis]
            quantized.append(max(0, min(65535, int(round(frac * 65535.0)))))
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


def _sample_track(scene, arm, pb, usage, frame_count, frame_start, action_override=None):
    """Sample a single keyed channel from an explicit Action."""
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
    for curve in _action_fcurves(action):
        try:
            for point in curve.keyframe_points:
                coords.append(float(point.co.x))
        except (AttributeError, TypeError, RuntimeError, ReferenceError):
            continue
    if coords:
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
        raise ValueError("Selected source Action has an invalid keyed frame range")
    return max(1, int(last - first + 1)), int(first)


def _find_split_companion_action(source_action, source_animation_id):
    """Find PRE/LOOP companion using the source Action's own source animation ID."""
    source_is_loop = bool(source_action.get("mtf_is_loop", False))
    if source_is_loop:
        companion_name = str(source_action.get("mtf_source_action", ""))
        companion = bpy.data.actions.get(companion_name) if companion_name else None
    else:
        companion = None
        for candidate in bpy.data.actions:
            if candidate == source_action:
                continue
            try:
                candidate_id = int(candidate.get("mtf_animation_index", -1))
            except (TypeError, ValueError):
                continue
            if (candidate_id == int(source_animation_id)
                    and bool(candidate.get("mtf_is_loop", False))
                    and not bool(candidate.get("mtf_full_loop", False))
                    and str(candidate.get("mtf_source_action", "")) == source_action.name):
                companion = candidate
                break
    if companion is None:
        raise ValueError("The linked PRE-LOOP / LOOP source Action is missing. Keep both imported Actions in Blender.")
    try:
        companion_id = int(companion.get("mtf_animation_index", -1))
    except (TypeError, ValueError):
        companion_id = -1
    if companion_id != int(source_animation_id):
        raise ValueError("The linked PRE-LOOP / LOOP Actions belong to a different source animation ID")
    source_owner = str(source_action.get("mtf_armature_name", "") or "")
    companion_owner = str(companion.get("mtf_armature_name", "") or "")
    if source_owner and companion_owner and source_owner != companion_owner:
        raise ValueError("The linked PRE-LOOP / LOOP Actions belong to different armatures")
    if bool(companion.get("mtf_full_loop", False)):
        raise ValueError("The linked Action is marked as a full loop, not a partial-loop companion")
    return companion


class AnimationClipTrack:
    """LMT-independent sampled track data.

    `bone_id` and `usage` are logical channel identifiers; `values` contains
    evaluated values for each clip frame. No LMT file pointers or offsets live
    in this object.
    """
    __slots__ = ("bone_id", "usage", "values")

    def __init__(self, bone_id, usage, values):
        self.bone_id = int(bone_id)
        self.usage = int(usage)
        self.values = list(values)


class AnimationClip:
    """LMT-independent animation payload ready for any destination adapter."""
    def __init__(self, frame_count, loop_frame, tracks):
        self.frame_count = int(frame_count)
        self.loop_frame = int(loop_frame)
        self.tracks = list(tracks)
        self.track_map = {(track.bone_id, track.usage): track for track in self.tracks}


def _lmt_ids_for_pose_bone(arm, pose_bone):
    """Reverse the armature's import remap so clip tracks are keyed by LMT bone ID."""
    try:
        mtf_index = int(pose_bone.bone.get("mtf_index", -1))
    except (TypeError, ValueError):
        mtf_index = -1
    root_id = int(arm.get("mtf_lmt_rootid", 255))
    if mtf_index == 255 or pose_bone.name == arm.get("mtf_root_bone"):
        return sorted(set((255, root_id)))
    remap = list(arm.get("mtf_remaptable", []))
    if remap:
        ids = []
        for index, mapped in enumerate(remap):
            try:
                if int(mapped) == mtf_index:
                    ids.append(index)
            except (TypeError, ValueError):
                continue
        return ids
    return [mtf_index] if mtf_index >= 0 else []


def _build_animation_clip(scene, arm, source_action, companion_action,
                          source_is_loop, frame_count, loop_frame,
                          source_frame_count, source_frame_start,
                          companion_frame_count, companion_frame_start,
                          selected_bones=None):
    """Sample Blender Actions into an independent AnimationClip.

    The builder is deliberately unaware of destination file paths, slot
    offsets, codecs, and pointer layouts. This is the future reuse point for
    standalone track export and other animation sources.
    """
    selected_bones = None if selected_bones is None else set(selected_bones)
    tracks = []
    for pb in arm.pose.bones:
        if selected_bones is not None and pb.name not in selected_bones:
            continue
        bone_ids = _lmt_ids_for_pose_bone(arm, pb)
        if not bone_ids:
            continue
        is_root = (pb.name == arm.get("mtf_game_root_bone")
                   or int(pb.bone.get("mtf_index", -1)) == 255
                   or pb.name == arm.get("mtf_root_bone"))
        usages = []
        if _action_has_curve(source_action, pb.name, 0):
            usages.append(0)
        if _action_has_curve(source_action, pb.name, 3):
            usages.append(3)
        if _action_has_curve(source_action, pb.name, 2):
            usages.append(2)
        if _action_has_curve(source_action, pb.name, 1):
            usages.append(1)
        if is_root and _action_has_curve(source_action, pb.name, 4):
            usages.append(4)
        elif is_root and _action_has_curve(source_action, pb.name, 1):
            # Older imports may store root translation with Usage 1.
            usages.append(4)
        for usage in sorted(set(usages)):
            if companion_action is not None:
                if not _action_has_curve(companion_action, pb.name, usage):
                    continue
                companion_values = _sample_track(
                    scene, arm, pb, usage, companion_frame_count,
                    companion_frame_start, action_override=companion_action)
                source_values = _sample_track(
                    scene, arm, pb, usage, source_frame_count,
                    source_frame_start, action_override=source_action)
                if source_is_loop:
                    values = companion_values + source_values
                else:
                    values = source_values + companion_values
                if usage in (0, 3):
                    for value_index in range(1, len(values)):
                        if values[value_index - 1].dot(values[value_index]) < 0:
                            values[value_index].negate()
            else:
                values = _sample_track(
                    scene, arm, pb, usage, source_frame_count,
                    source_frame_start, action_override=source_action)
            if not values:
                continue
            for bone_id in bone_ids:
                tracks.append(AnimationClipTrack(bone_id, usage, values))
    clip = AnimationClip(frame_count, loop_frame, tracks)
    if clip.frame_count <= 0 or clip.frame_count > 100000:
        raise ValueError("Source AnimationClip has an invalid frame count")
    return clip


def _encode_clip_track(track_data, version, loop_frame):
    values = track_data.values
    usage = track_data.usage
    bone_id = track_data.bone_id
    bounds_data = None
    if bone_id == 255 and usage == 4 and version > 55:
        buffer, bounds_data, ref = _encode_root_motion_position(values, loop_frame)
        buffer_type = 4
    elif usage in (0, 3):
        buffer = b"".join(_encode_quat(q, 1) for q in values)
        first = values[0]
        ref = (float(first.x), float(first.y), float(first.z), float(first.w))
        buffer_type = 6
    else:
        buffer = b"".join(
            struct.pack("<3fI", float(v.x), float(v.y), float(v.z), 1)
            for v in values)
        first = values[0]
        ref = (float(first.x), float(first.y), float(first.z), 0.0)
        buffer_type = 3
    return {"type": buffer_type, "usage": usage, "buffer": buffer,
            "bounds_data": bounds_data, "ref": ref}


def _write_lmt_updates(filepath, source, version, slot_offset, defs_offset,
                       track_count, def_size, tracks, updates, frame_count,
                       loop_frame):
    """Destination adapter: append buffers and clone only the selected slot."""
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
    struct.pack_into("<IIIi", out, slot_offset,
                     new_defs_offset, int(track_count), int(frame_count), int(loop_frame))
    if new_defs_offset % 4:
        raise ValueError("Internal error: cloned definition table is not 4-byte aligned")
    for index in updated_by_index:
        p = new_defs_offset + index * def_size
        struct.unpack_from("<4B", out, p)
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


def _resolve_animation_id(lmt_path, animation_id):
    """Map dense animation ID to physical LMT slot (ignoring empty entries)."""
    items = list_lmt_animations(lmt_path)
    if animation_id < 0 or animation_id >= len(items):
        raise ValueError("Destination animation ID %d is out of range (found %d populated clips)"
                         % (animation_id, len(items)))
    return items[animation_id]["physical_slot"]


def export_lmt(filepath, arm_obj, scene, destination_lmt,
               destination_animation_id, source_action_name, x64=False,
               selected_bones=None):
    """Replace a destination slot using an independently selected Blender Action.

    The imported Action is the animation source. The destination LMT is used
    only as a container/schema for the target slot. Source and destination
    animation IDs are intentionally independent; the source LMT is not needed
    after Actions have been imported into Blender.
    """
    if not destination_lmt or not os.path.isfile(destination_lmt):
        raise ValueError("Choose a valid Destination LMT")
    if os.path.abspath(filepath) == os.path.abspath(destination_lmt):
        raise ValueError("Output path must differ from Destination LMT; the original is never overwritten")
    if arm_obj.animation_data is None:
        raise ValueError("The selected armature has no animation data")
    source_action = bpy.data.actions.get(source_action_name) if source_action_name else None
    if source_action is None:
        raise ValueError("Choose a Source Animation Action from the imported Actions list")
    try:
        source_animation_id = int(source_action.get("mtf_animation_index", -1))
    except (TypeError, ValueError):
        source_animation_id = -1
    if source_animation_id < 0:
        raise ValueError("Selected Source Action has no imported LMT animation metadata")
    action_owner = str(source_action.get("mtf_armature_name", "") or "")
    if action_owner and action_owner != arm_obj.name:
        raise ValueError(
            "Source Action belongs to armature '%s', not '%s'. Choose an Action imported for the active rig."
            % (action_owner, arm_obj.name))

    physical_slot = _resolve_animation_id(destination_lmt, int(destination_animation_id))
    destination, version, slot_count, offsets, _slot, _, _, _ = _read_source(
        destination_lmt, physical_slot, x64=x64)
    if version <= 50:
        raise ValueError("Safe export currently requires LMT version 51 or newer")
    def_size = 36 if version > 55 else 32
    populated = [(i, off) for i, off in enumerate(offsets) if off]
    table_end = 8 + slot_count * 4
    for physical, source_offset in populated:
        if source_offset < table_end or source_offset + 16 > len(destination):
            raise ValueError("Destination slot %d has an invalid or truncated animation header" % physical)
        source_defs, source_tracks, source_frames, _source_loop = struct.unpack_from(
            "<IIIi", destination, source_offset)
        if (source_tracks <= 0 or source_tracks > 4096
                or source_defs < table_end
                or source_defs + source_tracks * def_size > len(destination)):
            raise ValueError("Destination slot %d has an invalid track-definition table" % physical)
        if source_frames <= 0 or source_frames > 100000:
            raise ValueError("Destination slot %d has an invalid frame count" % physical)

    slot_offset = offsets[physical_slot]
    if offsets.count(slot_offset) > 1:
        raise ValueError(
            "Destination animation ID %d shares its slot header with another ID; replacement is unsafe."
            % int(destination_animation_id))
    defs_offset, track_count, destination_frame_count, destination_loop_frame = struct.unpack_from(
        "<IIIi", destination, slot_offset)
    if not track_count or defs_offset < 8 or defs_offset + track_count * def_size > len(destination):
        raise ValueError("Destination slot has an invalid track-definition table")
    if destination_frame_count <= 0 or destination_frame_count > 100000:
        raise ValueError("Destination slot has an invalid source frame count")

    source_frame_count, source_frame_start = _action_frame_info(source_action)
    if source_frame_count <= 0 or source_frame_count > 100000:
        raise ValueError("Selected Source Action has an invalid keyed frame range")
    source_is_loop = bool(source_action.get("mtf_is_loop", False))
    source_is_full_loop = bool(source_action.get("mtf_full_loop", False))
    try:
        source_loop_metadata = int(source_action.get(
            "mtf_source_loop_frame" if source_is_loop else "mtf_loop_frame", 0) or 0)
    except (TypeError, ValueError):
        source_loop_metadata = 0
    is_split_segment = (not source_is_full_loop and source_loop_metadata > 0)

    companion_action = None
    companion_frame_count = 0
    companion_frame_start = 1
    if is_split_segment:
        companion_action = _find_split_companion_action(source_action, source_animation_id)
        companion_frame_count, companion_frame_start = _action_frame_info(companion_action)
        if source_is_loop:
            pre_loop_count = companion_frame_count
            loop_count = source_frame_count
        else:
            pre_loop_count = source_frame_count
            loop_count = companion_frame_count
        frame_count = pre_loop_count + loop_count
        loop_frame = pre_loop_count
        if frame_count <= 0 or frame_count > 100000:
            raise ValueError("Combined source PRE-LOOP + LOOP animation has an invalid frame count")
    else:
        frame_count = source_frame_count
        loop_frame = 0 if source_is_full_loop else -1

    tracks = []
    for index in range(track_count):
        pos = defs_offset + index * def_size
        raw = bytes(destination[pos:pos + def_size])
        if len(raw) != def_size:
            raise ValueError("Destination track definition is truncated")
        bt, usage, joint, bone = struct.unpack_from("<4B", raw, 0)
        weight, size, buffer_offset = struct.unpack_from("<fII", raw, 4)
        ref = struct.unpack_from("<4f", raw, 16)
        bounds = struct.unpack_from("<I", raw, 32)[0] if version > 55 else 0
        if size and (buffer_offset < 8 or buffer_offset + size > len(destination)):
            raise ValueError("Destination track %d points outside the LMT buffer" % index)
        if bounds and (bounds < 8 or bounds + 32 > len(destination)):
            raise ValueError("Destination track %d has an invalid bounds pointer" % index)
        tracks.append({"index": index, "raw": raw, "type": bt, "usage": usage,
                       "joint": joint, "bone": bone, "weight": weight,
                       "size": size, "offset": buffer_offset, "ref": ref,
                       "bounds": bounds})

    selected_bones = None if selected_bones is None else set(selected_bones)
    timeline_changed = (frame_count != destination_frame_count
                        or loop_frame != destination_loop_frame)
    if selected_bones is not None and timeline_changed:
        raise ValueError(
            "The source clip duration or LoopFrame differs from the destination "
            "(frames %d -> %d, LoopFrame %d -> %d). Disable 'Export Selected Bones Only'."
            % (destination_frame_count, frame_count, destination_loop_frame, loop_frame))

    # Sampling changes the scene frame while evaluating Actions. Always restore
    # the user's active Action and current frame, including on validation errors.
    original_action = arm_obj.animation_data.action
    original_slot = getattr(arm_obj.animation_data, "action_slot", None)
    old_frame = scene.frame_current
    try:
        clip = _build_animation_clip(
            scene, arm_obj, source_action, companion_action, source_is_loop,
            frame_count, loop_frame, source_frame_count, source_frame_start,
            companion_frame_count, companion_frame_start, selected_bones)
    finally:
        try:
            arm_obj.animation_data.action = original_action
            if hasattr(arm_obj.animation_data, "action_slot"):
                arm_obj.animation_data.action_slot = original_slot
            scene.frame_set(old_frame)
            bpy.context.view_layer.update()
        except Exception:
            pass
    selected_bones = None if selected_bones is None else set(selected_bones)

    clip_tracks = clip.track_map
    if timeline_changed:
        incomplete = []
        for tr in tracks:
            usage = int(tr["usage"])
            if usage not in (0, 1, 2, 3, 4):
                incomplete.append("track %d (unsupported usage %d)" % (tr["index"], usage))
                continue
            if (tr["bone"], usage) not in clip_tracks:
                incomplete.append("track %d / bone %d / usage %d" %
                                  (tr["index"], tr["bone"], usage))
        if incomplete:
            preview = ", ".join(incomplete[:8])
            if len(incomplete) > 8:
                preview += ", ..."
            raise ValueError(
                "Cannot change the destination animation duration because the Source Action "
                "does not contain every destination track: %s" % preview)

    updates = []
    updated_bones = set()
    for tr in tracks:
        usage, bone_id = int(tr["usage"]), int(tr["bone"])
        if usage not in (0, 1, 2, 3, 4):
            continue
        if selected_bones is not None:
            pb = _resolve_track_bone(arm_obj, bone_id, usage, source_action, selected_bones)
            if pb is None or pb.name not in selected_bones:
                continue
        track_data = clip_tracks.get((bone_id, usage))
        if track_data is None:
            continue
        if len(track_data.values) != frame_count:
            raise ValueError("Source track bone %d usage %d has %d values; expected %d"
                             % (bone_id, usage, len(track_data.values), frame_count))
        encoded = _encode_clip_track(track_data, version, loop_frame)
        updates.append({"track": tr, **encoded})
        updated_bones.add(bone_id)

    if selected_bones is not None and not updates:
        raise ValueError("No supported source tracks match the selected pose bones")
    if not updates:
        raise ValueError("No destination tracks could be matched to the Source Animation")

    return _write_lmt_updates(
        filepath, destination, version, slot_offset, defs_offset,
        track_count, def_size, tracks, updates, frame_count, loop_frame)


