"""Basic LMT exporter for Blender Actions.

Writes a little-endian MT Framework LMT v55 file using uncompressed key buffers.
The exporter is intentionally conservative: it exports the active Action as one
animation block and bakes keyed pose channels at integer frames.
"""
import math
import struct
import bpy
from mathutils import Quaternion


def _u8(v):
    return struct.pack("<B", max(0, min(255, int(v))))

def _u16(v):
    return struct.pack("<H", max(0, min(65535, int(v))))

def _i32(v):
    return struct.pack("<i", int(v))

def _u32(v):
    return struct.pack("<I", max(0, int(v)))

def _f(v):
    return struct.pack("<f", float(v))


def _action_range(action, scene):
    start = int(math.floor(getattr(action, "frame_start", scene.frame_start)))
    end = int(math.ceil(getattr(action, "frame_end", scene.frame_end)))
    if end < start:
        end = start
    return start, end


def _find_fcurves(action):
    # Blender 3.x/4.x compatibility.
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    result = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                result.extend(list(getattr(bag, "fcurves", [])))
    return result


def _curve_map(action):
    out = {}
    for fc in _find_fcurves(action):
        # Blender's pose channels use pose.bones["name"].property
        path = fc.data_path
        if not path.startswith('pose.bones["') or '"]' not in path:
            continue
        bone_name = path[len('pose.bones["'):path.index('"]')]
        prop = path[path.index('"].') + 3:]
        out.setdefault((bone_name, prop), {})[int(round(k.co.x))] = fc
    return out


def _sample_component(curves, frame, default=0.0):
    if not curves:
        return default
    # Evaluate the curve at the requested frame; keyframe insertion is not
    # required at every integer frame.
    return float(curves.evaluate(frame))


def _eval_curves(curves, frame, defaults):
    return [float(c.evaluate(frame)) if c else defaults[i]
            for i, c in enumerate(curves)]


def _get_track_curves(action):
    grouped = {}
    for fc in _find_fcurves(action):
        path = fc.data_path
        if not path.startswith('pose.bones["') or '"].' not in path:
            continue
        name = path[len('pose.bones["'):path.index('"]')]
        prop = path[path.index('"].') + 3:]
        grouped.setdefault((name, prop), {})[fc.array_index] = fc
    return grouped


def export_lmt(filepath, arm_obj, scene, version=55):
    if arm_obj is None or arm_obj.type != "ARMATURE":
        raise ValueError("Select an armature before exporting LMT.")
    ad = arm_obj.animation_data
    action = ad.action if ad else None
    if action is None:
        raise ValueError("The selected armature has no active Action.")
    if version != 55:
        raise ValueError("This exporter currently writes LMT version 55 only.")

    start, end = _action_range(action, scene)
    frame_count = max(1, end - start + 1)
    curves = _get_track_curves(action)
    # bone index -> channel/property -> array index -> FCurve
    by_bone = {}
    name_to_index = {}
    for bone in arm_obj.data.bones:
        idx = int(bone.get("mtf_index", -1))
        if idx >= 0:
            name_to_index[bone.name] = idx

    for (bone_name, prop), components in curves.items():
        bone_id = name_to_index.get(bone_name)
        if bone_id is None:
            # Imported synthetic root motion is often animated on the object,
            # not a pose bone; object-level root motion is not inferred here.
            continue
        if prop not in ("rotation_quaternion", "location", "scale"):
            continue
        usage = {"rotation_quaternion": 0, "location": 1, "scale": 2}[prop]
        by_bone.setdefault((bone_id, usage), {})[prop] = components

    tracks = []
    for (bone_id, usage), channel_data in sorted(by_bone.items()):
        prop = {0: "rotation_quaternion", 1: "location", 2: "scale"}[usage]
        components = channel_data[prop]
        if usage == 0:
            # LMT v55 BufferType 4 stores quaternion XYZ and reconstructs W.
            samples = []
            for frame in range(start, end + 1):
                qvals = _eval_curves([components.get(i) for i in range(4)],
                                     frame, [1.0, 0.0, 0.0, 0.0])
                q = Quaternion((qvals[0], qvals[1], qvals[2], qvals[3]))
                if q.length > 0:
                    q.normalize()
                if q.w < 0:
                    q.negate()
                samples.append((q.x, q.y, q.z))
            buffer_type = 4
        else:
            defaults = [0.0, 0.0, 0.0] if usage == 1 else [1.0, 1.0, 1.0]
            samples = [_eval_curves([components.get(i) for i in range(3)],
                                    frame, defaults)
                       for frame in range(start, end + 1)]
            # BufferType 3 is uncompressed Vector3 + relative-frame uint32.
            buffer_type = 3

        # Reduce to frames where the sampled values actually change, while
        # retaining the final frame so the animation has a defined end pose.
        keys = []
        previous = None
        for offset, value in enumerate(samples):
            changed = previous is None or any(abs(a - b) > 1e-5 for a, b in zip(value, previous))
            if changed or offset == len(samples) - 1:
                keys.append((offset, value))
                previous = value
        if not keys:
            keys = [(0, samples[0])]
        buf = bytearray()
        for ki, (frame_offset, value) in enumerate(keys):
            if usage == 0:
                buf.extend(_f(value[0]) + _f(value[1]) + _f(value[2]))
                if ki < len(keys) - 1:
                    buf.extend(_u32(max(1, keys[ki + 1][0] - frame_offset)))
            else:
                buf.extend(_f(value[0]) + _f(value[1]) + _f(value[2]))
                # BufferType 3 always carries a relative-frame uint32.
                delta = (keys[ki + 1][0] - frame_offset) if ki < len(keys) - 1 else 1
                buf.extend(_u32(max(1, delta)))
        ref = samples[0]
        if usage == 0:
            q = Quaternion((1.0, ref[0], ref[1], ref[2]))
            # ReferenceFrame stores x,y,z,w in this format.
            reference = (ref[0], ref[1], ref[2], math.sqrt(max(0.0, 1.0 - sum(v*v for v in ref))))
        else:
            reference = (ref[0], ref[1], ref[2], 0.0)
        tracks.append({
            "bone": bone_id, "usage": usage, "buffer_type": buffer_type,
            "buffer": bytes(buf), "reference": reference,
        })

    # v55 little-endian LMT layout: 8-byte file header, one animation offset;
    # 16-byte animation header; 36-byte key-buffer definitions.
    header_size = 8 + 4
    anim_header_size = 16
    keydef_size = 32
    block_offset = header_size
    defs_offset = block_offset + anim_header_size
    data_offset = defs_offset + keydef_size * len(tracks)
    cursor = data_offset
    definitions = bytearray()
    buffers = bytearray()
    for track in tracks:
        buf = track["buffer"]
        buffer_offset = cursor
        cursor += len(buf)
        # BufferType, Usage, JointType, BoneID, weight, BufferSize,
        # BufferOffset, ReferenceFrame[4].
        definitions.extend(_u8(track["buffer_type"]))
        definitions.extend(_u8(track["usage"]))
        definitions.extend(_u8(0))
        definitions.extend(_u8(track["bone"] if track["bone"] < 255 else 255))
        definitions.extend(_f(1.0))
        definitions.extend(_u32(len(buf)))
        definitions.extend(_u32(buffer_offset))
        definitions.extend(b"".join(_f(v) for v in track["reference"]))
        buffers.extend(buf)

    loop_frame = int(action.get("mtf_loop_frame", -1))
    if loop_frame < -1 or loop_frame >= frame_count:
        loop_frame = -1
    # LMT id is "LMT\\0" as little-endian bytes.
    out = bytearray(b"LMT\\x00")
    out.extend(_u16(version))
    out.extend(_u16(1))
    out.extend(_u32(block_offset))
    out.extend(_u32(defs_offset))
    out.extend(_u32(len(tracks)))
    out.extend(_u32(frame_count))
    out.extend(_i32(loop_frame))
    out.extend(definitions)
    out.extend(buffers)
    with open(filepath, "wb") as fh:
        fh.write(out)
    return len(tracks)
