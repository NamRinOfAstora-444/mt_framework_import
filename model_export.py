"""Standalone MT Framework model exporter.

This module is intentionally isolated from the import/parser implementation.
It creates new binary files from selected Blender meshes and does not require
an original MOD/MOD3 file.

Targets:
- MOD v212 (MHGU): skinned 4-influence vertex layout when a mesh armature is
  available; static fallback for meshes without an armature.
- MOD v230 (MHXX): static mesh layout.
- MOD3 v237 (Monster Hunter: World): native 64-bit MOD3 header and static
  one-UV vertex layout.

The writer is experimental. MHGU v212 emits skeleton, weights, two UV sets,
materials, group bounds, per-mesh bone bounds, and topology. MHXX/MOD3 remain
static-only; other advanced data such as morphs and multiple UV channels are
not serialized. Binary validation is not proof of in-game acceptance.
"""

import math
import os
import struct
import tempfile


class ModelExportError(RuntimeError):
    """Invalid scene data or an unsupported export condition."""


def _align(value, alignment=16):
    return value + ((alignment - value % alignment) % alignment)


def _finite(values, label):
    values = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in values):
        raise ModelExportError("Non-finite value while exporting %s." % label)
    return values


def _as_i8(value):
    return max(-127, min(127, int(round(float(value) * 127.0))))


def _fixed_name(name, length=128):
    raw = str(name or "default").encode("ascii", "replace")[:length - 1]
    return raw + b"\x00" * (length - len(raw))


def _get_loop_normal(mesh, loop_index):
    # Blender 4.x exposes corner_normals; older Blender versions expose loop.normal.
    try:
        n = mesh.corner_normals[loop_index].vector.copy()
    except Exception:
        n = mesh.loops[loop_index].normal.copy()
    if n.length > 1e-12:
        n.normalize()
    else:
        n = n.__class__((0.0, 0.0, 1.0))
    return n


def collect_blender_parts(objects, context):
    """Build material-separated static triangle parts from selected mesh objects.

    Each output vertex has engine-space position and normal plus one UV set.
    Vertices are split at UV and normal seams because MOD vertex buffers are
    per-vertex rather than per-loop.
    """
    import bpy

    mesh_objects = [obj for obj in objects if getattr(obj, "type", None) == "MESH"]
    if not mesh_objects:
        raise ModelExportError("Select at least one mesh object before exporting.")

    depsgraph = context.evaluated_depsgraph_get()
    parts = []
    material_names = []
    material_lookup = {}
    total_edge_keys = set()

    for obj in mesh_objects:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
            try:
                normal_matrix = eval_obj.matrix_world.to_3x3().inverted().transposed()
            except Exception:
                normal_matrix = eval_obj.matrix_world.to_3x3()

            by_material = {}
            for tri in mesh.loop_triangles:
                poly = mesh.polygons[tri.polygon_index]
                mat_slot = int(poly.material_index)
                bucket = by_material.setdefault(mat_slot, {"vertices": [], "faces": [], "vertex_map": {}})
                face = []
                for loop_index in tri.loops:
                    loop = mesh.loops[loop_index]
                    vertex_index = int(loop.vertex_index)
                    local_co = mesh.vertices[vertex_index].co
                    world_co = eval_obj.matrix_world @ local_co
                    # Inverse of the importer's coordinate conversion:
                    # Blender (x, y, z) -> MT Framework (x, z, -y).
                    pos = _finite((world_co.x, world_co.z, -world_co.y), "position")

                    normal_blender = normal_matrix @ _get_loop_normal(mesh, loop_index)
                    if normal_blender.length > 1e-12:
                        normal_blender.normalize()
                    normal = _finite((normal_blender.x, normal_blender.z, -normal_blender.y), "normal")
                    if uv_layer is not None:
                        uv = _finite((uv_layer[loop_index].uv.x, uv_layer[loop_index].uv.y), "UV")
                    else:
                        uv = (0.0, 0.0)

                    # Quantized key avoids tiny floating-point discrepancies
                    # producing duplicate vertices unnecessarily.
                    key = (vertex_index,) + tuple(round(v, 7) for v in uv + normal)
                    out_index = bucket["vertex_map"].get(key)
                    if out_index is None:
                        out_index = len(bucket["vertices"])
                        bucket["vertex_map"][key] = out_index
                        bucket["vertices"].append({"position": pos, "normal": normal, "uv": uv})
                    face.append(out_index)
                bucket["faces"].append(tuple(face))
                for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    total_edge_keys.add((obj.name, mat_slot, min(a, b), max(a, b)))

            for mat_slot in sorted(by_material):
                bucket = by_material[mat_slot]
                if not bucket["faces"]:
                    continue
                mat = mesh.materials[mat_slot] if mat_slot < len(mesh.materials) else None
                mat_name = mat.name if mat is not None else "%s_Material" % obj.name
                if mat_name not in material_lookup:
                    material_lookup[mat_name] = len(material_names)
                    material_names.append(mat_name)
                bucket.pop("vertex_map", None)
                parts.append({
                    "name": "%s_%s" % (obj.name, mat_name),
                    "vertices": bucket["vertices"],
                    "faces": bucket["faces"],
                    "material_name": mat_name,
                    "material_index": material_lookup[mat_name],
                    "object_index": mesh_objects.index(obj),
                })
        finally:
            eval_obj.to_mesh_clear()

    if not parts:
        raise ModelExportError("No triangle faces were found in the selected mesh objects.")
    if len(parts) > 65535:
        raise ModelExportError("The selected meshes contain too many material parts for this format.")
    if len(material_names) > 65535:
        raise ModelExportError("The selected meshes contain too many materials for this format.")
    for part in parts:
        if len(part["vertices"]) > 65535:
            raise ModelExportError(
                "Part '%s' has more than 65,535 exported vertices. Split it into smaller meshes." % part["name"])
    return parts, material_names, len(total_edge_keys)


def _vertex_bytes_mod(part, version):
    out = bytearray()
    for vertex in part["vertices"]:
        x, y, z = _finite(vertex["position"], "position")
        nx, ny, nz = (_as_i8(v) for v in _finite(vertex["normal"], "normal"))
        u, v = _finite(vertex["uv"], "UV")
        if version == 212:
            # MTF/5 hash d829702c: float3 position, signed-byte normal + pad,
            # reserved uint32, half-float UV. Importer flips V on read.
            out += struct.pack("<3f3bBI2e", x, y, z, nx, ny, nz, 0, 0, u, 1.0 - v)
        elif version == 230:
            # MTF/6 hash 82917009: float3 position, signed-byte normal + pad,
            # float2 UV. Importer flips V on read.
            out += struct.pack("<3f3bB2f", x, y, z, nx, ny, nz, 0, u, 1.0 - v)
        else:
            raise ModelExportError("Unsupported MOD target version %s." % version)
    return bytes(out)


def _mod_faces(part, version, vertex_sub=0):
    # The little-endian v212 and v230 import paths both consume triangle strips.
    # Each triangle is emitted as its own strip, using 0xFFFF as restart marker.
    # v230 indexes the shared vertex buffer globally; v212 stores a per-part
    # vertex range and indexes faces locally.
    indices = []
    for face_index, face in enumerate(part["faces"]):
        if face_index:
            indices.append(0xFFFF)
        if version == 230:
            indices.extend(vertex_sub + int(index) for index in face)
        else:
            indices.extend(int(index) for index in face)
    return indices


def _bone_block_mod():
    # Minimal one-bone root for a static mesh MOD. MOD's parser expects 24-byte
    # bone infos, a local and absolute 4x4 matrix (3 floats + pad per row), then
    # a 256-byte remap table. The vertex layout itself is unskinned.
    bone_info = struct.pack("<4B20x", 0, 255, 0, 0)
    identity_matrix = b"".join(
        struct.pack("<3fI", *row, 0)
        for row in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    )
    return bone_info + identity_matrix + identity_matrix + bytes(256)


def _mod_bounds60(parts):
    positions = [v["position"] for p in parts for v in p["vertices"]]
    mins = tuple(min(p[i] for p in positions) for i in range(3))
    maxs = tuple(max(p[i] for p in positions) for i in range(3))
    center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    # The 60-byte MOD section stores three center floats, min/max vec4 blocks,
    # and four opaque words. These defaults mirror the known MOD headers.
    return struct.pack("<3f4f4f4I", *center, *mins, 0.0, *maxs, 0.0, 1000, 3000, 0x80000001, 0)


def _mod_mesh_descriptor(part, part_id, vertex_offset, face_offset, face_count, version):
    # Exact 48-byte MOD v2 mesh descriptor layout used by the unchanged importer.
    stride = 24
    blocktype = 0xD829702C if version == 212 else 0x82917009
    first_word = 0xFFFF
    mat = int(part["material_index"]) * 16
    lod = 0xFF
    descriptor = struct.pack(
        "<HHBHBHBBIIIII",
        first_word, len(part["vertices"]), part_id & 0xFF, mat,
        lod, 0, stride, 0, 0, vertex_offset, blocktype, face_offset, face_count)
    descriptor += bytes(5)  # unused field before bone remap id
    descriptor += struct.pack("<B", 0)
    descriptor += bytes(10)
    if len(descriptor) != 48:
        raise ModelExportError("Internal MOD mesh descriptor size error (%d bytes)." % len(descriptor))
    return descriptor


def _write_mod(filepath, parts, material_names, vertex_ids, version):
    mesh_count = len(parts)
    material_count = len(material_names)
    if mesh_count > 65535 or material_count > 65535:
        raise ModelExportError("MOD supports at most 65,535 mesh parts and materials in this exporter.")
    if version not in (212, 230):
        raise ModelExportError("Unsupported MOD target version %s." % version)

    header_size = 64 if version == 212 else 68
    bounds = _mod_bounds60(parts)
    # GU's v212 has a four-byte word on either side of its 60-byte bounds
    # block. v230 stores the same bounds immediately after its 68-byte header.
    prefix = struct.pack("<I", 0) if version == 212 else b""
    suffix = struct.pack("<I", 60) if version == 212 else b""
    extra_header = prefix + bounds + suffix
    bones_offset = header_size + len(extra_header)
    bones = _bone_block_mod()
    group_offset = bones_offset + len(bones)
    texture_offset = group_offset
    material_blob = b"".join(_fixed_name(n) for n in material_names)
    mesh_offset = texture_offset + len(material_blob)
    mesh_table_size = mesh_count * 48
    vertex_offset = _align(mesh_offset + mesh_table_size, 16)

    vertex_blobs = []
    face_blobs = []
    descriptors = []
    vertex_byte_cursor = 0
    vertex_sub_cursor = 0
    face_cursor = 0
    total_index_values = 0
    total_vertices = 0
    padding_indices = 1 if version == 212 else 3

    for part_index, part in enumerate(parts):
        vb = _vertex_bytes_mod(part, version)
        vcount = len(part["vertices"])
        vertex_sub = 0 if version == 212 else vertex_sub_cursor
        vertex_relative_offset = vertex_byte_cursor if version == 212 else 0
        face_values = _mod_faces(part, version, vertex_sub=vertex_sub)
        if any(i < 0 or i > 65535 for i in face_values):
            raise ModelExportError("Part '%s' exceeds the 16-bit face index range." % part["name"])
        if any(i >= (vcount if version == 212 else vertex_sub + vcount)
               for i in face_values if i != 0xFFFF):
            raise ModelExportError("Part '%s' has an invalid triangle index." % part["name"])
        face_blob = struct.pack("<%dH" % len(face_values), *face_values) if face_values else b""
        descriptors.append(_mod_mesh_descriptor(
            part, part_index, vertex_relative_offset, face_cursor, len(face_values), version))
        # Descriptor fields vertexsub and vertexoffset have different meaning
        # per version. Rewrite these two fields without changing the descriptor layout.
        desc = bytearray(descriptors[-1])
        struct.pack_into("<I", desc, 12, vertex_sub)
        struct.pack_into("<I", desc, 16, vertex_relative_offset)
        descriptors[-1] = bytes(desc)

        vertex_blobs.append(vb)
        face_blobs.append(face_blob)
        # Existing MOD files include per-mesh index-buffer padding. v212 has
        # one 0xFFFF separator; v230 reserves three indices between descriptors.
        # For v230, use degenerate bridge indices to keep the shared index buffer
        # bounded without referring past a vertex range.
        if version == 212:
            pad_values = [0xFFFF]
        else:
            last_index = face_values[-1] if face_values else vertex_sub
            if part_index + 1 < mesh_count:
                first_index_next = vertex_sub + vcount
                pad_values = [last_index, last_index, first_index_next]
            else:
                # Last part has no following vertex range to bridge to.
                pad_values = [last_index, last_index, last_index]
        face_blobs.append(struct.pack("<%dH" % len(pad_values), *pad_values))

        vertex_byte_cursor += len(vb)
        vertex_sub_cursor += vcount
        face_cursor += len(face_values) + padding_indices
        total_index_values += len(face_values) + padding_indices
        total_vertices += vcount

    vertex_blob = b"".join(vertex_blobs)
    face_blob = b"".join(face_blobs)
    faces_offset = vertex_offset + len(vertex_blob)
    # v212 points its trailing pointer at a four-byte tail word; v230 commonly
    # leaves the corresponding pointer null. Preserve those version conventions.
    if version == 212:
        unk_offset = _align(faces_offset + len(face_blob), 16)
        tail = bytes(4)
    else:
        unk_offset = 0
        tail = b""

    # VertexIds is undocumented and the new standalone static mesh has no
    # separate vertex-ID/remap buffer, so write zero rather than guessing.
    header_fields = (b"MOD\x00", version, 0, 1, mesh_count, material_count,
                     total_vertices, total_index_values, 0, len(vertex_blob), 0)
    if version == 212:
        header = struct.pack(
            "<4sBBHHHIIIII I7I", *header_fields, 0, bones_offset, group_offset,
            texture_offset, mesh_offset, vertex_offset, faces_offset, unk_offset)
    else:
        header = struct.pack(
            "<4sBBHHHIIIII II7I", *header_fields, 0, 0, bones_offset, group_offset,
            texture_offset, mesh_offset, vertex_offset, faces_offset, unk_offset)
    if len(header) != header_size:
        raise ModelExportError("Internal MOD header size mismatch (%d instead of %d)." % (len(header), header_size))

    payload = bytearray(header)
    payload += extra_header
    if len(payload) != bones_offset:
        raise ModelExportError("Internal MOD bone offset mismatch.")
    payload += bones
    if len(payload) != group_offset:
        raise ModelExportError("Internal MOD group offset mismatch.")
    payload += material_blob
    payload += bytes(mesh_offset - len(payload))
    payload += b"".join(descriptors)
    payload += bytes(vertex_offset - len(payload))
    payload += vertex_blob
    payload += face_blob
    if version == 212:
        payload += bytes(unk_offset - len(payload))
    payload += tail

    _atomic_write(filepath, bytes(payload))
    return {"version": version, "meshes": mesh_count, "materials": material_count,
            "vertices": total_vertices, "triangles": sum(len(p["faces"]) for p in parts), "size": len(payload)}


def _mod3_vertex_bytes(part):
    out = bytearray()
    for vertex in part["vertices"]:
        x, y, z = _finite(vertex["position"], "position")
        nx, ny, nz = (_as_i8(v) for v in _finite(vertex["normal"], "normal"))
        u, v = _finite(vertex["uv"], "UV")
        # Native MHW MOD3 IANonSkin1UV block: position + normal + tangent + half UV.
        out += struct.pack("<3f4b4b2e", x, y, z, nx, ny, nz, 0,
                           127, 0, 0, 127, u, 1.0 - v)
    return bytes(out)


def _mod3_mesh_header(part, index, vertex_sub, face_offset):
    vertex_count = len(part["vertices"])
    indices_count = 3 * len(part["faces"])
    block_size = 24
    material_index = int(part["material_index"])
    # MOD3's mesh-part header is 80 bytes. UInt32 offsets/counters and the
    # bounding-box metadata are independent from the 64-bit file-level offsets.
    header = struct.pack(
        "<HHHHI HBB IIIIII BB HHH hh II I 4i",
        19, vertex_count, 0, material_index, 1, 1, block_size, 0,
        vertex_sub, 0, 0xA756F2F9, face_offset, indices_count, 0,
        0, 1, 1, vertex_sub, vertex_sub + vertex_count - 1,
        -1, -1, 0, 0x30000000, vertex_sub + vertex_count,
        0, 0, 0, 0)
    if len(header) != 80:
        raise ModelExportError("Internal MOD3 mesh descriptor size error (%d bytes)." % len(header))
    return header


def _mod3_bounding_box(part):
    positions = [v["position"] for v in part["vertices"]]
    mins = tuple(min(p[i] for p in positions) for i in range(3))
    maxs = tuple(max(p[i] for p in positions) for i in range(3))
    center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    radius = max((sum((p[i] - center[i]) ** 2 for i in range(3)) for p in positions), default=0.0) ** 0.5
    identity = (1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0)
    return struct.pack("<i12B3ff4f4f16f4f", 0, *([0xCD] * 12),
                       *center, radius, *mins, 0.0, *maxs, 0.0,
                       *identity, 0.0, 0.0, 0.0, 0.0)


def _write_mod3(filepath, parts, material_names, vertex_ids):
    mesh_count = len(parts)
    material_count = len(material_names)
    if mesh_count > 65535 or material_count > 65535:
        raise ModelExportError("MOD3 supports at most 65,535 mesh parts and materials in this exporter.")

    header_size = 320
    group_offset = header_size
    material_names_offset = group_offset
    material_blob = b"".join(_fixed_name(n) for n in material_names)
    mesh_offset = _align(material_names_offset + len(material_blob), 16)

    mesh_headers = []
    boxes = []
    vertex_blobs = []
    face_blobs = []
    v_sub = 0
    face_cursor = 0
    total_vertices = 0
    total_face_indices = 0
    for index, part in enumerate(parts):
        if len(part["vertices"]) > 65535:
            raise ModelExportError("MOD3 part '%s' exceeds 65,535 vertices." % part["name"])
        mesh_headers.append(_mod3_mesh_header(part, index, v_sub, face_cursor))
        boxes.append(_mod3_bounding_box(part))
        vertex_blobs.append(_mod3_vertex_bytes(part))
        global_faces = [v_sub + idx for face in part["faces"] for idx in face]
        if any(i < 0 or i > 65535 for i in global_faces):
            raise ModelExportError("MOD3 global face index exceeds 16-bit range; split the exported model.")
        face_blobs.append(struct.pack("<%dH" % len(global_faces), *global_faces) if global_faces else b"")
        v_sub += len(part["vertices"])
        face_cursor += len(global_faces)
        total_vertices += len(part["vertices"])
        total_face_indices += len(global_faces)

    mesh_headers_blob = b"".join(mesh_headers)
    bounding_blob = struct.pack("<I", len(boxes)) + b"".join(boxes)
    vertex_blob = b"".join(vertex_blobs)
    face_blob = b"".join(face_blobs)
    vertex_offset = mesh_offset + len(mesh_headers_blob) + len(bounding_blob)
    faces_offset = vertex_offset + len(vertex_blob)
    trail_offset = _align(faces_offset + len(face_blob), 4)

    all_positions = [v["position"] for p in parts for v in p["vertices"]]
    mins = tuple(min(p[i] for p in all_positions) for i in range(3))
    maxs = tuple(max(p[i] for p in all_positions) for i in range(3))
    center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    radius = max((sum((p[i] - center[i]) ** 2 for i in range(3)) for p in all_positions), default=0.0) ** 0.5
    boundaries = struct.pack("<12f", *center, radius, *mins, 0.0, *maxs, 0.0)
    float_segment = struct.pack(
        "<f2h8f8fi5f", 15000.0, 1, 5,
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1, 360.0, 60.0, 20.0, 10.0, 10.0)
    byte_segment = bytes([1] * 64)

    # Native MHW MOD3 file-level structure: 64-bit offsets, 320-byte header.
    header = struct.pack(
        "<4sBBHHHIIIII10Q",
        b"MOD\x00", 237, 0, 0, mesh_count, material_count,
        total_vertices, total_face_indices, vertex_ids, len(vertex_blob), 0,
        0, 0, 0, group_offset, material_names_offset, mesh_offset,
        vertex_offset, faces_offset, trail_offset, 0)
    header += boundaries + float_segment + byte_segment
    if len(header) != header_size:
        raise ModelExportError("Internal MOD3 header size mismatch (%d bytes)." % len(header))

    # Minimal trailing segment convention used by common MHW MOD3 exporters.
    trailing = bytes([0x04]) + bytes(23)
    payload = bytearray(header)
    payload += material_blob
    payload += bytes(mesh_offset - len(payload))
    payload += mesh_headers_blob
    payload += bounding_blob
    payload += vertex_blob
    payload += face_blob
    payload += bytes(trail_offset - len(payload))
    payload += trailing
    _atomic_write(filepath, bytes(payload))
    return {"version": 237, "meshes": mesh_count, "materials": material_count,
            "vertices": total_vertices, "triangles": total_face_indices // 3, "size": len(payload)}


def _atomic_write(filepath, payload):
    out_path = os.path.abspath(filepath)
    out_dir = os.path.dirname(out_path) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".mtf_model_export_", suffix=".tmp",
                                         dir=out_dir, delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
        os.replace(temporary, out_path)
    except OSError as exc:
        if temporary and os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass
        raise ModelExportError("Could not write output file: %s" % exc) from exc


def export_selected_meshes(filepath, objects, context, target):
    """Public exporter entry point used by the Blender operator."""
    parts, materials, vertex_ids = collect_blender_parts(objects, context)
    if target == "MOD_V212":
        return _write_mod(filepath, parts, materials, vertex_ids, 212)
    if target == "MOD_V230":
        return _write_mod(filepath, parts, materials, vertex_ids, 230)
    if target == "MOD3_V237":
        return _write_mod3(filepath, parts, materials, vertex_ids)
    raise ModelExportError("Unknown export target %r." % target)
# Helpers for skinned MHGU v212 output. Kept in the export module only.
import re as _re


def _find_export_armature(objects, context):
    for obj in objects:
        if getattr(obj, "type", None) == "ARMATURE":
            return obj
    for obj in objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        parent = getattr(obj, "parent", None)
        if parent is not None and getattr(parent, "type", None) == "ARMATURE":
            return parent
        for modifier in getattr(obj, "modifiers", []):
            if getattr(modifier, "type", None) == "ARMATURE" and getattr(modifier, "object", None):
                return modifier.object
    return None


def _real_bones_for_export(armature):
    bones = []
    for ordinal, bone in enumerate(armature.data.bones):
        try:
            mtf_index = int(bone.get("mtf_index", -1))
        except Exception:
            mtf_index = -1
        if mtf_index == 255 or bone.name.lower() == "bone255":
            continue
        if mtf_index < 0:
            match = _re.match(r"^bone(\d+)$", bone.name, _re.IGNORECASE)
            mtf_index = int(match.group(1)) if match else ordinal
        bones.append((mtf_index, bone))
    bones.sort(key=lambda item: item[0])
    if not bones:
        raise ModelExportError("The selected mesh has an armature, but it contains no exportable bones.")
    if len(bones) > 255:
        raise ModelExportError("MHGU v212 supports at most 255 exported bones.")
    # Writer uses a dense, local index array. Imported MT Framework rigs already
    # have contiguous mtf_index values; this map also handles ordinary Blender rigs.
    index_map = {old_index: new_index for new_index, (old_index, _) in enumerate(bones)}
    by_name = {bone.name: index_map[old_index] for old_index, bone in bones}
    return bones, index_map, by_name


def _group_id_from_object_name(name, fallback):
    match = _re.search(r"(?:^|[_\s])ID_(\d+)(?:[_\s(]|$)", str(name))
    if match:
        return int(match.group(1)) & 0xFF
    return int(fallback) & 0xFF


def _weight_map_for_source_vertex(obj, vertex, bone_index_by_name, bone_index_map):
    weights = {}
    for assignment in getattr(vertex, "groups", ()):
        group_index = int(assignment.group)
        if group_index < 0 or group_index >= len(obj.vertex_groups):
            continue
        group = obj.vertex_groups[group_index]
        bone_index = bone_index_by_name.get(group.name)
        # Imported meshes commonly name groups bone0, bone1 ... while custom rigs
        # may use their actual bone names; both routes are supported.
        if bone_index is None:
            match = _re.match(r"^bone(\d+)$", group.name, _re.IGNORECASE)
            if match:
                bone_index = bone_index_map.get(int(match.group(1)))
        weight = float(assignment.weight)
        if bone_index is not None and weight > 1.0e-8:
            weights[bone_index] = weights.get(bone_index, 0.0) + weight
    positive = sorted(((i, w) for i, w in weights.items() if w > 1.0e-8), key=lambda x: x[1], reverse=True)[:4]
    total = sum(w for _, w in positive)
    if total <= 1.0e-8:
        return [(0, 1.0)]
    return [(i, w / total) for i, w in positive]


def _collect_skinned_mod212_parts(objects, context, armature):
    """Collect original mesh datablocks (not evaluated armature output) plus skin weights."""
    mesh_objects = [obj for obj in objects if getattr(obj, "type", None) == "MESH"]
    if not mesh_objects:
        raise ModelExportError("Select at least one mesh object before exporting.")
    bones, bone_index_map, bone_index_by_name = _real_bones_for_export(armature)
    import mathutils
    parts = []
    material_names = []
    material_lookup = {}
    edge_keys = set()
    missing_weights = 0
    group_positions = {}

    for obj_index, obj in enumerate(mesh_objects):
        mesh = obj.data
        if mesh is None or len(mesh.vertices) == 0:
            continue
        try:
            mesh.calc_loop_triangles()
        except Exception:
            pass
        # Keep rest mesh coordinates: evaluated armature modifiers bake current pose
        # into positions and would double-transform the exported skin.
        transform = armature.matrix_world.inverted() @ obj.matrix_world
        normal_matrix = transform.to_3x3().inverted().transposed()
        uv_layers = list(mesh.uv_layers)
        active_uv = mesh.uv_layers.active
        uv1_layer = active_uv if active_uv is not None else (uv_layers[0] if uv_layers else None)
        remaining_uv = [layer for layer in uv_layers if layer != uv1_layer]
        uv2_layer = remaining_uv[0] if remaining_uv else uv1_layer
        group_id = _group_id_from_object_name(obj.name, obj_index)
        group_positions.setdefault(group_id, [])

        # Cache influences by source vertex, preserving vertex-group weights.
        source_weights = []
        for source_vertex in mesh.vertices:
            influences = _weight_map_for_source_vertex(
                obj, source_vertex, bone_index_by_name, bone_index_map)
            if not getattr(source_vertex, "groups", None) or not influences:
                missing_weights += 1
            source_weights.append(influences)

        by_material = {}
        loop_normals = []
        try:
            loop_normals = [n.vector.copy() for n in mesh.corner_normals]
        except Exception:
            loop_normals = []
        for tri in mesh.loop_triangles:
            polygon = mesh.polygons[tri.polygon_index]
            mat_slot = int(polygon.material_index)
            bucket = by_material.setdefault(mat_slot, {"vertices": [], "faces": [], "vertex_map": {}, "group_id": group_id})
            face = []
            for loop_index in tri.loops:
                loop = mesh.loops[loop_index]
                source_index = int(loop.vertex_index)
                p = transform @ mesh.vertices[source_index].co
                # Inverse of import cormat: (x, y, z)_Blender -> (x, z, -y)_MTF.
                position = _finite((p.x, p.z, -p.y), "skinned position")
                if loop_normals and loop_index < len(loop_normals):
                    n_src = loop_normals[loop_index]
                else:
                    n_src = loop.normal.copy()
                n = normal_matrix @ n_src
                if n.length > 1.0e-12:
                    n.normalize()
                normal = _finite((n.x, n.z, -n.y), "skinned normal")
                uv1 = tuple(uv1_layer.data[loop_index].uv) if uv1_layer else (0.0, 0.0)
                uv2 = tuple(uv2_layer.data[loop_index].uv) if uv2_layer else uv1
                uv1 = _finite(uv1, "UV1")
                uv2 = _finite(uv2, "UV2")
                influence = source_weights[source_index]
                # Imported MT Framework meshes may contain a separate Blender vertex
                # for every triangle corner. Rebuild the game's per-vertex buffer by
                # welding identical positions only when UVs, normals and skin weights
                # also match. Keeping source_index in this key caused a 3x+ vertex
                # explosion on re-export and made the MOD much less like a native mesh.
                weight_key = tuple((int(bone), round(float(weight), 6))
                                   for bone, weight in influence)
                key = (
                    tuple(round(v, 6) for v in position)
                    + tuple(round(v, 6) for v in uv1 + uv2 + normal)
                    + weight_key
                )
                out_index = bucket["vertex_map"].get(key)
                if out_index is None:
                    out_index = len(bucket["vertices"])
                    bucket["vertex_map"][key] = out_index
                    vertex_data = {
                        "position": position, "normal": normal,
                        "uv": uv1, "uv2": uv2, "weights": influence,
                        "source_vertex": source_index,
                    }
                    bucket["vertices"].append(vertex_data)
                    group_positions[group_id].append(position)
                face.append(out_index)
            bucket["faces"].append(tuple(face))
            # Count edges in the welded output topology, not the source loop-expanded
            # Blender mesh. The source-index version counted duplicate corner vertices
            # multiple times and inflated the MOD VertexIds header field.
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_keys.add((obj.name, mat_slot, min(a, b), max(a, b)))

        for mat_slot, bucket in sorted(by_material.items()):
            if not bucket["faces"]:
                continue
            material = mesh.materials[mat_slot] if mat_slot < len(mesh.materials) else None
            if material is None:
                raise ModelExportError(
                    "Mesh '%s' uses material slot %d, but that slot has no Blender material. "
                    "The previous exporter silently wrote a single 'default' material, which "
                    "does not match the game's MRL material names. Re-import with 'Import "
                    "Textures / Materials' enabled, or assign a valid material name before exporting."
                    % (obj.name, mat_slot)
                )
            try:
                mat_name = str(material.get("mtf_material_name", material.name))
            except Exception:
                mat_name = material.name
            if not mat_name or mat_name.strip().lower() == "default":
                raise ModelExportError(
                    "Mesh '%s' has no game material name. Assign the matching MT Framework material "
                    "name (or re-import with 'Import Textures / Materials' enabled) before exporting."
                    % obj.name
                )
            if mat_name not in material_lookup:
                material_lookup[mat_name] = len(material_names)
                material_names.append(mat_name)
            bucket.pop("vertex_map", None)
            bucket.update({
                "name": "%s_%s" % (obj.name, mat_name),
                "material_name": mat_name,
                "material_index": material_lookup[mat_name],
                "object_index": obj_index,
            })
            parts.append(bucket)

    if not parts:
        raise ModelExportError("No triangle faces were found in the selected mesh objects.")
    if missing_weights:
        # Missing per-vertex assignments are routed to root bone 0 by the helper;
        # make this visible so users know the output was repaired rather than exact.
        print("MT Framework export: %d source vertex(es) had no valid bone weights; assigned to bone 0." % missing_weights)
    for part in parts:
        if len(part["vertices"]) > 65535:
            raise ModelExportError("Part '%s' has more than 65,535 exported vertices." % part["name"])
        active_bones = sorted({idx for v in part["vertices"] for idx, weight in v["weights"] if weight > 1.0e-8})
        if not active_bones:
            active_bones = [0]
        if len(active_bones) > 255:
            raise ModelExportError("Part '%s' uses more than 255 bones." % part["name"])
        part["unique_bones"] = active_bones
    return parts, material_names, len(edge_keys), armature, bones, bone_index_map, group_positions


def _as_mtf_i8(value):
    # MHGU signed-byte normal decoder divides by 128 (not 127).
    return max(-128, min(127, int(round(float(value) * 128.0))))


def _half(value):
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    value = max(-65504.0, min(65504.0, value))
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _write_v212_skinned_vertex(vertex, mins, scale):
    """Write the GU-native SkinnedCharacter011[U1NS] (0x14d40022) layout.

    Layout is 28 bytes as observed in MHGU v212 files:
      int16x3 position, int16 weight0, int8x3 normal + pad,
      4-byte tangent, uint8x4 bone indices, half2 UV1, half2 weight1/weight2.
    The implicit fourth weight is decoded as 1 - weight0 - weight1 - weight2.
    """
    p = vertex["position"]
    normalized = [0.0 if scale <= 1.0e-12 else (float(p[i]) - mins[i]) / scale for i in range(3)]
    positions = [max(-32767, min(32767, int(round(v * 32767.0)))) for v in normalized]

    influences = sorted(vertex["weights"], key=lambda x: x[1], reverse=True)[:4]
    if not influences:
        influences = [(0, 1.0)]
    total = sum(max(0.0, w) for _, w in influences)
    if total <= 1.0e-12:
        influences = [(0, 1.0)]
        total = 1.0
    normalized_weights = [max(0.0, w) / total for _, w in influences]
    bone_ids = [int(i) for i, _ in influences]
    while len(bone_ids) < 4:
        bone_ids.append(0)
        normalized_weights.append(0.0)
    if any(i < 0 or i > 254 for i in bone_ids):
        raise ModelExportError("MHGU v212 skin data has an invalid bone index (must be 0..254).")

    # GU stores weight0 as normalized signed int16 and weights1/2 as half floats.
    # Keep a tiny margin so the implicit fourth weight cannot become negative.
    weights = [w * 0.9995 for w in normalized_weights[:4]]
    w0 = max(0, min(32767, int(round(weights[0] * 32767.0))))
    w1 = _half(weights[1])
    w2 = _half(weights[2])
    w0_float = w0 / 32767.0
    if w0_float + w1 + w2 > 1.0:
        w2 = _half(max(0.0, 1.0 - w0_float - w1 - 0.0001))

    nx, ny, nz = (_as_mtf_i8(v) for v in _finite(vertex["normal"], "skinned normal"))
    u, v = _finite(vertex.get("uv", (0.0, 0.0)), "UV1")

    # Tangent bytes 49 49 49 7F are present in the supplied native GU layout.
    # Keep that native default until tangent generation is implemented.
    tangent_word = 0x7F494949
    return struct.pack(
        "<4h3bBI4B2e2e",
        positions[0], positions[1], positions[2], w0,
        nx, ny, nz, 127,
        tangent_word,
        *bone_ids,
        u, 1.0 - v,
        w1, w2,
    )


def _matrix_blob(matrix):
    """Write a 4x4 Blender column matrix as MT Framework's 4 rows of (3f + pad)."""
    rows = matrix.transposed()
    out = bytearray()
    for row in range(4):
        pad = 0x3F800000 if row == 3 else 0
        out += struct.pack("<3fI", float(rows[row][0]), float(rows[row][1]), float(rows[row][2]), pad)
    return bytes(out)


def _v212_position_scale_from_armature(armature, fallback_scale):
    """Recover an imported MHGU model's original uniform position scale.

    The unchanged importer creates a non-deforming helper bone with mtf_index=255
    and sets its displayed length to max(1, length(vtbuffscale) * 0.02). For the
    uniform vertex scales used by MHGU v212, this lets Export recover the source
    scale without asking the user to choose the original MOD again.

    If the sentinel is missing or its length was clamped/edited, use the geometry
    bounds as a fallback for newly-created Blender models.
    """
    if armature is not None:
        try:
            for bone in armature.data.bones:
                try:
                    mtf_index = int(bone.get("mtf_index", -1))
                except Exception:
                    mtf_index = -1
                if mtf_index == 255 or str(getattr(bone, "name", "")).lower() == "bone255":
                    helper_length = float(bone.length)
                    if math.isfinite(helper_length) and helper_length > 1.0001:
                        scale = helper_length / (math.sqrt(3.0) * 0.02)
                        if math.isfinite(scale) and 1.0e-4 < scale < 1.0e6:
                            return scale
                    break
        except Exception:
            pass
    return float(fallback_scale) if math.isfinite(float(fallback_scale)) and float(fallback_scale) > 1.0e-8 else 1.0


def _build_v212_skeleton(armature, bones, bone_index_map, bbox_min, position_scale):
    import mathutils
    count = len(bones)
    c = mathutils.Matrix((
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    c_inv = c.inverted()
    scale_matrix = mathutils.Matrix.Diagonal((position_scale, position_scale, position_scale, 1.0))
    bone_by_name = {bone.name: new_index for new_index, (_, bone) in enumerate(bones)}
    world_engine = []
    local_engine = []
    for new_index, (old_index, bone) in enumerate(bones):
        parent = bone.parent
        if parent is not None and parent.name in bone_by_name:
            local = parent.matrix_local.inverted() @ bone.matrix_local
        else:
            # Strip the importer-created synthetic MTFCormat parent (bone255).
            local = c_inv @ bone.matrix_local
        local_engine.append(local)
        world_engine.append(c_inv @ bone.matrix_local)

    # The MOD v212 header gets its vertex scale and minimum from absolute matrix 0.
    # Vertex positions are stored as q=(p-bbox_min)/scale. For each bone, map q
    # back to model space and then into bone-local space: inverse(bind) * T(min) * S.
    local_blob = b"".join(_matrix_blob(m) for m in local_engine)
    offset_matrix = mathutils.Matrix.Identity(4)
    offset_matrix.translation = mathutils.Vector(bbox_min)
    absolute_blob = bytearray()
    for idx, world in enumerate(world_engine):
        absolute = world.inverted() @ offset_matrix @ scale_matrix
        if idx == 0:
            # Header parser recovers bbmin as absolute[0].translation +
            # local[0].translation. Keep that equation exact even when the root
            # bone has a non-zero rest translation.
            absolute = mathutils.Matrix.Identity(4)
            absolute[0][0] = position_scale
            absolute[1][1] = position_scale
            absolute[2][2] = position_scale
            absolute.translation = mathutils.Vector(bbox_min) - local_engine[0].translation
        absolute_blob += _matrix_blob(absolute)

    remap_table = None
    try:
        candidate = list(armature.get("mtf_remaptable", []))
        if len(candidate) == 256 and all(0 <= int(x) <= 255 for x in candidate):
            remap_table = [int(x) for x in candidate]
    except Exception:
        remap_table = None
    if remap_table is None:
        remap_table = [i if i < count else 255 for i in range(256)]

    inverse_game_ids = {}
    for game_id, ordinal in enumerate(remap_table):
        if ordinal < count and ordinal not in inverse_game_ids:
            inverse_game_ids[ordinal] = game_id

    # Mirror index: respect standard left/right naming where available; otherwise
    # reflect bone-head X around the model's centerline and use the nearest counterpart.
    mirror = {}
    replacement_pairs = ((".L", ".R"), ("_L", "_R"), ("Left", "Right"), ("left", "right"))
    for idx, (_, bone) in enumerate(bones):
        mirror_index = idx
        for left, right in replacement_pairs:
            target_name = bone.name.replace(left, right) if left in bone.name else (bone.name.replace(right, left) if right in bone.name else None)
            if target_name and target_name in bone_by_name:
                mirror_index = bone_by_name[target_name]
                break
        else:
            try:
                # Use distinct tolerances for identifying a centerline bone and
                # accepting a mirrored counterpart. The previous code reused one
                # tolerance for both checks; paired bones offset only 3 units from
                # the centerline were incorrectly treated as center bones when the
                # model scale made threshold > 3. This produced wrong mirror indices
                # for the MHGU sample (bone indices 11 and 15).
                centerline_threshold = max(0.025, float(position_scale) * 0.005)
                pair_distance_threshold = max(0.025, float(position_scale) * 0.05)
                source_x = float(bone.head_local.x)
                if abs(source_x) > centerline_threshold:
                    head = bone.head_local.copy(); head.x = -head.x
                    candidates = [
                        j for j in range(count)
                        if j != idx and float(bones[j][1].head_local.x) * source_x < 0.0
                    ]
                    if candidates:
                        nearest = min(candidates, key=lambda j: (bones[j][1].head_local - head).length)
                        if (bones[nearest][1].head_local - head).length <= pair_distance_threshold:
                            mirror_index = nearest
            except Exception:
                pass
        mirror[idx] = mirror_index

    info_blob = bytearray()
    for idx, (_, bone) in enumerate(bones):
        parent = bone.parent
        if parent is None or parent.name not in bone_by_name:
            parent_index = 255
        else:
            parent_index = bone_by_name[parent.name]
        game_id = inverse_game_ids.get(idx, idx)
        # MHGU's 24-byte bone record contains the four ID/mirror bytes, then one
        # bone-size scalar and a Vector4 encoding (length(local translation), tx, ty, tz).
        # The unchanged importer creates all visible Blender bones with a synthetic
        # display length, so it cannot recover the native per-bone scalar exactly; keep
        # a finite fallback for that scalar, but serialize the translation Vector4 in
        # the native field order (the previous version shifted these components left).
        local_matrix = local_engine[idx]
        local_translation = tuple(float(local_matrix.translation[axis]) for axis in range(3))
        translation_length = math.sqrt(sum(v * v for v in local_translation))
        try:
            bone_length = float(bone.length)
        except Exception:
            bone_length = 0.0
        if not math.isfinite(bone_length) or bone_length <= 1.0e-6:
            bone_length = 1.0e-3
        info_blob += struct.pack(
            "<4B5f", game_id & 0xFF, parent_index & 0xFF, mirror[idx] & 0xFF, 0,
            bone_length, translation_length, *local_translation)
    return bytes(info_blob) + local_blob + bytes(absolute_blob) + bytes(remap_table), world_engine, remap_table


def _weight_bound_record(bone_index, world_matrix, vertices):
    import mathutils
    inv = world_matrix.inverted()
    local_positions = [inv @ mathutils.Vector(v) for v in vertices]
    mins = tuple(min(p[i] for p in local_positions) for i in range(3))
    maxs = tuple(max(p[i] for p in local_positions) for i in range(3))
    center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    radius = max(((p - mathutils.Vector(center)).length for p in local_positions), default=0.0)
    half = tuple((maxs[i] - mins[i]) * 0.5 for i in range(3))
    yz_radius = math.sqrt(half[1] * half[1] + half[2] * half[2])
    out = bytearray(struct.pack("<I12x4f4f4f", int(bone_index), *center, radius,
                                mins[0], mins[1], mins[2], 0.0,
                                maxs[0], maxs[1], maxs[2], 0.0))
    # This matrix is the inverse rest transform in the bone-bound record's space.
    out += _matrix_blob(inv)
    out += struct.pack("<4f", half[0], yz_radius, half[2], 0.0)
    if len(out) != 144:
        raise ModelExportError("Internal MHGU bone-bound record size error (%d bytes)." % len(out))
    return bytes(out)


def _mod212_descriptor_skinned(part, part_index, vertex_offset, face_offset, face_count):
    vertex_count = len(part["vertices"])
    unique_bones = part["unique_bones"]
    if vertex_count <= 0 or vertex_count > 65535:
        raise ModelExportError("MOD v212 mesh part has an invalid vertex count.")
    if len(unique_bones) > 255:
        raise ModelExportError("MOD v212 mesh part uses more than 255 bones.")
    group_id = int(part.get("group_id", 0)) & 0xFF
    mat = int(part["material_index"]) * 16
    # Native MHGU v212 descriptors for 0x14d40022 do not use the generic
    # all-zero defaults: the first word and the small descriptor flag are set
    # in the supplied game files. In the inspected GU character sample, group
    # 100 uses 0xE4C0 / 0x21, while ordinary group 0 skinned meshes use
    # 0xEFFF / 0x19. Byte 11 is 0x04 for this vertex format (not zero).
    # The 0x119 variant seen on one source part carries an additional per-part
    # condition that is not recoverable from a generic Blender mesh, so use
    # the common native value rather than inventing that special case.
    descriptor_tag = 0xE4C0 if group_id == 100 else 0xEFFF
    # Native MHGU descriptors parsed field-by-field from the supplied v212
    # file: tag, vertex count, group id, material*16, LOD=0xFF, flags=0x0019
    # (ordinary group) or 0x0021 (group 100), stride=28, subtype=4, vertexSub,
    # vertexOffset, block hash, faceOffset and faceCount.
    # Do not swap the LOD byte with the following 16-bit flags field.
    descriptor_flags = 0x0021 if group_id == 100 else 0x0019
    descriptor = bytearray(struct.pack(
        "<HHBHBHBBIIIII",
        descriptor_tag, vertex_count, group_id, mat, 0xFF, descriptor_flags,
        28, 4, 0, vertex_offset, 0x14D40022, face_offset, face_count))
    descriptor += bytes(5)
    descriptor += struct.pack("<B", len(unique_bones))
    descriptor += struct.pack("<HHHI", part_index + 1, 0, vertex_count - 1, 0)
    if len(descriptor) != 48:
        raise ModelExportError("Internal MHGU mesh descriptor must be 48 bytes.")
    return bytes(descriptor)


def _write_mod212_skinned(filepath, parts, material_names, vertex_ids, armature, bones, bone_index_map):
    import mathutils
    mesh_count = len(parts)
    material_count = len(material_names)
    if mesh_count > 65535 or material_count > 65535:
        raise ModelExportError("MOD v212 has too many mesh parts or materials.")

    positions = [v["position"] for part in parts for v in part["vertices"]]
    if not positions:
        raise ModelExportError("No exported vertices.")
    mins = tuple(min(p[i] for p in positions) for i in range(3))
    maxs = tuple(max(p[i] for p in positions) for i in range(3))
    bounds_scale = max(maxs[i] - mins[i] for i in range(3))
    if bounds_scale <= 1.0e-8:
        bounds_scale = 1.0
    # Imported MHGU armatures carry the original scale implicitly in the synthetic
    # bone255 display length. Use it for both position quantization and skeleton
    # absolute matrices. Deriving scale only from the visible mesh bounds is wrong
    # when the importer has skipped source layouts that it cannot decode: the model
    # still imports in Blender but the game deforms/offsets it when exported.
    position_scale = _v212_position_scale_from_armature(armature, bounds_scale)

    bone_blob, world_engine, remap_table = _build_v212_skeleton(
        armature, bones, bone_index_map, mins, position_scale)
    if len(bone_blob) != len(bones) * 152 + 256:
        raise ModelExportError("Internal MOD v212 skeleton block size mismatch.")

    # The v212 group table is independent of Blender vertex groups. Derive game-style
    # group IDs from imported object names (ID_0, ID_100, ...) and generate bounds.
    group_positions = {}
    for part in parts:
        group_positions.setdefault(int(part.get("group_id", 0)) & 0xFF, []).extend(
            v["position"] for v in part["vertices"])
    group_blob = bytearray()
    for group_id in sorted(group_positions):
        ps = group_positions[group_id]
        gmin = tuple(min(p[i] for p in ps) for i in range(3))
        gmax = tuple(max(p[i] for p in ps) for i in range(3))
        center = tuple((gmin[i] + gmax[i]) * 0.5 for i in range(3))
        radius = max((math.sqrt(sum((p[i] - center[i]) ** 2 for i in range(3))) for p in ps), default=0.0)
        group_blob += struct.pack("<I7f", group_id, 0.0, 0.0, 0.0, *center, radius)

    material_blob = b"".join(_fixed_name(n) for n in material_names)
    header_size = 64
    # v212's extension block is exactly 68 bytes after the 64-byte main header.
    extra_header_size = 68
    bones_offset = header_size + extra_header_size
    group_offset = bones_offset + len(bone_blob)
    texture_offset = group_offset + len(group_blob)
    mesh_offset = texture_offset + len(material_blob)
    mesh_table_size = mesh_count * 48

    # Build one 144-byte weight-bound record for every unique bone influence in each
    # mesh. Native GU files show precisely this stride (descriptor byte 37 stores count).
    bound_blob = bytearray()
    descriptors = []
    vertex_blobs = []
    face_blob = bytearray()
    vertex_cursor = 0
    face_cursor = 0
    total_vertices = 0
    total_face_count = 0
    total_triangles = 0
    for part_index, part in enumerate(parts):
        part_bone_positions = {b: [] for b in part["unique_bones"]}
        for vertex in part["vertices"]:
            for bone_index, weight in vertex["weights"]:
                if weight > 1.0e-8 and bone_index in part_bone_positions:
                    part_bone_positions[bone_index].append(vertex["position"])
        for bone_index in part["unique_bones"]:
            pts = part_bone_positions.get(bone_index) or [v["position"] for v in part["vertices"]]
            bound_blob += _weight_bound_record(bone_index, world_engine[bone_index], pts)

        vb = b"".join(_write_v212_skinned_vertex(v, mins, position_scale) for v in part["vertices"])
        if len(vb) != len(part["vertices"]) * 28:
            raise ModelExportError("Internal MHGU-native stride-28 vertex mismatch in %s." % part["name"])
        indices = _mod_faces(part, 212, vertex_sub=0)
        if any(idx < 0 or idx > 65535 for idx in indices):
            raise ModelExportError("Mesh part '%s' contains an index outside the 16-bit range." % part["name"])
        if any(idx >= len(part["vertices"]) for idx in indices if idx != 0xFFFF):
            raise ModelExportError("Mesh part '%s' has an out-of-range triangle index." % part["name"])
        descriptors.append(_mod212_descriptor_skinned(part, part_index, vertex_cursor, face_cursor, len(indices)))
        vertex_blobs.append(vb)
        face_blob += struct.pack("<%dH" % len(indices), *indices) if indices else b""
        face_blob += struct.pack("<H", 0xFFFF)  # one native per-mesh restart/pad
        vertex_cursor += len(vb)
        face_cursor += len(indices) + 1
        total_face_count += len(indices) + 1
        total_vertices += len(part["vertices"])
        total_triangles += len(part["faces"])

    descriptor_blob = b"".join(descriptors)
    vertex_offset = mesh_offset + mesh_table_size + len(bound_blob)
    vertex_blob = b"".join(vertex_blobs)
    faces_offset = vertex_offset + len(vertex_blob)
    unk_offset = _align(faces_offset + len(face_blob), 16)

    # Game-facing v212 header extension: bounding sphere (center xyz + radius),
    # bounding min/max vec4s, followed by the MHGU fixed flags.
    center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    sphere_radius = max((math.sqrt(sum((p[i] - center[i]) ** 2 for i in range(3))) for p in positions), default=0.0)
    # Native v212 header word at offset 128 is the number of per-mesh
    # bone-bound records. It must equal the actual serialized 144-byte records.
    # For f_body663, the prior formula wrote 48 while the descriptors/buffer
    # contained 44, causing a table-count/offset disagreement in the game.
    bone_bound_count = sum(len(part["unique_bones"]) for part in parts)
    if len(bound_blob) != bone_bound_count * 144:
        raise ModelExportError("Bone-bound header count does not match serialized records.")
    extra = struct.pack("<12f5I", *center, sphere_radius, *mins, 0.0, *maxs, 0.0,
                        1000, 3000, 0x80000001, 0, bone_bound_count)
    if len(extra) != extra_header_size:
        raise ModelExportError("Internal MHGU header extension size mismatch.")

    header = struct.pack(
        "<4sBBHHHIIIII I7I",
        b"MOD\x00", 212, 0, len(bones), mesh_count, material_count,
        total_vertices, total_face_count, max(0, int(vertex_ids)), len(vertex_blob), 0,
        len(group_positions), bones_offset, group_offset, texture_offset, mesh_offset,
        vertex_offset, faces_offset, unk_offset)
    if len(header) != header_size:
        raise ModelExportError("Internal MHGU header size mismatch.")

    payload = bytearray(header + extra)
    if len(payload) != bones_offset:
        raise ModelExportError("Internal MHGU bones offset mismatch.")
    payload += bone_blob
    payload += group_blob
    payload += material_blob
    payload += descriptor_blob
    if len(payload) + len(bound_blob) != vertex_offset:
        raise ModelExportError("Internal MHGU bone-bound offset mismatch.")
    payload += bound_blob
    payload += vertex_blob
    payload += face_blob
    payload += bytes(unk_offset - len(payload))
    payload += bytes(4)
    if len(payload) != unk_offset + 4:
        raise ModelExportError("Internal MHGU file tail size mismatch.")

    _atomic_write(filepath, bytes(payload))
    return {"version": 212, "meshes": mesh_count, "materials": material_count,
            "vertices": total_vertices, "triangles": total_triangles,
            "bones": len(bones), "size": len(payload), "skinned": True,
            "bone_bound_records": sum(len(p["unique_bones"]) for p in parts)}


def _export_selected_meshes_with_skinned_gu(filepath, objects, context, target):
    """Override public entrypoint only for the GU skinned path; other targets retain old behavior."""
    if target == "MOD_V212":
        armature = _find_export_armature(objects, context)
        if armature is not None:
            parts, materials, vertex_ids, armature, bones, bone_index_map, _ = _collect_skinned_mod212_parts(
                objects, context, armature)
            return _write_mod212_skinned(filepath, parts, materials, vertex_ids,
                                         armature, bones, bone_index_map)
    # Keep the existing static MOD v212 fallback and all v230/MOD3 behavior unchanged.
    parts, materials, vertex_ids = collect_blender_parts(objects, context)
    if target == "MOD_V212":
        return _write_mod(filepath, parts, materials, vertex_ids, 212)
    if target == "MOD_V230":
        return _write_mod(filepath, parts, materials, vertex_ids, 230)
    if target == "MOD3_V237":
        return _write_mod3(filepath, parts, materials, vertex_ids)
    raise ModelExportError("Unknown export target %r." % target)


# Replace the initial static-only entry point with the skinned-aware router.
export_selected_meshes = _export_selected_meshes_with_skinned_gu
