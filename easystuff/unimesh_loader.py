"""
unimesh_loader.py
Port of EasyStuff/UniMeshLoader.ms, adapted from the 3ds Max mesh API to
Blender's bpy / bmesh.

Original responsibilities:
  ESUVArray         - a UV/colour channel container (id selects the channel)
  ESCreateUniMesh   - builds a 3ds Max `mesh` from a decoded vertex buffer
  ESSkinOptimizer   - collapses per-vertex bone/weight pairs

In Max, ESCreateUniMesh wrote directly into mesh map channels. Here it returns
the assembled arrays and `build_blender_mesh` constructs an actual bpy mesh +
optional armature skinning, because the two APIs differ too much for a literal
port. The decoding contract (which arrays exist, their meaning) is preserved.
"""

import mathutils

# MTFCormat = matrix3 [1,0,0] [0,0,1] [0,-1,0] [0,0,0]
# In MaxScript `v * MTFCormat` is a ROW-vector multiply: result = x*row1 + y*row2
# + z*row3 = (x, -z, y). (Engine Y-up -> Z-up; Max and Blender are both Z-up.)
def cormat(v):
    return mathutils.Vector((v[0], -v[2], v[1]))


# Equivalent matrix for Blender's COLUMN-vector convention (M @ v). Because
# Blender multiplies on the left, this is the TRANSPOSE of the Max rows above:
# rows become (1,0,0)/(0,0,-1)/(0,1,0), so M @ (x,y,z) = (x, -z, y).
MTFCORMAT = mathutils.Matrix((
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
))


class ESUVArray:
    """UV / vertex-colour channel.

    id meaning (from the MaxScript source):
        id  > 0 : UV map channel (1, 2, 3 ...)
        id == 0 : vertex colour
        id == -2: vertex alpha (stored as a separate colour channel here)
    """

    __slots__ = ("id", "data")

    def __init__(self, id=1, data=None):
        self.id = id
        self.data = data if data is not None else []


def proxima(value1, value2=0.0, epsilon=0.0001):
    return (value1 <= value2 + epsilon) and (value1 >= value2 - epsilon)


def es_skin_optimizer(struct_ref, count, bones, weights):
    """Port of ESSkinOptimizer.

    Collapses duplicate bone references and drops zero / near-zero weights,
    appending the cleaned (bone_ids, weights) onto the decoder's arrays.
    `bones` are 1-based local bone indices (already +1 in the decoders).
    """
    cwt = []
    cbn = []
    for b in range(count):
        w = weights[b]
        if w > 0 and not proxima(w):
            bone_id = bones[b]
            try:
                bone_id_f = float(bone_id)
                bone_id_i = int(round(bone_id_f))
                if abs(bone_id_f - bone_id_i) > 1e-4:
                    continue
            except (TypeError, ValueError):
                continue
            if bone_id_i in cbn:
                idx = cbn.index(bone_id_i)
                cwt[idx] += w
            else:
                # Some older MT Framework formats store bone IDs in a
                # half-float field. They represent integer bone IDs, but
                # Python keeps the decoded value as float. Normalize it here
                # so downstream list indexing always uses an int.
                bone_id = bones[b]
                try:
                    bone_id_f = float(bone_id)
                    bone_id_i = int(round(bone_id_f))
                    if abs(bone_id_f - bone_id_i) > 1e-4:
                        continue
                except (TypeError, ValueError):
                    continue
                cbn.append(bone_id_i)
                cwt.append(w)
    struct_ref.boneidarray.append(cbn)
    struct_ref.weightarray.append(cwt)


# ----------------------------------------------------------------------------
# Blender mesh construction
# ----------------------------------------------------------------------------
def build_blender_mesh(name, meshdata, bone_names_for_part, context, collection):
    """Create a bpy mesh object from a decoded vertex buffer (`meshdata`).

    meshdata exposes (mirroring the decoder structs):
        vertarray    : list[Vector]            already coordinate-transformed
        facearray    : list[[a,b,c]]           1-based indices (+modf applied)
        normalarray  : list[Vector] | []
        uvarray      : list[ESUVArray]
        boneidarray  : list[list[int]] | []    per-vertex local bone indices (1-based)
        weightarray  : list[list[float]] | []

    bone_names_for_part : list[str] mapping local bone index -> blender bone
                          name, or None to skip skinning.
    """
    import bpy
    import bmesh

    verts = meshdata.vertarray
    faces = meshdata.facearray or []

    me = bpy.data.meshes.new(name)
    # Convert 1-based decoder indices to 0-based, skip degenerate tris.
    py_faces = []
    nv = len(verts)
    # MHGU MOD v212 can contain triangles whose winding disagrees with the
    # decoded per-vertex normals. Compare each geometric face normal with the
    # average vertex normal, flipping only that triangle when they oppose.
    # Other versions, including MHXX v230, keep their original face order.
    header = getattr(meshdata, "headerref", None)
    is_mhgu_v212 = header is not None and getattr(header, "Version", None) == 212
    decoded_normals = getattr(meshdata, "normalarray", None) or []
    can_check_winding = is_mhgu_v212 and len(decoded_normals) == nv

    for f in faces:
        a, b, c = f[0] - 1, f[1] - 1, f[2] - 1
        if a == b or b == c or a == c:
            continue
        if a < 0 or b < 0 or c < 0 or a >= nv or b >= nv or c >= nv:
            continue

        if can_check_winding:
            try:
                va, vb, vc = verts[a], verts[b], verts[c]
                geometric = (vb - va).cross(vc - va)
                expected = decoded_normals[a] + decoded_normals[b] + decoded_normals[c]
                if geometric.length > 1e-12 and expected.length > 1e-12:
                    if geometric.dot(expected) < 0.0:
                        b, c = c, b
            except (AttributeError, TypeError, ValueError):
                pass

        py_faces.append((a, b, c))

    me.from_pydata([tuple(v) for v in verts], [], py_faces)
    me.update()

    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)

    # -- custom split normals -------------------------------------------------
    normals = getattr(meshdata, "normalarray", None)
    if normals and len(normals) == nv:
        try:
            me.normals_split_custom_set_from_vertices(
                [tuple(n.normalized()) if n.length > 0 else (0.0, 0.0, 1.0)
                 for n in normals])
        except Exception:
            pass  # normals are best-effort; geometry still imports

    # -- UV / colour channels -------------------------------------------------
    uvarrays = getattr(meshdata, "uvarray", None) or []
    loops = me.loops
    polys = me.polygons
    for ch in uvarrays:
        if not ch.data:
            continue
        if ch.id > 0:
            uv_layer = me.uv_layers.new(name="UV%d" % ch.id)
            for poly in polys:
                for li in poly.loop_indices:
                    vi = loops[li].vertex_index
                    if vi < len(ch.data):
                        d = ch.data[vi]
                        uv_layer.data[li].uv = (d[0], d[1])
        else:
            # vertex colour (id 0) or alpha (id -2) -> colour attribute
            cname = "Color" if ch.id == 0 else "Alpha"
            try:
                col = me.color_attributes.new(
                    name=cname, type="BYTE_COLOR", domain="CORNER")
                for poly in polys:
                    for li in poly.loop_indices:
                        vi = loops[li].vertex_index
                        if vi < len(ch.data):
                            d = ch.data[vi]
                            r = d[0] if len(d) > 0 else 0.0
                            g = d[1] if len(d) > 1 else r
                            b = d[2] if len(d) > 2 else r
                            col.data[li].color = (r, g, b, 1.0)
            except Exception:
                pass

    # -- skin weights via vertex groups --------------------------------------
    boneids = getattr(meshdata, "boneidarray", None)
    weights = getattr(meshdata, "weightarray", None)
    if (bone_names_for_part and boneids and weights
            and len(boneids) == nv):
        groups = {}
        for vi in range(nv):
            bids = boneids[vi]
            wts = weights[vi]
            for k in range(len(bids)):
                local = bids[k]            # 1-based local index
                try:
                    local_i = int(round(float(local)))
                except (TypeError, ValueError):
                    continue
                idx = local_i - 1
                if idx < 0 or idx >= len(bone_names_for_part):
                    continue
                bname = bone_names_for_part[idx]
                if bname is None:
                    continue
                grp = groups.get(bname)
                if grp is None:
                    grp = obj.vertex_groups.get(bname) or obj.vertex_groups.new(name=bname)
                    groups[bname] = grp
                w = wts[k] if k < len(wts) else 0.0
                if w != 0.0:
                    grp.add([vi], w, "REPLACE")

    return obj
