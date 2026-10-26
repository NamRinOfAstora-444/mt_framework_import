# MT Framework (MOD/LMT) Importer for Blender
# Copyright (C) 2026 axestyra
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. This program is distributed WITHOUT ANY WARRANTY. See the LICENSE
# file, or <https://www.gnu.org/licenses/>, for the full license text.
"""
MT Framework (LMT, MOD) importer for Blender
=============================================

Blender port of Lukas Cone's "MT Framework Import Tool" (BETA 5.1), originally
a 3ds Max MaxScript. This module corresponds to the top-level
"MT Framework (LMT, MOD).ms": it wires the format parsers (mod_format.py),
the EasyStuff helpers (easystuff/) and the per-game decoders (mtf/) into a
Blender import operator, and contains MODRuntime -- the model import
orchestrator -- plus the armature builder that replaces the Max scene/IK code.

STATUS (read me):
  * MOD model import: positions, faces, UVs, normals, vertex colours and skin
    weights are wired end-to-end. The newer MTF/7 vertex layouts are also
    registered for MHW-era/v237 models.
  * Skeleton: imported as a real Blender armature with parenting from the bone
    table. The Max IK rig (IKHISolver chains, .tit presets) is NOT ported -- it
    is a different system in Blender and is rarely needed for asset ripping.
  * LMT animation import and basic v55 export are available. Export currently
    writes the active Action as one little-endian animation block.

Every byte offset is a direct port of the MaxScript and must be validated
against real game files.
"""

bl_info = {
    "name": "MT Framework (MOD/LMT) Importer",
    "author": "axestyra",
    "version": (0, 20, 8),
    "blender": (3, 0, 0),
    "location": "File > Import > MT Framework Model / Animation",
    "description": "Import Capcom MT Framework MOD models and LMT animations.",
    "doc_url": "",
    "tracker_url": "",
    "category": "Import-Export",
}

import os

import bpy
import mathutils
from bpy.props import StringProperty, BoolProperty, CollectionProperty, IntProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper

from .easystuff.read_stream import BinaryReader
from .easystuff.unimesh_loader import build_blender_mesh, MTFCORMAT
from . import mod_format
from . import mtf
from . import model_export


# MOD versions whose non-outline materials need backface culling. These newer
# character models (e.g. chr201, chr123) have double-walled geometry that shows
# black artifacts when rendered two-sided. Older v231 models are left two-sided.
# Add a version here if a model of that version needs culling.
_BACKFACE_CULL_MOD_VERSIONS = frozenset({230})


# MTFCormat as a 4x4 in Blender's column-vector convention (transpose of the
# Max rows). M @ (x,y,z) = (x, -z, y); used to place the bone hierarchy root.
MTFCORMAT_4 = mathutils.Matrix((
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


def _max_to_blender_matrix(m_max):
    """Convert a Max row-vector matrix (as read by _read_matrix44) to a Blender
    column-vector matrix by transposing."""
    return m_max.transposed()


def build_armature(mbones, header, context, collection):
    """Build a Blender armature from MODBones.

    Faithful port of MODBones.generate(): each bone's world rest transform is its
    LOCAL matrix composed up the hierarchy from the MTFCormat root
    (world = local * parent_world in Max row-vector terms; in Blender that becomes
    parent_world @ local, with every Max matrix transposed to column-vector form).
    The earlier version used the absolute matrices, which bake in the ~271x model
    scale and threw the bones far off the mesh -- that was the misalignment.

    Returns: (armature_object, bone_name_by_index dict)
    """
    if not mbones.bones:
        return None, {}

    arm_data = bpy.data.armatures.new("MT_Armature")
    arm_obj = bpy.data.objects.new("MT_Armature", arm_data)
    collection.objects.link(arm_obj)

    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_data.edit_bones
    name_by_index = {}
    nbones = len(mbones.bones)

    # Local matrices -> Blender column-vector convention (transpose).
    L = [m.transposed() for m in mbones.lmatrices]
    root = MTFCORMAT_4

    world_cache = {}

    def world(i):
        if i in world_cache:
            return world_cache[i]
        world_cache[i] = root  # guard against cycles while resolving
        pid = mbones.bones[i].parentid
        if pid is None or pid == 255 or pid >= nbones or pid == i:
            pw = root
        else:
            pw = world(pid)
        w = pw @ L[i]
        world_cache[i] = w
        return w

    # Visible bone length scaled to the model (skinning is unaffected by length).
    blen = max(1.0, header.vtbuffscale.length * 0.02)

    # The original 3ds Max importer always creates a synthetic bone255 root
    # using MTFCormat.  Older LMT generations (including MHGU v67 files) store
    # root motion on BoneID=255 / Usage=4.  Without this helper bone Blender
    # silently drops the track and travelling animations become in-place.
    root_eb = edit_bones.new("bone255")
    root_eb.head = (0.0, 0.0, 0.0)
    root_eb.tail = (0.0, blen, 0.0)
    root_eb.matrix = root
    root_eb.use_deform = False
    # Synthetic MT root sentinel used by generations whose LMT root-motion
    # track is stored on BoneID 255.  This property is independent of the
    # visible Blender name, so users can rename bone255 as well.
    root_eb["mtf_index"] = 255

    for i in range(nbones):
        name = "bone%d" % i
        eb = edit_bones.new(name)
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, blen, 0.0)
        eb.matrix = world(i)  # sets armature-space rest transform
        # Keep the MT Framework bone index separate from the Blender bone name.
        # LMT tracks reference the skeleton-local bone index; users may freely
        # rename Blender bones without breaking animation lookup.
        eb["mtf_index"] = i
        name_by_index[i] = name

    for i in range(nbones):
        pid = mbones.bones[i].parentid
        eb = edit_bones.get(name_by_index[i])
        if pid is None or pid == 255:
            eb.parent = root_eb
        elif 0 <= pid < nbones and pid != i:
            pe = edit_bones.get(name_by_index[pid])
            if pe is not None and pe is not eb:
                eb.parent = pe

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj, name_by_index


def _resolve_texture_image(model_dir, base_name):
    """Find a loadable image file in the model folder for an MRL texture name."""
    import bpy
    # MT Framework ships .tex (not Blender-readable); folders usually also have a
    # converted .png/.dds/.tga sitting next to the model. Try those in order.
    for ext in (".png", ".tga", ".dds", ".tif", ".tiff", ".jpg", ".jpeg"):
        cand = os.path.join(model_dir, base_name + ext)
        if os.path.exists(cand):
            try:
                return bpy.data.images.load(cand, check_existing=True)
            except Exception:
                return None
    return None


def _make_emission_material(name, img, strength=1.0, backface_cull=False):
    """Toon/unshaded material: Image Texture -> Emission -> Output."""
    import bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.location = (0, 0)
    emi.inputs["Strength"].default_value = strength
    nt.links.new(emi.outputs[0], out.inputs["Surface"])
    if img is not None:
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        tex_node.location = (-320, 0)
        nt.links.new(tex_node.outputs["Color"], emi.inputs["Color"])
    mat.use_backface_culling = backface_cull
    return mat


def _material_image(mat):
    """Return the Image of a material's texture node, or None."""
    if mat is None or not mat.use_nodes:
        return None
    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node.image
    return None


def _read_mod_material_names(reader, header):
    """Read the MOD material-name table.

    It lives at header.TextureOffset (the MaxScript reads this offset but never
    uses it), 128 bytes per entry, MaterialCount entries. Each mesh's material
    index (part.material // 16) selects a name here; that name binds to the .mrl
    material of the same name. Returns a list of names (index-aligned).
    """
    names = []
    off = getattr(header, "TextureOffset", 0) or 0
    count = getattr(header, "MaterialCount", 0) or 0
    data = getattr(reader, "data", b"")
    if off <= 0 or count <= 0:
        return names
    for i in range(count):
        ent = data[off + i * 128: off + i * 128 + 128]
        end = ent.find(b"\x00")
        raw = ent[:end] if end >= 0 else ent
        names.append(raw.decode("ascii", "ignore"))
    return names


def build_materials(model_path, used_indices, mod_mat_names, options, mod_version=0):
    """Build {material_index: bpy.types.Material}.

    Mesh -> material is part.material // 16, which selects a name from the MOD
    material-name table (mod_mat_names). That name binds to the .mrl material of
    the same name -- matched by JAMCRC hash, since the MOD and .mrl order their
    materials independently -- and that material's albedo gives the texture.

    Materials whose name begins "ol_" are outline shaders: rendered as dark
    (Emission strength 0.1), backface-culled inverted hulls. All other materials
    render normally at full strength.

    Backface culling for the normal (non-outline) materials is version-dependent:
    the newer MOD v230 character models have double-walled geometry that shows
    black artifacts when drawn two-sided, so those are culled; the older v231
    models are left two-sided (the safe default, since a wrong cull hides faces).
    """
    import bpy
    from . import mrl_format

    base_cull = mod_version in _BACKFACE_CULL_MOD_VERSIONS

    model_dir = os.path.dirname(model_path)
    base = os.path.splitext(os.path.basename(model_path))[0]
    mrl_path = os.path.join(model_dir, base + ".mrl")

    tex_names = []
    hash_to_tex = {}
    if os.path.exists(mrl_path):
        try:
            with open(mrl_path, "rb") as fh:
                mrl_data = fh.read()
            tex_names = mrl_format.parse_mrl_textures(mrl_data)
            hash_to_tex = mrl_format.parse_mrl_albedo_by_hash(
                mrl_data, len(tex_names))
        except Exception:
            tex_names, hash_to_tex = [], {}

    # Resolve each texture name to a folder image (lazily, cached).
    image_cache = {}

    def image_for(tex_index):
        if tex_index is None or tex_index >= len(tex_names):
            return None
        if tex_index in image_cache:
            return image_cache[tex_index]
        img = _resolve_texture_image(model_dir, tex_names[tex_index])
        image_cache[tex_index] = img
        return img

    materials = {}
    for idx in sorted(used_indices):
        name = mod_mat_names[idx] if idx < len(mod_mat_names) else ""
        lname = name.lower()
        is_outline = lname.startswith("ol_")
        # The material-name prefix encodes a render category: "*A__" (e.g.
        # offA__, onA__) marks alpha/transparent surfaces -- effects and glass --
        # which must be two-sided regardless of version, else culling leaves
        # black artifacts where the back of the surface should show.
        is_alpha = lname.split("__", 1)[0].endswith("a")
        tex_index = hash_to_tex.get(mrl_format.mrl_name_hash(name)) if name else None
        img = image_for(tex_index)
        if is_outline:
            mat = _make_emission_material(
                "%s_mat_%d" % (base, idx), img, 0.1, True)
        else:
            cull = base_cull and not is_alpha
            mat = _make_emission_material(
                "%s_mat_%d" % (base, idx), img, 1.0, cull)
        mat["mtf_material_index"] = idx
        if name:
            mat["mtf_material_name"] = name
        if tex_index is not None and tex_index < len(tex_names):
            mat["mtf_texture"] = tex_names[tex_index]
        materials[idx] = mat
    return materials


def mod_runtime(reader, options, context):
    """Port of fn MODRuntime. Parses the MOD and builds the Blender scene.

    options: dict with keys
        proc_parts (bool), x64 (bool), keep_object_id (bool),
        layer_per_lod (bool), keep_debug_info (bool)
    """
    header = mod_format.MODHeader(reader, stresstest=False, x64=options["x64"])

    reader.seek(header.BonesOffset, reader.SEEK_SET)
    mbones = mod_format.MODBones(header)

    # Header bbmin/scale refinement from absolute matrices (non-v1 path),
    # mirroring MODRuntime.
    if not (header.Version < 190) and len(mbones.amatrices) > 0:
        a0 = mbones.amatrices[0]
        l0 = mbones.lmatrices[0]
        # Max: vtbuffscale = amatrices[1].scale ; bbmin = amatrices[1].row4 + lmatrices[1].row4
        scale = a0.to_3x3().to_scale()
        header.vtbuffscale = mathutils.Vector((scale[0], scale[1], scale[2]))
        # row4 in Max == translation row (index 3)
        a_row4 = mathutils.Vector((a0[3][0], a0[3][1], a0[3][2]))
        l_row4 = mathutils.Vector((l0[3][0], l0[3][1], l0[3][2]))
        header.bbmin = a_row4 + l_row4

    # Collection to hold everything for this import.
    collection = bpy.data.collections.new(
        "MTF_" + os.path.splitext(os.path.basename(options.get("filepath", "model")))[0])
    context.scene.collection.children.link(collection)

    arm_obj, name_by_index = build_armature(mbones, header, context, collection)
    if arm_obj is not None:
        # The LMT animation importer maps anim bone ids through this table.
        arm_obj["mtf_remaptable"] = list(mbones.remaptable)
        arm_obj["mtf_lmt_rootid"] = 255  # version<=90 uses 255; >90 uses -1
        arm_obj["mtf_root_bone"] = "bone255"

    if not options["proc_parts"]:
        return mbones, None

    reader.seek(header.MeshOffset, reader.SEEK_SET)
    parts = mod_format.MODMeshParts(header)

    reader.seek(header.VertexOffset, reader.SEEK_SET)
    for part in parts.parts:
        part.loadmeshdata(mtf.REGISTRY)

    # Materials: each mesh's material index (part.material // 16) selects a name
    # from the MOD material-name table; that name binds to the .mrl material of
    # the same name (by hash). Outline materials are named "ol_*" and are
    # rendered dark + backface-culled inside build_materials.
    materials = {}
    if options.get("import_textures"):
        used = set(p.material // 16 for p in parts.parts)
        mod_mat_names = _read_mod_material_names(reader, header)
        try:
            materials = build_materials(
                options.get("filepath", ""), used, mod_mat_names, options,
                mod_version=getattr(header, "Version", 0))
        except Exception as exc:  # noqa: BLE001
            print("Texture/material import failed: %s" % exc)
            materials = {}

    lod_collections = {}
    cur_part = 0
    for part in parts.parts:
        md = part.meshdata
        if md is None:
            cur_part += 1
            continue

        # Name (mirrors the MODRuntime naming logic).
        name = ""
        if options["keep_object_id"]:
            name += "Object_%d_" % cur_part
        name += "ID_%s" % str(part.id)
        if not options["layer_per_lod"]:
            name += "_LOD_%s" % str(part.LOD)
        if options["keep_debug_info"]:
            name += " (%s)" % str(getattr(md, "id", "?"))

        # Which collection does this mesh go in (LOD grouping).
        target_collection = collection
        if options["layer_per_lod"]:
            lname = "LOD %s" % str(part.LOD)
            target_collection = lod_collections.get(lname)
            if target_collection is None:
                target_collection = bpy.data.collections.new(lname)
                collection.children.link(target_collection)
                lod_collections[lname] = target_collection

        # Resolve local-bone -> blender bone names for skinning.
        bone_names = None
        if name_by_index and getattr(md, "boneidarray", None):
            bone_names = _resolve_part_bones(mbones, part, name_by_index)

        obj = build_blender_mesh(name, md, bone_names, context, target_collection)

        # Assign material by index (part.material // 16). The material already
        # carries its texture and, for "ol_*" names, its dark outline shading.
        mat = materials.get(part.material // 16)
        if mat is not None:
            obj.data.materials.append(mat)

        # Parent to armature + add armature modifier when skinned.
        if arm_obj is not None:
            obj.parent = arm_obj
            if bone_names:
                mod = obj.modifiers.new("Armature", "ARMATURE")
                mod.object = arm_obj

        cur_part += 1

    return mbones, parts


def _resolve_part_bones(mbones, part, name_by_index):
    """Map a part's local bone indices to blender bone names.

    Mirrors MODRuntime: if remaps exist, the part's boneremapid selects a remap
    table whose entries are global bone ids; otherwise local index == global id.
    Returns a list where position (local_index-1) holds the blender bone name.
    """
    if mbones.remaps:
        rid = part.boneremapid - 1
        if 0 <= rid < len(mbones.remaps):
            remap = mbones.remaps[rid].remaps
            return [name_by_index.get(g) for g in remap]
        return []
    # No remaps: local index i -> bone i (0-based across all bones).
    return [name_by_index.get(i) for i in range(len(mbones.bones))]


# ----------------------------------------------------------------------------
# Blender operator + menu
# ----------------------------------------------------------------------------
class IMPORT_OT_mt_framework_mod(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.mt_framework_mod"
    bl_label = "Import MT Framework Model"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".mod"
    filter_glob: StringProperty(default="*.mod;*.mod3;*.dom", options={"HIDDEN"})

    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    # Options mirror the Max rollout's "Model/Skeleton Loading" group.
    import_meshes: BoolProperty(
        name="Import Meshes", default=True,
        description="Import mesh geometry (uncheck to import skeleton only)")
    arch_64bit: BoolProperty(
        name="64-bit Architecture", default=False,
        description="File uses 64-bit pointers (matches the Max '64 bit Architecture' toggle)")
    keep_object_id: BoolProperty(
        name="Keep Object ID in name", default=False,
        description="Prefix the object index onto each mesh name")
    layer_per_lod: BoolProperty(
        name="Collection per LOD", default=False,
        description="Group meshes into a collection per LOD level")
    keep_debug_info: BoolProperty(
        name="Keep debug info in name", default=True,
        description="Append the decoder format id to each mesh name")
    import_textures: BoolProperty(
        name="Import Textures / Materials", default=True,
        description="Read the sibling .mrl and folder images, build materials, "
                    "and assign them to meshes by material index")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "arch_64bit")
        box = layout.box()
        box.label(text="Model / Skeleton Loading")
        box.prop(self, "import_meshes")
        col = box.column()
        col.enabled = self.import_meshes
        col.prop(self, "keep_object_id")
        col.prop(self, "layer_per_lod")
        col.prop(self, "keep_debug_info")
        box.prop(self, "import_textures")

    def execute(self, context):
        mtf.load_all()  # populate the decoder registry

        directory = self.directory or os.path.dirname(self.filepath)
        targets = [f.name for f in self.files] if self.files else [os.path.basename(self.filepath)]

        ok = 0
        for fname in targets:
            path = os.path.join(directory, fname)
            try:
                with open(path, "rb") as fh:
                    reader = BinaryReader(fh.read())
                options = {
                    "proc_parts": self.import_meshes,
                    "x64": self.arch_64bit,
                    "keep_object_id": self.keep_object_id,
                    "layer_per_lod": self.layer_per_lod,
                    "keep_debug_info": self.keep_debug_info,
                    "import_textures": self.import_textures,
                    "filepath": path,
                }
                mod_runtime(reader, options, context)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, "Failed on %s: %s" % (fname, exc))
                import traceback
                traceback.print_exc()

        if ok:
            self.report({"INFO"}, "Imported %d MOD file(s)." % ok)
            return {"FINISHED"}
        return {"CANCELLED"}


def menu_func_import(self, context):
    self.layout.operator(
        IMPORT_OT_mt_framework_mod.bl_idname,
        text="MT Framework Model (.mod/.mod3)")
    self.layout.operator(
        IMPORT_OT_mt_framework_lmt.bl_idname,
        text="MT Framework Animation (.lmt)")


class IMPORT_OT_mt_framework_lmt(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.mt_framework_lmt"
    bl_label = "Import MT Framework Animation"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".lmt"
    filter_glob: StringProperty(default="*.lmt;*.tml", options={"HIDDEN"})

    arch_64bit: BoolProperty(
        name="64-bit Architecture", default=False,
        description="File uses 64-bit pointers")
    import_all: BoolProperty(
        name="Import All Animations", default=True,
        description="Import every animation in the LMT as a separate Action")
    anim_id: IntProperty(
        name="Animation Index", default=0, min=0,
        description="Which animation to import when 'Import All' is off "
                    "(0-based index among valid animations)")
    flip_handedness: BoolProperty(
        name="Flip Rotation Handedness", default=False,
        description="If the pose looks twisted / limbs bend the wrong way, "
                    "enable this to conjugate the animation quaternions")
    import_location: BoolProperty(
        name="Import Bone Translation", default=True,
        description="Key per-bone position tracks too. Turn off for "
                    "rotation-only animation if a position track misbehaves")
    set_timeline: BoolProperty(
        name="Set Timeline to Animation Length", default=True,
        description="Set the scene's frame range (and each Action's range) to "
                    "the animation length, like the 3ds Max MT Framework tool")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "arch_64bit")
        layout.prop(self, "import_all")
        row = layout.row()
        row.enabled = not self.import_all
        row.prop(self, "anim_id")
        layout.prop(self, "flip_handedness")
        layout.prop(self, "import_location")
        layout.prop(self, "set_timeline")
        arm = _active_armature(context)
        box = layout.box()
        if arm is None:
            box.label(text="Select the imported armature first", icon="ERROR")
        else:
            box.label(text="Target: %s" % arm.name, icon="ARMATURE_DATA")

    def execute(self, context):
        arm = _active_armature(context)
        if arm is None:
            self.report({"ERROR"},
                        "Select the MT Framework armature before importing animation.")
            return {"CANCELLED"}
        if "mtf_remaptable" not in arm:
            self.report({"WARNING"},
                        "Armature has no MT Framework remap table; import the model "
                        "with this addon first for correct bone mapping.")
        from . import lmt_import
        single = None if self.import_all else self.anim_id
        try:
            n = lmt_import.import_lmt(
                self.filepath, arm, context, x64=self.arch_64bit,
                single_id=single, conjugate=self.flip_handedness,
                include_location=self.import_location,
                set_timeline=self.set_timeline,
                name_prefix=os.path.splitext(
                    os.path.basename(self.filepath))[0])
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "LMT import failed: %s" % exc)
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        self.report({"INFO"}, "Imported %d animation(s) as Actions." % n)
        return {"FINISHED"}



class EXPORT_OT_mt_framework_lmt(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.mt_framework_lmt"
    bl_label = "Export MT Framework Animation"
    bl_options = {"REGISTER"}

    filename_ext = ".lmt"
    filter_glob: StringProperty(default="*.lmt", options={"HIDDEN"})

    def draw(self, context):
        layout = self.layout
        layout.label(text="Exports the active Action as one LMT animation.")
        layout.label(text="Current exporter format: little-endian v55.")
        arm = _active_armature(context)
        if arm is None:
            layout.label(text="Select an armature first", icon="ERROR")
        elif not arm.animation_data or not arm.animation_data.action:
            layout.label(text="Assign an active Action first", icon="ERROR")
        else:
            layout.label(text="Action: %s" % arm.animation_data.action.name,
                         icon="ACTION")

    def execute(self, context):
        arm = _active_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select an armature before exporting LMT.")
            return {"CANCELLED"}
        try:
            from .lmt_export import export_lmt
            count = export_lmt(self.filepath, arm, context.scene)
        except Exception as exc:
            self.report({"ERROR"}, "LMT export failed: %s" % exc)
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        self.report({"INFO"}, "Exported LMT animation (%d bone tracks)." % count)
        return {"FINISHED"}


class EXPORT_OT_mt_framework_model(bpy.types.Operator, ExportHelper):
    """Create a new MOD/MOD3 binary from selected Blender mesh objects."""
    bl_idname = "export_scene.mt_framework_model"
    bl_label = "MT Framework Model (.mod / .mod3)"
    bl_options = {"REGISTER"}

    filename_ext = ".mod"
    filter_glob: StringProperty(default="*.mod;*.mod3", options={"HIDDEN"})
    target_format: EnumProperty(
        name="Target Format",
        items=[
            ("MOD_V212", "MOD v212 — MHGU", "Monster Hunter Generations Ultimate"),
            ("MOD_V230", "MOD v230 — MHXX", "Monster Hunter Double Cross"),
            ("MOD3_V237", "MOD3 v237 — Monster Hunter: World", "Native 64-bit MHW MOD3 file"),
        ],
        default="MOD_V212",
    )
    include_armature_meshes: BoolProperty(
        name="Include all meshes attached to armature",
        description="For MHGU v212, export every mesh parented to or skinned by the same armature, not only selected mesh objects",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "target_format")
        if self.target_format == "MOD_V212":
            layout.prop(self, "include_armature_meshes")
        layout.separator()
        layout.label(text="Exports to a new file; no source MOD required.", icon="INFO")
        layout.label(text="MHGU v212 requires valid game material names.")
        layout.label(text="MHXX v230 / MHW MOD3: existing static-mesh path.", icon="INFO")

    def check(self, context):
        wanted = ".mod3" if self.target_format == "MOD3_V237" else ".mod"
        changed = False
        if self.filename_ext != wanted:
            self.filename_ext = wanted
            changed = True
        wanted_filter = "*.mod3" if self.target_format == "MOD3_V237" else "*.mod"
        if self.filter_glob != wanted_filter:
            self.filter_glob = wanted_filter
            changed = True
        try:
            changed = ExportHelper.check(self, context) or changed
        except Exception:
            pass
        return changed

    def execute(self, context):
        import os
        filepath = bpy.path.abspath(self.filepath)
        ext = ".mod3" if self.target_format == "MOD3_V237" else ".mod"
        root, existing_ext = os.path.splitext(filepath)
        if existing_ext.lower() != ext:
            filepath = root + ext
        export_objects = list(context.selected_objects)
        try:
            if self.target_format == "MOD_V212" and self.include_armature_meshes:
                armature = model_export._find_export_armature(export_objects, context)
                if armature is not None:
                    # Export is model-scoped for skinned GU: gather all scene meshes
                    # that are children of the same armature or use its Armature modifier.
                    # This is confined to the export operator; Import code is unchanged.
                    related_meshes = []
                    for obj in context.scene.objects:
                        if getattr(obj, "type", None) != "MESH":
                            continue
                        is_child = getattr(obj, "parent", None) == armature
                        has_modifier = any(
                            getattr(modifier, "type", None) == "ARMATURE"
                            and getattr(modifier, "object", None) == armature
                            for modifier in getattr(obj, "modifiers", [])
                        )
                        if is_child or has_modifier:
                            related_meshes.append(obj)
                    if related_meshes:
                        export_objects = related_meshes
            result = model_export.export_selected_meshes(
                filepath, export_objects, context, self.target_format)
        except Exception as exc:
            self.report({"ERROR"}, "MT Framework model export failed: %s" % exc)
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        scope_note = " (skinned model meshes)" if self.target_format == "MOD_V212" and self.include_armature_meshes else ""
        self.report({"INFO"}, "Exported %s%s: %d mesh parts, %d vertices, %d triangles." % (
            ext, scope_note, result["meshes"], result["vertices"], result["triangles"]))
        return {"FINISHED"}


def menu_func_export(self, context):
    self.layout.operator(
        EXPORT_OT_mt_framework_model.bl_idname,
        text="MT Framework Model (.mod / .mod3)")
    self.layout.operator(
        EXPORT_OT_mt_framework_lmt.bl_idname,
        text="MT Framework Animation (.lmt)")


def _active_armature(context):
    obj = context.active_object
    if obj is not None and obj.type == "ARMATURE":
        return obj
    for o in context.selected_objects:
        if o.type == "ARMATURE":
            return o
    return None


class VIEW3D_PT_mt_framework(bpy.types.Panel):
    """Persistent sidebar panel -- the Blender analogue of the Max rollout.

    3ds Max kept an always-open utility rollout with buttons; Blender's idiomatic
    equivalent is a panel in the 3D Viewport sidebar (press N -> "MT Framework"
    tab). The buttons here just invoke the same operators, so the file browser
    still opens for picking the file -- Blender importers always go through the
    file browser rather than a custom file field.
    """

    bl_label = "MT Framework"
    bl_idname = "VIEW3D_PT_mt_framework"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MT Framework"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Model / Skeleton")
        col.operator(IMPORT_OT_mt_framework_mod.bl_idname,
                     text="Import MOD...", icon="IMPORT")
        col.operator(EXPORT_OT_mt_framework_model.bl_idname,
                     text="Export MOD...", icon="EXPORT")
        box = layout.box()
        box.label(text="Animation (LMT)", icon="ANIM")
        arm = _active_armature(context)
        if arm is None:
            box.label(text="Select armature to enable", icon="INFO")
        else:
            box.label(text="Target: %s" % arm.name)
        r = box.row()
        r.enabled = arm is not None
        r.operator(IMPORT_OT_mt_framework_lmt.bl_idname,
                   text="Import LMT...", icon="IMPORT")
        r2 = box.row()
        r2.enabled = arm is not None and arm.animation_data is not None and arm.animation_data.action is not None
        r2.operator(EXPORT_OT_mt_framework_lmt.bl_idname,
                    text="Export LMT...", icon="EXPORT")


_classes = (IMPORT_OT_mt_framework_mod, IMPORT_OT_mt_framework_lmt,
            EXPORT_OT_mt_framework_lmt, EXPORT_OT_mt_framework_model,
            VIEW3D_PT_mt_framework)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
