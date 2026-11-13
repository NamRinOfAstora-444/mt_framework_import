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

# MTF_INDEPENDENT_LMT_REPLACE_UI_V1

bl_info = {
    "name": "MT Framework (MOD/LMT) Importer",
    "author": "axestyra",
    "version": (0, 21, 28),
    "blender": (3, 0, 0),
    "location": "File > Import > MT Framework Model / Animation",
    "description": "Import/export MT Framework LMT with native BoneID 255 root-motion vector encoding.",
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

    # Create the MT root bone with index 255 using MTFCormat. LMT tracks for
    # BoneID 255 are normal pose-bone animation: they are keyed on bone255's
    # pose channels, never on Armature Object.location. This root-bone mapping
    # is retained for MHGU and DMC4 SE.
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
        # MOD3 v237 stores a direct material index; older MOD formats store
        # the index in 16-byte units.
        used = set(
            p.material if header.Version == 237 else p.material // 16
            for p in parts.parts
        )
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
        material_index = (
            part.material if header.Version == 237 else part.material // 16
        )
        mat = materials.get(material_index)
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
        name="Keep debug info in name", default=False,
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


class MTF_LMT_AnimationItem(bpy.types.PropertyGroup):
    """A source Action or a dense destination slot row."""
    action_name: StringProperty(name="Action Name")
    animation_index: IntProperty(name="Animation ID", default=-1, min=-1)
    is_loop: BoolProperty(name="Loop Segment", default=False)
    is_split_segment: BoolProperty(name="Part of Split Loop", default=False)


def _refresh_lmt_animation_items(context):
    """Refresh the searchable Source Animation list from imported Blender Actions."""
    wm = context.window_manager
    selected_name = None
    old_index = int(getattr(wm, "mtf_lmt_animation_index", 0))
    old_items = getattr(wm, "mtf_lmt_animation_items", None)
    if old_items and 0 <= old_index < len(old_items):
        selected_name = old_items[old_index].action_name

    target_armature = _active_armature(context)
    target_armature_name = target_armature.name if target_armature is not None else ""
    actions = []
    for action in bpy.data.actions:
        if bool(action.get("mtf_internal_source_companion", False)):
            continue
        owner_name = str(action.get("mtf_armature_name", "") or "")
        # Keep legacy untagged Actions visible for compatibility, but never mix
        # Actions explicitly tagged to a different rig.
        if target_armature_name and owner_name and owner_name != target_armature_name:
            continue
        try:
            animation_id = int(action.get("mtf_animation_index", -1))
        except (TypeError, ValueError):
            continue
        if animation_id < 0:
            continue
        is_loop = bool(action.get("mtf_is_loop", False))
        is_full_loop = bool(action.get("mtf_full_loop", False))
        try:
            pre_loop = int(action.get("mtf_loop_frame", 0) or 0)
            source_loop = int(action.get("mtf_source_loop_frame", 0) or 0)
        except (TypeError, ValueError):
            pre_loop = source_loop = 0
        split = not is_full_loop and (pre_loop > 0 or (is_loop and source_loop > 0))
        actions.append((animation_id, action.name.casefold(), action.name, is_loop, split))

    actions.sort(key=lambda entry: (entry[0], entry[1]))
    items = wm.mtf_lmt_animation_items
    items.clear()
    selected_index = 0
    for index, (animation_id, _sort_name, action_name, is_loop, split) in enumerate(actions):
        item = items.add()
        item.action_name = action_name
        item.animation_index = animation_id
        item.is_loop = is_loop
        item.is_split_segment = split
        if action_name == selected_name:
            selected_index = index
    wm.mtf_lmt_animation_index = selected_index if items else 0
    return len(items)


def _lmt_animation_item_search_text(item):
    """Build consistent searchable text for source and destination rows."""
    parts = [item.action_name, str(item.animation_index)]
    if item.is_split_segment:
        parts.append("LOOP" if item.is_loop else "PRE-LOOP")
        parts.append("PRE" if not item.is_loop else "LOOP")
    return " ".join(parts).casefold()


def _on_lmt_animation_search_changed(wm, context):
    search = (wm.mtf_lmt_animation_search or "").casefold().strip()
    items = wm.mtf_lmt_animation_items
    current = int(wm.mtf_lmt_animation_index)
    if 0 <= current < len(items):
        current_item = items[current]
        searchable = _lmt_animation_item_search_text(current_item)
        if search in searchable:
            return
    for index, item in enumerate(items):
        searchable = _lmt_animation_item_search_text(item)
        if not search or search in searchable:
            wm.mtf_lmt_animation_index = index
            return


def _on_lmt_destination_search_changed(wm, context):
    search = (wm.mtf_lmt_destination_search or "").casefold().strip()
    items = wm.mtf_lmt_destination_items
    current = int(wm.mtf_lmt_destination_index)
    if 0 <= current < len(items):
        current_item = items[current]
        searchable = _lmt_animation_item_search_text(current_item)
        if search in searchable:
            return
    for index, item in enumerate(items):
        searchable = _lmt_animation_item_search_text(item)
        if not search or search in searchable:
            wm.mtf_lmt_destination_index = index
            return


class UI_UL_mtf_lmt_animations(bpy.types.UIList):
    """Searchable list of named Actions / LMT animation targets."""
    def filter_items(self, context, data, property_name):
        items = getattr(data, property_name)
        wm = context.window_manager
        search_value = (wm.mtf_lmt_destination_search if property_name == "mtf_lmt_destination_items"
                        else wm.mtf_lmt_animation_search)
        search = (search_value or "").casefold().strip()
        flags = []
        for item in items:
            searchable = _lmt_animation_item_search_text(item)
            visible = not search or search in searchable
            flags.append(self.bitflag_filter_item if visible else 0)
        return flags, []

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_property, index):
        row = layout.row(align=True)
        row.label(text=item.action_name, icon="ACTION")
        row.label(text="ID %03d" % item.animation_index)
        if item.is_split_segment:
            row.label(text="LOOP" if item.is_loop else "PRE-LOOP",
                      icon="FILE_REFRESH")



def _refresh_lmt_destination_items(context):
    # Read populated animation slots from the chosen Destination LMT.
    wm = context.window_manager
    raw_path = (wm.mtf_lmt_destination_path or "").strip()
    if len(raw_path) >= 2 and raw_path[0] == raw_path[-1] and raw_path[0] in ("\"", "'"):
        raw_path = raw_path[1:-1].strip()
    path = os.path.normpath(
        bpy.path.abspath(os.path.expandvars(os.path.expanduser(raw_path)))
    ) if raw_path else ""
    if not path or not os.path.isfile(path):
        raise ValueError("Destination LMT path is empty or does not exist.")
    if not path.lower().endswith(".lmt"):
        raise ValueError("Destination file must have the .lmt extension.")
    from .lmt_export import list_lmt_animations
    slots = list_lmt_animations(path)
    items = wm.mtf_lmt_destination_items
    items.clear()
    for slot in slots:
        animation_id = int(slot["animation_index"])
        animation_name = "Animation %03d" % animation_id
        if slot.get("is_split_segment", False):
            # PRE-LOOP and LOOP are two timeline regions in the same LMT slot.
            # Keep the same dense animation ID so either row resolves to that
            # one physical destination clip; export replaces the paired clip.
            for is_loop in (False, True):
                item = items.add()
                item.action_name = animation_name
                item.animation_index = animation_id
                item.is_loop = is_loop
                item.is_split_segment = True
        else:
            item = items.add()
            item.action_name = animation_name
            item.animation_index = animation_id
            item.is_loop = False
            item.is_split_segment = False
    wm.mtf_lmt_destination_index = 0
    wm.mtf_lmt_destination_path = path
    wm.mtf_lmt_destination_loaded_path = path
    return len(slots)


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
        name="Set Timeline to Animation Length", default=False,
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
            box.label(text="Armature: %s" % arm.name, icon="ARMATURE_DATA")

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
    bl_label = "MT Framework Animation"
    bl_options = {"REGISTER"}

    filename_ext = ".lmt"
    filter_glob: StringProperty(default="*.lmt", options={"HIDDEN"})
    arch_64bit: BoolProperty(
        name="64-bit Pointer Layout", default=False,
        description="Reserved for newer LMT variants; currently unsupported by safe export")
    selected_bones_only: BoolProperty(
        name="Export Selected Bones Only", default=False,
        description="Only rewrite destination tracks belonging to selected pose bones")

    def invoke(self, context, event):
        _refresh_lmt_animation_items(context)
        wm = context.window_manager
        if wm.mtf_lmt_destination_path and not wm.mtf_lmt_destination_items:
            try:
                _refresh_lmt_destination_items(context)
            except Exception as exc:
                print("Destination LMT list refresh: %s" % exc)
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager

        source_box = layout.box()
        source_box.label(text="SOURCE ANIMATION — Imported Blender Actions", icon="ACTION")
        arm = _active_armature(context)
        if arm is not None:
            source_box.label(text="Actions tagged to: %s" % arm.name, icon="ARMATURE_DATA")
        source_box.prop(wm, "mtf_lmt_animation_search", text="", icon="VIEWZOOM")
        source_box.template_list(
            "UI_UL_mtf_lmt_animations", "mtf_lmt_source_actions",
            wm, "mtf_lmt_animation_items",
            wm, "mtf_lmt_animation_index",
            rows=5,
        )
        source_items = wm.mtf_lmt_animation_items
        source_index = int(wm.mtf_lmt_animation_index)
        source_search = (wm.mtf_lmt_animation_search or "").casefold().strip()
        source_item = None
        if 0 <= source_index < len(source_items):
            candidate = source_items[source_index]
            searchable = "%s %d" % (candidate.action_name, candidate.animation_index)
            if not source_search or source_search in searchable.casefold():
                source_item = candidate
        if source_item is not None:
            source_box.label(
                text="Source: %s (source ID %d)" %
                     (source_item.action_name, source_item.animation_index),
                icon="CHECKMARK",
            )
            if source_item.is_split_segment:
                source_box.label(text="PRE-LOOP + LOOP will be combined from the two source Actions.",
                                 icon="INFO")
        elif not source_items:
            source_box.label(text="No imported Actions. Import the source LMT animation first.",
                             icon="ERROR")

        dest_box = layout.box()
        dest_box.label(text="DESTINATION LMT — Container to modify", icon="FILE_FOLDER")
        dest_box.prop(wm, "mtf_lmt_destination_path", text="Destination")
        row = dest_box.row(align=True)
        row.operator("wm.mtf_lmt_use_selected_file",
                     text="Use Selected File as Destination", icon="FILE_TICK")
        row.operator("wm.mtf_lmt_refresh_destination",
                     text="Load / Refresh Slots", icon="FILE_REFRESH")
        dest_box.prop(wm, "mtf_lmt_destination_search", text="", icon="VIEWZOOM")
        dest_box.template_list(
            "UI_UL_mtf_lmt_animations", "mtf_lmt_destination_slots",
            wm, "mtf_lmt_destination_items",
            wm, "mtf_lmt_destination_index",
            rows=6,
        )
        dest_items = wm.mtf_lmt_destination_items
        dest_index = int(wm.mtf_lmt_destination_index)
        dest_search = (wm.mtf_lmt_destination_search or "").casefold().strip()
        dest_item = None
        if 0 <= dest_index < len(dest_items):
            candidate = dest_items[dest_index]
            searchable = _lmt_animation_item_search_text(candidate)
            if not dest_search or dest_search in searchable:
                dest_item = candidate
        if dest_item is not None:
            segment_label = " — %s" % ("LOOP" if dest_item.is_loop else "PRE-LOOP") \
                if dest_item.is_split_segment else ""
            dest_box.label(text="Target: Destination Animation ID %d%s" %
                           (dest_item.animation_index, segment_label), icon="CHECKMARK")
            if dest_item.is_split_segment:
                dest_box.label(
                    text="PRE-LOOP and LOOP share one LMT slot; Replace writes the paired animation.",
                    icon="INFO",
                )
        elif not dest_items:
            dest_box.label(text="Choose a Destination LMT and load its populated slots.",
                           icon="ERROR")

        layout.separator()
        layout.prop(self, "arch_64bit")
        layout.prop(self, "selected_bones_only")
        layout.label(text="Source ID and Destination ID are independent.", icon="INFO")
        layout.label(text="Output is a new file; Destination LMT is never overwritten.")
        layout.label(text="Native LMT v51+; 64-bit layouts are not supported.")
        if arm is None:
            layout.label(text="Select an armature first", icon="ERROR")

    def execute(self, context):
        arm = _active_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select an armature before exporting LMT.")
            return {"CANCELLED"}

        selected_names = [pb.name for pb in arm.pose.bones if pb.bone.select]
        if self.selected_bones_only and not selected_names:
            self.report({"ERROR"}, "Select the edited bone(s) in Pose Mode before exporting.")
            return {"CANCELLED"}

        wm = context.window_manager
        source_items = wm.mtf_lmt_animation_items
        source_index = int(wm.mtf_lmt_animation_index)
        source_search = (wm.mtf_lmt_animation_search or "").casefold().strip()
        if not source_items or not (0 <= source_index < len(source_items)):
            self.report({"ERROR"}, "Choose a Source Animation Action. Import the source LMT first.")
            return {"CANCELLED"}
        source_item = source_items[source_index]
        source_query = _lmt_animation_item_search_text(source_item)
        if source_search and source_search not in source_query:
            self.report({"ERROR"}, "Choose a visible Source Action from the search results.")
            return {"CANCELLED"}

        dest_items = wm.mtf_lmt_destination_items
        dest_index = int(wm.mtf_lmt_destination_index)
        dest_search = (wm.mtf_lmt_destination_search or "").casefold().strip()
        if not dest_items or not (0 <= dest_index < len(dest_items)):
            self.report({"ERROR"}, "Choose and load a Destination LMT animation slot.")
            return {"CANCELLED"}
        dest_item = dest_items[dest_index]
        dest_query = _lmt_animation_item_search_text(dest_item)
        if dest_search and dest_search not in dest_query:
            self.report({"ERROR"}, "Choose a visible Destination slot from the search results.")
            return {"CANCELLED"}

        raw_dest = (wm.mtf_lmt_destination_path or "").strip()
        if len(raw_dest) >= 2 and raw_dest[0] == raw_dest[-1] and raw_dest[0] in ("\"", "'"):
            raw_dest = raw_dest[1:-1].strip()
        destination_path = os.path.normpath(
            bpy.path.abspath(os.path.expandvars(os.path.expanduser(raw_dest)))
        ) if raw_dest else ""
        if not destination_path or not os.path.isfile(destination_path):
            self.report({"ERROR"}, "Destination LMT path is empty or does not exist.")
            return {"CANCELLED"}
        loaded_path = os.path.normpath(bpy.path.abspath(wm.mtf_lmt_destination_loaded_path)) if wm.mtf_lmt_destination_loaded_path else ""
        if not loaded_path or loaded_path != destination_path:
            self.report({"ERROR"}, "Destination path changed. Click 'Load / Refresh Slots' before exporting.")
            return {"CANCELLED"}

        try:
            from .lmt_export import export_lmt
            count = export_lmt(
                self.filepath, arm, context.scene,
                destination_lmt=destination_path,
                destination_animation_id=dest_item.animation_index,
                source_action_name=source_item.action_name,
                x64=self.arch_64bit,
                selected_bones=selected_names if self.selected_bones_only else None,
            )
        except Exception as exc:
            self.report({"ERROR"}, "LMT export failed: %s" % exc)
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}

        self.report({"INFO"}, "Replaced Destination ID %d using '%s'; %d track(s) updated." %
                    (dest_item.animation_index, source_item.action_name, count))
        return {"FINISHED"}

class WM_OT_mtf_lmt_use_selected_file(bpy.types.Operator):
    'Use the active file-browser entry as Destination LMT.'
    bl_idname = "wm.mtf_lmt_use_selected_file"
    bl_label = "Use Selected File as Destination"
    bl_description = "Load the active .lmt file-browser entry as the replacement destination"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        space = context.space_data
        if not space or getattr(space, "type", None) != "FILE_BROWSER":
            self.report({"ERROR"}, "Use this button from the Export LMT file browser.")
            return {"CANCELLED"}
        params = getattr(space, "params", None)
        if params is None:
            self.report({"ERROR"}, "Could not read the file-browser selection.")
            return {"CANCELLED"}
        directory = getattr(params, "directory", "") or ""
        if isinstance(directory, bytes):
            directory = os.fsdecode(directory)
        filename = (getattr(params, "filename", "") or "").strip()
        if not filename:
            self.report({"ERROR"}, "Select an existing .lmt file in the left file list first.")
            return {"CANCELLED"}
        path = filename if os.path.isabs(filename) else os.path.join(directory, filename)
        path = os.path.normpath(bpy.path.abspath(os.path.expandvars(os.path.expanduser(path))))
        if not os.path.isfile(path) or not path.lower().endswith(".lmt"):
            self.report({"ERROR"}, "Select an existing .lmt file in the left file list.")
            return {"CANCELLED"}
        context.window_manager.mtf_lmt_destination_path = path
        try:
            count = _refresh_lmt_destination_items(context)
        except Exception as exc:
            self.report({"ERROR"}, "Could not load Destination LMT: %s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "Loaded %d destination animation slot(s) from %s." %
                    (count, os.path.basename(path)))
        return {"FINISHED"}


class WM_OT_mtf_lmt_refresh_destination(bpy.types.Operator):
    'Refresh destination slots from the Destination LMT path field.'
    bl_idname = "wm.mtf_lmt_refresh_destination"
    bl_label = "Load / Refresh Destination Slots"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        try:
            count = _refresh_lmt_destination_items(context)
        except Exception as exc:
            self.report({"ERROR"}, "Could not load Destination LMT: %s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "Loaded %d destination animation slot(s)." % count)
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
        # The export dialog chooses its target from the searchable Action list;
        # an active Action on the armature is no longer required.
        r2.enabled = arm is not None
        r2.operator(EXPORT_OT_mt_framework_lmt.bl_idname,
                    text="Export LMT...", icon="EXPORT")


# ---------------------------------------------------------------------------
# Bone ID Manager
# ---------------------------------------------------------------------------
def _bone_manager_current_id(arm, bone):
    """Return the LMT-facing Bone ID for a Blender armature bone."""
    if arm is None or bone is None:
        return None

    root_names = {
        str(arm.get("mtf_root_bone", "") or ""),
        str(arm.get("mtf_game_root_bone", "") or ""),
    }
    try:
        scene_index = int(bone.get("mtf_index", -1))
    except (TypeError, ValueError):
        scene_index = -1

    if scene_index == 255 or bone.name in root_names:
        return 255

    remap = list(arm.get("mtf_remaptable", []))
    if remap:
        # A MOD-imported rig stores its original skeleton index on the Bone;
        # the remap table converts LMT IDs into those skeleton indices.
        for lmt_id, mapped_index in enumerate(remap):
            try:
                mapped_index = int(mapped_index)
            except (TypeError, ValueError):
                continue
            if lmt_id != 255 and mapped_index == scene_index and scene_index >= 0:
                return lmt_id
        return None

    return scene_index if 0 <= scene_index <= 254 else None


def _bone_manager_get_selected(context):
    """Resolve the bone currently selected in the Bone ID Manager list."""
    arm = _active_armature(context)
    if arm is None or arm.type != "ARMATURE":
        return None, None

    try:
        index = int(arm.data.mtf_bone_id_index)
    except (AttributeError, TypeError, ValueError):
        index = -1

    bones = arm.data.bones
    if 0 <= index < len(bones):
        return arm, bones[index]

    active = bones.active
    if active is not None:
        return arm, active
    return arm, None


def _bone_manager_assign_id(arm, bone, bone_id):
    """Assign or edit one LMT Bone ID without changing MOD skeleton indices."""
    bone_id = int(bone_id)
    if not 0 <= bone_id <= 255:
        raise ValueError("Bone ID must be between 0 and 255.")

    try:
        scene_index = int(bone.get("mtf_index", -1))
    except (TypeError, ValueError):
        scene_index = -1

    root_name = str(arm.get("mtf_root_bone", "") or "")
    game_root_name = str(arm.get("mtf_game_root_bone", "") or "")
    is_root = scene_index == 255 or bone.name in {root_name, game_root_name}
    remap = list(arm.get("mtf_remaptable", []))

    if bone_id == 255:
        # Root motion is resolved by a dedicated code path in both LMT import
        # and export. Preserve an already-designated game root as-is; for a
        # custom rig, assigning 255 designates this bone as its root controller.
        if is_root:
            arm["mtf_lmt_rootid"] = 255
            if not root_name and scene_index == 255:
                arm["mtf_root_bone"] = bone.name
            return
        if remap and scene_index != 255:
            raise ValueError("Bone ID 255 is reserved for the MOD root bone (index 255).")
        for other in arm.data.bones:
            if other == bone:
                continue
            try:
                other_id = int(other.get("mtf_index", -1))
            except (TypeError, ValueError):
                other_id = -1
            if other_id == 255 or other.name in {root_name, game_root_name}:
                raise ValueError("Bone ID 255 is already assigned to %s." % other.name)
        bone["mtf_index"] = 255
        arm["mtf_root_bone"] = bone.name
        arm["mtf_lmt_rootid"] = 255
        return

    if is_root:
        raise ValueError("Bone ID 255 is reserved for the MT root bone.")

    if remap:
        # Preserve the MOD skeleton index in Bone["mtf_index"].  Only edit the
        # LMT-ID -> skeleton-index table consumed by lmt_import.py/export.py.
        if scene_index < 0 or scene_index == 255:
            raise ValueError("This bone has no valid MOD skeleton index for remapping.")

        if bone_id < len(remap):
            try:
                occupied = int(remap[bone_id])
            except (TypeError, ValueError):
                occupied = 255
            if occupied not in (255, scene_index):
                other = None
                for candidate in arm.data.bones:
                    try:
                        candidate_index = int(candidate.get("mtf_index", -1))
                    except (TypeError, ValueError):
                        candidate_index = -1
                    if candidate_index == occupied:
                        other = candidate.name
                        break
                raise ValueError("Bone ID %d is already mapped to %s." %
                                 (bone_id, other or ("skeleton index %d" % occupied)))

        # Keep this UI one-ID-per-bone: remove aliases for this scene bone,
        # then write the requested LMT ID.  All unrelated mappings stay intact.
        for index, mapped in enumerate(remap):
            try:
                mapped = int(mapped)
            except (TypeError, ValueError):
                continue
            if index != bone_id and index != 255 and mapped == scene_index:
                remap[index] = 255

        if bone_id >= len(remap):
            remap.extend([255] * (bone_id + 1 - len(remap)))
        remap[bone_id] = scene_index
        arm["mtf_remaptable"] = remap
        return

    # A custom Blender armature has no MOD remap table: Bone["mtf_index"] is
    # the direct persistent LMT ID used by the existing importer and exporter.
    for other in arm.data.bones:
        if other == bone:
            continue
        try:
            other_id = int(other.get("mtf_index", -1))
        except (TypeError, ValueError):
            other_id = -1
        if other_id == bone_id:
            raise ValueError("Bone ID %d is already assigned to %s." %
                             (bone_id, other.name))

    bone["mtf_index"] = bone_id


def _bone_manager_delete_id(arm, bone):
    current_id = _bone_manager_current_id(arm, bone)
    if current_id is None:
        raise ValueError("The selected bone does not have a Bone ID to delete.")
    if current_id == 255:
        raise ValueError("Bone ID 255 is the root-motion ID and cannot be deleted.")

    remap = list(arm.get("mtf_remaptable", []))
    if remap:
        try:
            scene_index = int(bone.get("mtf_index", -1))
        except (TypeError, ValueError):
            scene_index = -1
        changed = False
        for index, mapped in enumerate(remap):
            try:
                mapped = int(mapped)
            except (TypeError, ValueError):
                continue
            if index != 255 and mapped == scene_index:
                remap[index] = 255
                changed = True
        if changed:
            arm["mtf_remaptable"] = remap
    elif "mtf_index" in bone:
        del bone["mtf_index"]


def _on_bone_id_index_changed(arm_data, context):
    """Keep the Pose Mode active bone in sync with the list selection."""
    if context is None or getattr(context, "mode", None) != "POSE":
        return
    arm = _active_armature(context)
    if arm is None or arm.data != arm_data:
        return

    try:
        index = int(arm_data.mtf_bone_id_index)
        if not 0 <= index < len(arm_data.bones):
            return
        target = arm_data.bones[index]
        for bone in arm_data.bones:
            bone.select = (bone == target)
        arm_data.bones.active = target
        current_id = _bone_manager_current_id(arm, target)
        if hasattr(context.window_manager, "mtf_bone_id_value"):
            context.window_manager.mtf_bone_id_value = (
                current_id if current_id is not None else 0)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return


class UI_UL_mtf_bone_ids(bpy.types.UIList):
    """Searchable Bone list showing name and its LMT-facing ID."""
    def filter_items(self, context, data, property_name):
        bones = getattr(data, property_name)
        wm = context.window_manager
        search = (wm.mtf_bone_id_search or "").casefold().strip()
        arm = _active_armature(context)
        flags = []
        for bone in bones:
            current_id = _bone_manager_current_id(arm, bone)
            id_text = str(current_id) if current_id is not None else "no id"
            searchable = (bone.name + " " + id_text).casefold()
            flags.append(self.bitflag_filter_item if not search or search in searchable else 0)
        return flags, []

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_property, index):
        arm = _active_armature(context)
        current_id = _bone_manager_current_id(arm, item)
        row = layout.row(align=True)
        row.label(text=item.name, icon="BONE_DATA")
        row.label(text=("ID %03d" % current_id) if current_id is not None else "No ID")


class WM_OT_mtf_bone_id_apply(bpy.types.Operator):
    bl_idname = "wm.mtf_bone_id_apply"
    bl_label = "Apply Bone ID"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "POSE":
            self.report({"ERROR"}, "Switch the Armature to Pose Mode first.")
            return {"CANCELLED"}
        arm, bone = _bone_manager_get_selected(context)
        if arm is None or bone is None:
            self.report({"ERROR"}, "Select a bone from the Bone ID list.")
            return {"CANCELLED"}
        try:
            _bone_manager_assign_id(
                arm, bone, context.window_manager.mtf_bone_id_value)
        except (TypeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Bone ID saved for %s." % bone.name)
        if context.area:
            context.area.tag_redraw()
        return {"FINISHED"}


class WM_OT_mtf_bone_id_remove(bpy.types.Operator):
    bl_idname = "wm.mtf_bone_id_remove"
    bl_label = "Delete Bone ID"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "POSE":
            self.report({"ERROR"}, "Switch the Armature to Pose Mode first.")
            return {"CANCELLED"}
        arm, bone = _bone_manager_get_selected(context)
        if arm is None or bone is None:
            self.report({"ERROR"}, "Select a bone from the Bone ID list.")
            return {"CANCELLED"}
        try:
            _bone_manager_delete_id(arm, bone)
        except (TypeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        context.window_manager.mtf_bone_id_value = 0
        self.report({"INFO"}, "Bone ID removed from %s." % bone.name)
        if context.area:
            context.area.tag_redraw()
        return {"FINISHED"}


class VIEW3D_PT_mtf_bone_id_manager(bpy.types.Panel):
    bl_label = "Bone ID Manager"
    bl_idname = "VIEW3D_PT_mtf_bone_id_manager"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MT Framework"

    def draw(self, context):
        layout = self.layout
        arm = _active_armature(context)
        if arm is None:
            layout.label(text="Select an Armature.", icon="INFO")
            return
        if context.mode != "POSE":
            layout.label(text="Switch the Armature to Pose Mode to manage Bone IDs.",
                         icon="INFO")
            return

        bones = arm.data.bones
        if not bones:
            layout.label(text="This Armature has no bones.", icon="INFO")
            return

        # Keep the manager's active row synchronized with the Pose viewport.
        try:
            active_index = int(arm.data.mtf_bone_id_index)
        except (AttributeError, TypeError, ValueError):
            active_index = -1
        if bones.active is not None:
            try:
                viewport_index = list(bones).index(bones.active)
                if viewport_index != active_index:
                    arm.data.mtf_bone_id_index = viewport_index
                    active_index = viewport_index
            except (ValueError, AttributeError, TypeError):
                pass
        if not 0 <= active_index < len(bones) and bones.active is not None:
            try:
                arm.data.mtf_bone_id_index = list(bones).index(bones.active)
                active_index = int(arm.data.mtf_bone_id_index)
            except (ValueError, AttributeError, TypeError):
                active_index = -1

        wm = context.window_manager
        layout.prop(wm, "mtf_bone_id_search", text="", icon="VIEWZOOM")
        layout.template_list(
            "UI_UL_mtf_bone_ids", "mtf_bone_id_manager",
            arm.data, "bones", arm.data, "mtf_bone_id_index",
            # Keep the action buttons visible in Blender's narrow N-panel.
            rows=5, maxrows=6,
        )

        selected = (bones[active_index]
                    if 0 <= active_index < len(bones) else bones.active)
        if selected is None:
            layout.label(text="Select a bone from the list.", icon="INFO")
            return

        current_id = _bone_manager_current_id(arm, selected)
        box = layout.box()
        box.label(text="Selected Bone", icon="BONE_DATA")
        box.label(text=selected.name)
        box.label(text=("Current ID: %03d" % current_id)
                  if current_id is not None else "Current ID: Not assigned")
        box.prop(wm, "mtf_bone_id_value", text="Bone ID")

        row = box.row(align=True)
        row.operator(
            WM_OT_mtf_bone_id_apply.bl_idname,
            text=("Update" if current_id is not None else "Add ID"),
            icon=("FILE_TICK" if current_id is not None else "ADD"),
        )
        delete_col = row.row(align=True)
        delete_col.enabled = current_id is not None and current_id != 255
        delete_col.operator(
            WM_OT_mtf_bone_id_remove.bl_idname,
            text="Delete", icon="TRASH",
        )


_classes = (IMPORT_OT_mt_framework_mod, IMPORT_OT_mt_framework_lmt,
            MTF_LMT_AnimationItem, UI_UL_mtf_lmt_animations,
            WM_OT_mtf_lmt_use_selected_file, WM_OT_mtf_lmt_refresh_destination,
            UI_UL_mtf_bone_ids, WM_OT_mtf_bone_id_apply,
            WM_OT_mtf_bone_id_remove,
            EXPORT_OT_mt_framework_lmt, EXPORT_OT_mt_framework_model,
            VIEW3D_PT_mt_framework, VIEW3D_PT_mtf_bone_id_manager)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Armature.mtf_bone_id_index = IntProperty(
        name="Bone ID List Selection", default=0, min=0,
        update=_on_bone_id_index_changed,
        options={"HIDDEN"},
    )
    bpy.types.WindowManager.mtf_bone_id_search = StringProperty(
        name="Search Bones", default="", maxlen=256,
    )
    bpy.types.WindowManager.mtf_bone_id_value = IntProperty(
        name="Bone ID", default=0, min=0, max=255,
        description="LMT Bone ID for the selected Armature bone.",
    )
    # Source Actions are already imported into Blender; the source LMT file is
    # no longer required after import. Destination LMT is a separate container.
    bpy.types.WindowManager.mtf_lmt_destination_path = StringProperty(
        name="Destination LMT", default="", maxlen=4096,
        description="LMT file whose selected animation slot will be replaced.",
    )
    bpy.types.WindowManager.mtf_lmt_destination_loaded_path = StringProperty(
        name="Loaded Destination LMT", default="", maxlen=4096,
        options={"HIDDEN"},
    )
    bpy.types.WindowManager.mtf_lmt_animation_items = CollectionProperty(
        type=MTF_LMT_AnimationItem,
    )
    bpy.types.WindowManager.mtf_lmt_animation_index = IntProperty(
        name="Source Action", default=0, min=0,
    )
    bpy.types.WindowManager.mtf_lmt_animation_search = StringProperty(
        name="Search Source Actions", default="", maxlen=256,
        update=_on_lmt_animation_search_changed,
    )
    bpy.types.WindowManager.mtf_lmt_destination_items = CollectionProperty(
        type=MTF_LMT_AnimationItem,
    )
    bpy.types.WindowManager.mtf_lmt_destination_index = IntProperty(
        name="Destination Animation", default=0, min=0,
    )
    bpy.types.WindowManager.mtf_lmt_destination_search = StringProperty(
        name="Search Destination Slots", default="", maxlen=256,
        update=_on_lmt_destination_search_changed,
    )
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for prop_name in (
        "mtf_bone_id_search",
        "mtf_bone_id_value",
        "mtf_lmt_destination_search",
        "mtf_lmt_destination_index",
        "mtf_lmt_destination_items",
        "mtf_lmt_destination_path",
        "mtf_lmt_destination_loaded_path",
        "mtf_lmt_animation_search",
        "mtf_lmt_animation_index",
        "mtf_lmt_animation_items",
    ):
        if hasattr(bpy.types.WindowManager, prop_name):
            delattr(bpy.types.WindowManager, prop_name)
    if hasattr(bpy.types.Armature, "mtf_bone_id_index"):
        del bpy.types.Armature.mtf_bone_id_index
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
