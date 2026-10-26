"""
mod_format.py
Port of the MOD (model) structs from "MT Framework (LMT, MOD).ms":
  MODHeader, MODBoneInfo, MODBoneInfo2, MODBoneRemap, MODBoneRemapv2,
  MODBones (matrix + remap reading; the Max scene-building `generate` is
  replaced by an armature builder in mod_import.py), and the mesh-part
  structs MODMeshPartv1 / v2 / v3 plus the base MODVertexBuffer.

Field order and every fseek are kept byte-for-byte identical to the original,
because these offsets are the part that must match the real files exactly.
"""

import mathutils

from .easystuff.read_stream import (
    collect_tris, collect_be_tris, collect_strips, collect_be_strips,
)
from .easystuff.unimesh_loader import cormat


class MODHeader:
    """Port of struct MODHeader."""

    def __init__(self, reader, stresstest=False, x64=False):
        self.fl = reader
        self.st = stresstest
        self.x64 = x64
        self.bendian = False
        self.version2 = None
        self.BoneMapCount = None
        self.Vertex2Offset = None
        self.bbmin = mathutils.Vector((0.0, 0.0, 0.0))
        self.bbmax = None
        self.vtbuffscale = mathutils.Vector((1.0, 1.0, 1.0))
        self.TextureCount = None
        self.bbsphereposition = None
        self.bbsphereradius = None
        self.unkOffset2 = None

        fl = reader
        self.Id = fl.read_long(signed=False)

        if self.Id == 0x4D4F4400:  # big-endian "MOD\0"
            self.version2 = fl.read_byte(signed=True)
            self.Version = fl.read_byte(signed=False)
            if self.version2 < 0:
                self.Version += 0xFF  # RE6 PS3
            self.bendian = True
            self.BoneCount = fl.read_be_short()
            self.MeshCount = fl.read_be_short()
            self.MaterialCount = fl.read_be_short()
            self.VertexCount = fl.read_be_long()
            self.FaceCount = fl.read_be_long()
            self.VertexIds = fl.read_be_long()
            self.VertexBufferSize = fl.read_be_long()
            self.SecondBufferSize = fl.read_be_long()
            if self.Version < 190:
                self.TextureCount = fl.read_be_long()
            self.GroupCount = fl.read_be_pointer(x64)
            if self.Version < 190 or self.Version > 220:
                self.BoneMapCount = fl.read_be_pointer(x64)
            self.BonesOffset = fl.read_be_pointer(x64)
            self.GroupOffset = fl.read_be_pointer(x64)
            self.TextureOffset = fl.read_be_pointer(x64)
            self.MeshOffset = fl.read_be_pointer(x64)
            self.VertexOffset = fl.read_be_pointer(x64)
            if self.Version < 190:
                self.Vertex2Offset = fl.read_be_long()
            self.FacesOffset = fl.read_be_pointer(x64)
            self.unkOffset = fl.read_be_pointer(x64)
            if self.Version < 190:
                self.unkOffset2 = fl.read_be_long()
                self.bbsphereposition = fl.read_be_vector3()
                self.bbsphereradius = fl.read_be_float()
                self.bbmin = fl.read_be_vector3(); fl.read_long()
                self.bbmax = fl.read_be_vector3(); fl.read_long()
                self.vtbuffscale = mathutils.Vector(
                    tuple(self.bbmax[s] - self.bbmin[s] for s in range(3)))

        elif self.Id == 0x444F4D:  # little-endian "MOD"
            self.Version = fl.read_byte(signed=False)
            self.version2 = fl.read_byte(signed=True)
            self.BoneCount = fl.read_short(signed=False)
            self.MeshCount = fl.read_short(signed=False)
            self.MaterialCount = fl.read_short(signed=False)
            self.VertexCount = fl.read_long(signed=False)
            self.FaceCount = fl.read_long(signed=False)
            self.VertexIds = fl.read_long(signed=False)
            self.VertexBufferSize = fl.read_long(signed=False)
            self.SecondBufferSize = fl.read_long(signed=False)
            if self.Version < 190:
                self.TextureCount = fl.read_long(signed=False)
            self.GroupCount = fl.read_pointer(x64)
            if self.Version < 190 or self.Version > 220:
                self.BoneMapCount = fl.read_pointer(x64)
            self.BonesOffset = fl.read_pointer(x64)
            self.GroupOffset = fl.read_pointer(x64)
            self.TextureOffset = fl.read_pointer(x64)
            self.MeshOffset = fl.read_pointer(x64)
            self.VertexOffset = fl.read_pointer(x64)
            if self.Version < 190:
                self.Vertex2Offset = fl.read_long(signed=False)
            self.FacesOffset = fl.read_pointer(x64)
            self.unkOffset = fl.read_pointer(x64)
            if self.Version < 190:
                self.unkOffset2 = fl.read_long(signed=False)
                self.bbsphereposition = fl.read_vector3()
                self.bbsphereradius = fl.read_float()
                self.bbmin = fl.read_vector3(); fl.read_long()
                self.bbmax = fl.read_vector3(); fl.read_long()
                self.vtbuffscale = mathutils.Vector(
                    tuple(self.bbmax[s] - self.bbmin[s] for s in range(3)))
        else:
            raise ValueError("Invalid Header (not a MOD file)")

        if self.Version == 237:
            self.BoneMapCount = None


def _read_matrix44(header):
    """Port of MODBones.readmatrix44 -> returns a mathutils.Matrix (row-major
    as in Max: row1..row4 each a vec3 followed by a discarded long)."""
    fl = header.fl
    rows = []
    rd = fl.read_be_vector3 if header.bendian else fl.read_vector3
    for _ in range(4):
        rows.append(rd())
        fl.read_long()
    m = mathutils.Matrix((
        (rows[0][0], rows[0][1], rows[0][2], 0.0),
        (rows[1][0], rows[1][1], rows[1][2], 0.0),
        (rows[2][0], rows[2][1], rows[2][2], 0.0),
        (rows[3][0], rows[3][1], rows[3][2], 1.0),
    ))
    return m  # Max row-vector matrix; transpose before use as a Blender matrix.


class MODBoneInfo:
    def __init__(self, header):
        fl = header.fl
        self.id = fl.read_byte(signed=False)
        self.parentid = fl.read_byte(signed=False)
        self.child = fl.read_byte(signed=False)
        self.unk = fl.read_byte(signed=False)
        fl.skip(20)


class MODBoneInfo2:
    def __init__(self, header):
        fl = header.fl
        self.id = fl.read_short(signed=False)
        self.parentid = fl.read_byte(signed=False)
        self.child = fl.read_byte(signed=False)
        fl.skip(20)


class MODBoneRemap:
    def __init__(self, header, cntpad=32):
        fl = header.fl
        count = fl.read_be_long() if header.bendian else fl.read_long(signed=False)
        self.remaps = [fl.read_byte(signed=False) for _ in range(count)]
        if count < cntpad:
            fl.skip(cntpad - count)


class MODBoneRemapv2:
    def __init__(self, header):
        fl = header.fl
        count = fl.read_byte(signed=False)
        self.remaps = [fl.read_byte(signed=False) for _ in range(count)]
        if count < 42:
            fl.skip(42 - count)


class MODBones:
    """Reads bone infos, local + absolute matrices and the remap tables.

    The Max `generate()` (which spawned dummy helpers + IK) is intentionally
    NOT ported here; mod_import.build_armature consumes this data instead.
    """

    def __init__(self, header, skipboneloading=False):
        self.headerref = header
        self.bones = []
        self.lmatrices = []   # local matrices (Max row-major)
        self.amatrices = []   # absolute matrices
        self.remaptable = []
        self.remaps = []
        self.skipboneloading = skipboneloading

        fl = header.fl
        if skipboneloading:
            fl.skip(24 * header.BoneCount + 128 * header.BoneCount)
        else:
            info_cls = MODBoneInfo2 if header.Version == 237 else MODBoneInfo
            for _ in range(header.BoneCount):
                self.bones.append(info_cls(header))
            for _ in range(header.BoneCount):
                self.lmatrices.append(_read_matrix44(header))
            for _ in range(header.BoneCount):
                self.amatrices.append(_read_matrix44(header))

        remaptablesize = 512 if header.Version == 137 else 256
        for _ in range(remaptablesize):
            self.remaptable.append(fl.read_byte(signed=False))

        if header.BoneMapCount is not None:
            v = header.Version
            if v > 250:
                for _ in range(header.BoneMapCount):
                    self.remaps.append(MODBoneRemapv2(header))
            elif v > 220:
                for _ in range(header.BoneMapCount):
                    self.remaps.append(MODBoneRemap(header, cntpad=24))
            elif v < 190 and header.bendian:
                for _ in range(header.BoneMapCount):
                    self.remaps.append(MODBoneRemap(header, cntpad=64))
            else:
                for _ in range(header.BoneMapCount):
                    self.remaps.append(MODBoneRemap(header))


# ----------------------------------------------------------------------------
# Base vertex buffer (positions-only fallback) -- struct MODVertexBuffer
# ----------------------------------------------------------------------------
class MODVertexBufferBase:
    """Positions-only fallback decoder (the base MODVertexBuffer struct).

    Decoders in the mtf/ package subclass-by-convention: they expose the same
    attributes (vertarray, facearray, uvarray, normalarray, boneidarray,
    weightarray) and a constructor (header, count).
    """

    id = "unk[N]"

    def __init__(self, header, count):
        self.vertarray = []
        self.facearray = None
        self.uvarray = []
        self.normalarray = []
        self.boneidarray = []
        self.weightarray = []
        self.count = count
        self.headerref = header
        fl = header.fl
        if header.bendian:
            for _ in range(count):
                v = mathutils.Vector((fl.read_be16s(), fl.read_be16s(), fl.read_be16s()))
                v = mathutils.Vector(tuple(v[i] * header.vtbuffscale[i] for i in range(3)))
                self.vertarray.append(cormat(v - header.bbmin))
                fl.skip(18)
        else:
            for _ in range(count):
                v = mathutils.Vector((fl.read16s(), fl.read16s(), fl.read16s()))
                v = mathutils.Vector(tuple(v[i] * header.vtbuffscale[i] for i in range(3)))
                self.vertarray.append(cormat(v - header.bbmin))
                fl.skip(18)


# ----------------------------------------------------------------------------
# Mesh parts (v1 / v2 / v3)
# ----------------------------------------------------------------------------
class MODMeshPartv1:
    def __init__(self, header):
        self.headerref = header
        self.meshdata = None
        fl = header.fl
        if header.bendian:
            self.id = fl.read_be_short()
            self.material = fl.read_be_short()
            fl.read_byte(signed=False)
            self.LOD = fl.read_byte(signed=False)
            fl.read_short()
            self.blocksize = fl.read_byte(signed=False)
            self.blocktype = fl.read_byte(signed=False)
            fl.skip(2)
            self.vertexcount = fl.read_be_short()
            fl.read_short()
            self.vertexsub = fl.read_be_long()
            self.vertexoffset = fl.read_be_long()
            fl.skip(4)
            self.FaceOffset = fl.read_be_long()
            self.FaceCount = fl.read_be_long()
            self.FaceAdd = fl.read_be_long()
            fl.skip(6)
            self.boneremapid = fl.read_byte(signed=False) + 1
            fl.skip(5)
        else:
            self.id = fl.read_short(signed=False)
            self.material = fl.read_short(signed=False)
            fl.read_byte()
            self.LOD = fl.read_byte(signed=False)
            fl.read_short()
            self.blocksize = fl.read_byte(signed=False)
            self.blocktype = fl.read_byte(signed=False)
            fl.skip(2)
            self.vertexcount = fl.read_short(signed=False)
            fl.read_short()
            self.vertexsub = fl.read_long()
            self.vertexoffset = fl.read_long()
            fl.skip(4)
            self.FaceOffset = fl.read_long(signed=False)
            self.FaceCount = fl.read_long(signed=False)
            self.FaceAdd = fl.read_long(signed=False)
            fl.skip(5)
            self.boneremapid = fl.read_byte(signed=False) + 1
            fl.skip(6)

    def loadmeshdata(self, registry):
        header = self.headerref
        fl = header.fl
        cls = registry.get(self.blocktype)  # v1 keys are raw ints
        if cls is None:
            print("Unknown block hash [%s] for MTF v1 model format." % self.blocktype)
            return
        fl.seek((header.VertexOffset + self.vertexoffset)
                + (self.blocksize * (self.vertexsub + self.FaceAdd)),
                fl.SEEK_SET)
        self.meshdata = cls(header, self.vertexcount)
        fl.seek(header.FacesOffset + self.FaceOffset * 2, fl.SEEK_SET)
        modf = 1 - self.vertexsub
        if header.bendian:
            self.meshdata.facearray = collect_be_strips(fl, self.FaceCount, modf=modf)
        else:
            self.meshdata.facearray = collect_strips(fl, self.FaceCount, modf=modf)


class MODMeshPartv2:
    def __init__(self, header):
        self.headerref = header
        self.meshdata = None
        fl = header.fl
        if header.bendian:
            fl.read_be_short()
            self.vertexcount = fl.read_be_short()
            self.id = fl.read_byte()
            self.material = fl.read_be_short()
            self.LOD = fl.read_byte(signed=False)
            fl.read_short()
            self.blocksize = fl.read_byte(signed=False)
            fl.read_byte()
            self.vertexsub = fl.read_be_long()
            self.vertexoffset = fl.read_be_long()
            self.blocktype = fl.read_be_long()
            self.FaceOffset = fl.read_be_long()
            self.FaceCount = fl.read_be_long()
            fl.read_long()
            if header.Version < 240:  # MH3U WiiU
                fl.read_byte()
                self.boneremapid = fl.read_byte(signed=False) + 1
                fl.skip(10)
            else:
                self.boneremapid = fl.read_byte(signed=False) + 1
                fl.skip(11)
        else:
            fl.read_short(signed=False)
            self.vertexcount = fl.read_short(signed=False)
            self.id = fl.read_byte()
            self.material = fl.read_short(signed=False)
            self.LOD = fl.read_byte(signed=False)
            fl.read_short()
            self.blocksize = fl.read_byte(signed=False)
            fl.read_byte()
            self.vertexsub = fl.read_long(signed=False)
            self.vertexoffset = fl.read_long(signed=False)
            self.blocktype = fl.read_long(signed=False)
            self.FaceOffset = fl.read_long(signed=False)
            self.FaceCount = fl.read_long(signed=False)
            fl.skip(5)
            self.boneremapid = fl.read_byte(signed=False) + 1
            fl.skip(10)
        # RE6PS3 / UMvC3 trailing pad
        if (header.version2 != 0 or header.x64
                or (header.Version < 230 and header.bendian)):
            fl.skip(8)
        self.blocktype = format(self.blocktype, "x")  # bit.intashex

    def loadmeshdata(self, registry):
        header = self.headerref
        fl = header.fl
        cls = registry.get(self.blocktype)  # v2/v3 keys are hex strings
        if cls is None:
            print("Unknown block hash [%s] for MTF v2 model format." % self.blocktype)
            return
        if header.st:
            return
        fl.seek((header.VertexOffset + self.vertexoffset)
                + (self.blocksize * self.vertexsub), fl.SEEK_SET)
        self.meshdata = cls(header, self.vertexcount)
        fl.seek(header.FacesOffset + self.FaceOffset * 2, fl.SEEK_SET)
        modf = 1 - self.vertexsub
        v = header.Version
        # Condition lifted verbatim from the source comment block:
        #   NOT UMvC3 PC ... Dragons Dogma all
        if ((v != 230 or not header.bendian)
                and (v > 220 or v == 212 or (v < 230 and header.bendian))):
            if header.bendian:
                self.meshdata.facearray = collect_be_strips(fl, self.FaceCount, modf=modf)
            else:
                self.meshdata.facearray = collect_strips(fl, self.FaceCount, modf=modf)
        else:
            fc = self.FaceCount // 3
            if header.bendian:
                self.meshdata.facearray = collect_be_tris(fl, fc, modf=modf)
            else:
                self.meshdata.facearray = collect_tris(fl, fc, modf=modf)


class MODMeshPartv3:
    def __init__(self, header):
        self.headerref = header
        self.meshdata = None
        fl = header.fl
        fl.read_short(signed=False)
        self.vertexcount = fl.read_short(signed=False)
        self.id = fl.read_short(signed=False)
        fl.read_short(signed=False)
        self.LOD = fl.read_long(signed=False)
        fl.read_short(signed=False)
        self.blocksize = fl.read_byte(signed=False)
        fl.read_byte()
        self.vertexsub = fl.read_long(signed=False)
        self.vertexoffset = fl.read_long(signed=False)
        self.blocktype = fl.read_long(signed=False)
        self.FaceOffset = fl.read_long(signed=False)
        self.FaceCount = fl.read_long(signed=False)
        self.vertexBase = fl.read_long(signed=False)
        self.boneremapid = fl.read_byte(signed=False) + 1
        fl.skip(39)
        self.blocktype = format(self.blocktype, "x")

    def loadmeshdata(self, registry):
        header = self.headerref
        fl = header.fl
        cls = registry.get(self.blocktype)
        if cls is None:
            print("Unknown block hash [%s] for MTF v3 model format." % self.blocktype)
            return
        if header.st:
            return
        fl.seek((header.VertexOffset + self.vertexoffset)
                + (self.blocksize * (self.vertexsub + self.vertexBase)), fl.SEEK_SET)
        self.meshdata = cls(header, self.vertexcount)
        fl.seek(header.FacesOffset + self.FaceOffset * 2, fl.SEEK_SET)
        fc = self.FaceCount // 3
        modf = 1 - self.vertexsub
        self.meshdata.facearray = collect_tris(fl, fc, modf=modf)


class MODMeshParts:
    def __init__(self, header):
        self.headerref = header
        self.parts = []
        if header.Version == 237:
            cls = MODMeshPartv3
        elif header.Version < 190:
            cls = MODMeshPartv1
        else:
            cls = MODMeshPartv2
        self.parts = [cls(header) for _ in range(header.MeshCount)]
