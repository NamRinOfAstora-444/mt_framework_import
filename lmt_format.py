"""
lmt_format.py
Port of the LMT (animation) parsing from "MT Framework (LMT, MOD).ms":
  LMTHeader, LMTAniBlockHeader, LMTKeyframeDef, LMTKeyframeBounds and every
  keyframe-buffer decoder (LMTVec3, LMTVec3Frame, LMTQuadraticVector3,
  LMTQuatized8/16Vec3, LMTQuat3Frame, LMTQuatFramev14, LMTQuatized32Quat,
  LMTXW/YW/ZWQuat, LMTQuatized11/9Quat, LMTPolarFrame).

This module only PARSES; turning the decoded frames into Blender keyframes lives
in lmt_import.py. Each decoder produces a `frame` (LMTUniKey) plus `relframe`
(frames until the next key) and `size` (bytes consumed), matching the original.

Value types follow Usage, not the encoding: rotation tracks hold a Quaternion,
position/scale tracks hold a Vector3.
"""

import math
import mathutils

Quaternion = mathutils.Quaternion
Vector = mathutils.Vector


# -- bit helpers (MaxScript bit.shift/and/or semantics) ----------------------
def _shift(x, n):
    """MaxScript bit.shift: left if n>0, logical right if n<0, on 32-bit."""
    x &= 0xFFFFFFFF
    if n >= 0:
        return (x << n) & 0xFFFFFFFF
    return x >> (-n)


def _band(a, b):
    return (a & b) & 0xFFFFFFFF


def _bor(a, b):
    return (a | b) & 0xFFFFFFFF


# ----------------------------------------------------------------------------
class LMTUniKey:
    __slots__ = ("value", "time", "intangenttype", "outtangenttype",
                 "intangent", "outtangent")

    def __init__(self, value=None, time=0, intangenttype="linear",
                 outtangenttype="linear"):
        self.value = value
        self.time = time
        self.intangenttype = intangenttype
        self.outtangenttype = outtangenttype
        self.intangent = Vector((0.0, 0.0, 0.0))
        self.outtangent = Vector((0.0, 0.0, 0.0))


class LMTKeyframeBounds:
    """addin/offset are point4; lerp3 / lerpq map a normalised value into range."""

    def __init__(self, reader, bendian):
        rf = reader.read_be_float if bendian else reader.read_float
        self.addin = [rf(), rf(), rf(), rf()]
        self.offset = [rf(), rf(), rf(), rf()]

    def lerp3(self, frac):
        # frac is a Vector3; (offset + frac*addin) as point3
        return Vector(tuple(self.offset[i] + frac[i] * self.addin[i]
                            for i in range(3)))

    def lerpq(self, q):
        # q is a Quaternion; component-wise offset + q*addin
        return Quaternion((
            self.offset[3] + q.w * self.addin[3],
            self.offset[0] + q.x * self.addin[0],
            self.offset[1] + q.y * self.addin[1],
            self.offset[2] + q.z * self.addin[2],
        ))


# ---------------------------------------------------------------------------
# Keyframe-buffer decoders. Each sets: self.frame (LMTUniKey), self.relframe,
# self.size. Constructors take (reader, bendian, **kw) and read one keyframe.
# ---------------------------------------------------------------------------
class LMTVec3:
    def __init__(self, reader, bendian, **kw):
        self.frame = LMTUniKey()
        self.relframe = 1
        self.size = 12
        self.frame.value = reader.read_be_vector3() if bendian else reader.read_vector3()


class LMTVec3Frame:
    def __init__(self, reader, bendian, cmpinp=False, **kw):
        self.frame = LMTUniKey()
        self.size = 12
        self.relframe = 0
        if bendian:
            self.frame.value = reader.read_be_vector3()
            if cmpinp:
                self.relframe = reader.read_be_long()
        else:
            self.frame.value = reader.read_vector3()
            if cmpinp:
                self.relframe = reader.read_long(signed=False)
        if cmpinp:
            self.size += 4
        else:
            self.relframe = 0


class LMTQuadraticVector3:
    def __init__(self, reader, bendian, **kw):
        self.frame = LMTUniKey(intangenttype="custom", outtangenttype="custom")
        self.nextframeintangent = Vector((0.0, 0.0, 0.0))
        addcount = reader.read_byte(signed=False)
        flags = reader.read_byte(signed=False)
        if bendian:
            self.relframe = reader.read_be_short(signed=False)
            self.frame.value = reader.read_be_vector3()
            rf = reader.read_be_float
        else:
            self.relframe = reader.read_short(signed=False)
            self.frame.value = reader.read_vector3()
            rf = reader.read_float
        ot = self.frame.outtangent
        nf = self.nextframeintangent
        for b in range(1, 9):
            if (flags >> (b - 1)) & 1:
                if b == 1: ot.x = rf()
                elif b == 2: ot.y = rf()
                elif b == 3: ot.z = rf()
                elif b == 4: nf.x = rf()
                elif b == 5: nf.y = rf()
                elif b == 6: nf.z = rf()
        rel = self.relframe if self.relframe else 1
        self.frame.outtangent = ot * (0.19 / rel)
        self.nextframeintangent = nf * (-0.19 / rel)
        self.size = addcount


class LMTQuatized8Vec3:
    def __init__(self, reader, bendian, bounds=None, **kw):
        self.frame = LMTUniKey()
        self.size = 4
        v = Vector((reader.read8(), reader.read8(), reader.read8()))
        self.relframe = reader.read_byte(signed=False)
        self.frame.value = bounds.lerp3(v) if bounds else v


class LMTQuatized16Vec3:
    def __init__(self, reader, bendian, bounds=None, **kw):
        self.frame = LMTUniKey()
        self.size = 8
        if bendian:
            v = Vector((reader.read_be16(), reader.read_be16(), reader.read_be16()))
            self.relframe = reader.read_be_short(signed=False)
        else:
            v = Vector((reader.read16(), reader.read16(), reader.read16()))
            self.relframe = reader.read_short(signed=False)
        self.frame.value = bounds.lerp3(v) if bounds else v


class LMTQuat3Frame:
    def __init__(self, reader, bendian, cmpinp=False, **kw):
        self.frame = LMTUniKey()
        self.size = 12
        self.relframe = 0
        if bendian:
            x, y, z = reader.read_be_float(), reader.read_be_float(), reader.read_be_float()
            if cmpinp:
                self.relframe = reader.read_be_long()
        else:
            x, y, z = reader.read_float(), reader.read_float(), reader.read_float()
            if cmpinp:
                self.relframe = reader.read_long(signed=False)
        w = math.sqrt(max(0.0, 1.0 - x * x - y * y - z * z))
        self.frame.value = Quaternion((w, x, y, z))
        if cmpinp:
            self.size += 4
        else:
            self.relframe = 0


class LMTQuatFramev14:
    BITMASK = 0x3FFF

    def __init__(self, reader, bendian, **kw):
        self.frame = LMTUniKey(value=Quaternion((1, 0, 0, 0)))
        self.size = 8
        bm = self.BITMASK
        if bendian:
            b = [0] * 9
            for i in range(8, 0, -1):
                b[i] = reader.read_byte(signed=False)
            iv1 = b[1]; iv2 = b[5]
            for i in range(2, 5):
                iv1 = _bor(iv1, _shift(b[i], 8 * (i - 1)))
                iv2 = _bor(iv2, _shift(b[i + 4], 8 * (i - 1)))
        else:
            iv1 = reader.read_long(signed=False)
            iv2 = reader.read_long(signed=False)
        q = self.frame.value
        q.w = _band(iv1, bm)
        q.z = _shift(_band(iv1, _shift(bm, 14)), -14)
        q.y = _bor(_shift(_band(iv1, _shift(bm, 28)), -28), _shift(_band(iv2, _shift(bm, -4)), 4))
        q.x = _shift(_band(iv2, _shift(bm, 10)), -10)
        self.relframe = _shift(_band(iv2, _shift(bm, 24)), -24)
        for comp in "wxyz":
            v = getattr(q, comp)
            if v > bm * 0.5:
                v = -(bm - v)
            setattr(q, comp, v * 0.000244156)


class LMTQuatized32Quat:
    BITMASK = 0x7F

    def __init__(self, reader, bendian, bounds=None, **kw):
        self.frame = LMTUniKey(value=Quaternion((1, 0, 0, 0)))
        self.size = 4
        bm = self.BITMASK
        iv = reader.read_be_long() if bendian else reader.read_long(signed=False)
        q = self.frame.value
        q.w = _band(iv, bm)
        q.z = _shift(_band(iv, _shift(bm, 7)), -7)
        q.y = _shift(_band(iv, _shift(bm, 14)), -14)
        q.x = _shift(_band(iv, _shift(bm, 21)), -21)
        self.relframe = _shift(_band(iv, 0xF0000000), -28)
        for comp in "wxyz":
            setattr(q, comp, (getattr(q, comp) - 8) * 0.0089285718)
        if bounds:
            self.frame.value = bounds.lerpq(q)


class _LMTAxisWQuat:
    """Shared XW/YW/ZW quaternion decode; `axis` is 'x','y' or 'z'."""
    BITMASK = 0x3FFF
    axis = "x"

    def __init__(self, reader, bendian, bounds=None, **kw):
        self.frame = LMTUniKey(value=Quaternion((1, 0, 0, 0)))
        self.size = 4
        bm = self.BITMASK
        iv = reader.read_be_long() if bendian else reader.read_long(signed=False)
        q = self.frame.value
        setattr(q, self.axis, _band(iv, bm))
        q.w = _shift(_band(iv, _shift(bm, 14)), -14)
        self.relframe = _shift(_band(iv, 0xF0000000), -28)
        if bounds is None:
            q.w *= 0.000244141
            av = getattr(q, self.axis)
            if av > bm * 0.5:
                av = -(bm - av)
            setattr(q, self.axis, av * 0.000122085)
        else:
            setattr(q, self.axis, getattr(q, self.axis) * 0.000061039)
            q.w *= 0.000061039
            self.frame.value = bounds.lerpq(q)


class LMTXWQuat(_LMTAxisWQuat):
    axis = "x"


class LMTYWQuat(_LMTAxisWQuat):
    axis = "y"


class LMTZWQuat(_LMTAxisWQuat):
    axis = "z"


class LMTQuatized11Quat:
    BITMASK = 0x7FF

    def __init__(self, reader, bendian, bounds=None, **kw):
        self.frame = LMTUniKey(value=Quaternion((1, 0, 0, 0)))
        self.size = 6
        if bendian:
            iv1 = reader.read_be_short(signed=False)
            iv2 = reader.read_be_short(signed=False)
            iv3 = reader.read_be_short(signed=False)
        else:
            iv1 = reader.read_short(signed=False)
            iv2 = reader.read_short(signed=False)
            iv3 = reader.read_short(signed=False)
        q = self.frame.value
        q.x = _band(iv1, 0x7FF)
        q.y = _bor(_band(iv2, 0x3F), _shift(_band(iv1, 0xF800), -5))
        q.z = _bor(_band(iv3, 1), _shift(_band(iv2, 0xFFC0), -5))
        q.w = _shift(_band(iv3, _shift(0x7FF, 1)), -1)
        self.relframe = _shift(_band(iv3, 0xF000), -12)
        for comp in "wxyz":
            setattr(q, comp, getattr(q, comp) * 0.00048852)
        if bounds:
            self.frame.value = bounds.lerpq(q)


class LMTQuatized9Quat:
    def __init__(self, reader, bendian, bounds=None, **kw):
        self.frame = LMTUniKey(value=Quaternion((1, 0, 0, 0)))
        self.size = 5
        iv1 = reader.read_byte(signed=False)
        iv2 = reader.read_byte(signed=False)
        iv3 = reader.read_byte(signed=False)
        iv4 = reader.read_byte(signed=False)
        iv5 = reader.read_byte(signed=False)
        q = self.frame.value
        q.x = _bor(_band(iv2, 1), _shift(iv1, 1))
        q.y = _bor(_band(iv3, 3), _shift(_band(iv2, 0xFE), 1))
        q.z = _bor(_band(iv4, 7), _shift(_band(iv3, 0xF8), 1))
        q.w = _bor(_band(iv5, 0xF), _shift(_band(iv4, 0xE0), 1))
        self.relframe = _shift(_band(iv5, 0xF0), -4)
        for comp in "wxyz":
            setattr(q, comp, getattr(q, comp) * 0.00195695)
        if bounds:
            self.frame.value = bounds.lerpq(q)


class LMTPolarFrame:
    def __init__(self, reader, bendian, **kw):
        self.frame = LMTUniKey(value=Quaternion((1, 0, 0, 0)))
        self.size = 8
        if bendian:
            b = [0] * 9
            for i in range(8, 0, -1):
                b[i] = reader.read_byte(signed=False)
            iv1 = b[1]; iv2 = b[5]
            for i in range(2, 5):
                iv1 = _bor(iv1, _shift(b[i], 8 * (i - 1)))
                iv2 = _bor(iv2, _shift(b[i + 4], 8 * (i - 1)))
        else:
            iv1 = reader.read_long(signed=False)
            iv2 = reader.read_long(signed=False)
        q = self.frame.value
        X = _band(iv1, 0x1FFFF)
        Y = _bor(_shift(_band(iv1, _shift(0x1FFFF, 17)), -17), _shift(_band(iv2, 3), 15))
        W = _shift(_band(iv2, _shift(0x7FFFF, 2)), -2)
        blx = (iv2 & 0x200000) == 0x200000
        bly = (iv2 & 0x400000) == 0x400000
        blz = (iv2 & 0x800000) == 0x800000
        self.relframe = _shift(_band(iv2, 0xFF000000), -24)
        Wf = W * 0.0000019073523
        dequant = [X * 0.000011984317, Y * 0.000011984317,
                   X * 0.000011984317, Y * 0.000011984317]
        var1 = [0.0, 0.0, math.pi / 2, math.pi / 2]
        fvar0 = [dequant[i] - var1[i] for i in range(4)]
        fvar1 = [fvar0[i] * fvar0[i] for i in range(4)]
        fvar2 = [fvar0[i] * fvar1[i] for i in range(4)]
        fvar3 = [fvar1[i] * fvar2[i] for i in range(4)]
        out = []
        for i in range(4):
            out.append((0.008199913 * fvar3[i])
                       + ((-0.16652186 * fvar2[i]) + fvar0[i])
                       + (fvar1[i] * fvar3[i] * -0.00016147594))
        qw = 1.0 - (Wf * Wf)
        magvar = math.sqrt(max(0.0, 1.0 - qw * qw))
        q.w = qw
        q.x = 0.0 - (out[0] * out[3] * magvar)
        q.y = out[1] * magvar
        q.z = out[2] * out[3] * magvar
        if blx: q.x *= -1
        if bly: q.y *= -1
        if blz: q.z *= -1


# ---------------------------------------------------------------------------
class LMTKeyframeDef:
    """One bone's animation track within a block."""

    def __init__(self, reader, bendian, header):
        self.FrameArray = []
        self.BufferRef = []
        self.Bounds = None
        self.BoundsOffset = 0
        fl = reader
        ver = header.header.Version
        x64 = header.header.x64

        if ver > 55 and bendian:
            self.BoneID = fl.read_byte(signed=False)
            self.JointType = fl.read_byte(signed=False)
            self.Usage = fl.read_byte(signed=False)
            self.BufferType = fl.read_byte(signed=False)
        else:
            self.BufferType = fl.read_byte(signed=False)
            self.Usage = fl.read_byte(signed=False)
            self.JointType = fl.read_byte(signed=False)
            self.BoneID = fl.read_byte(signed=False)

        if bendian:
            if ver > 90:
                self.BoneID = fl.read_be_long()
            self.weight = fl.read_be_float()
            self.BufferSize = fl.read_be_long() if ver > 90 else fl.read_be_pointer(x64)
            self.BufferOffset = fl.read_be_pointer(x64)
            self.ReferenceFrame = [fl.read_be_float() for _ in range(4)]
            if ver > 55:
                self.BoundsOffset = fl.read_be_pointer(x64)
        else:
            if ver > 90:
                self.BoneID = fl.read_long(signed=False)
            self.weight = fl.read_float()
            self.BufferSize = fl.read_long(signed=False) if ver > 90 else fl.read_pointer(x64)
            self.BufferOffset = fl.read_pointer(x64)
            self.ReferenceFrame = [fl.read_float() for _ in range(4)]
            if ver > 55:
                self.BoundsOffset = fl.read_pointer(x64)

        savepos = fl.tell()
        if self.BoundsOffset and self.BoundsOffset > 0:
            fl.seek(self.BoundsOffset, fl.SEEK_SET)
            self.Bounds = LMTKeyframeBounds(fl, bendian)

        if self.BufferSize and self.BufferSize > 0:
            fl.seek(self.BufferOffset, fl.SEEK_SET)
            cursize = 0
            curframe = 0
            while cursize < self.BufferSize:
                fr = self._decode_one(fl, bendian, ver)
                if fr is None:
                    break
                f = fr.frame
                f.time = curframe
                if curframe == 0:
                    f.intangenttype = "step"
                if self.BufferType == 5 and ver < 56:
                    self.BufferRef.append(fr)
                    if len(self.BufferRef) > 1:
                        f.intangent = self.BufferRef[-2].nextframeintangent
                self.FrameArray.append(f)
                curframe += fr.relframe
                cursize += fr.size
        else:
            # No buffer: single reference-frame key.
            rf = self.ReferenceFrame
            if self.Usage == 0 or self.Usage == 3:
                k = LMTUniKey(value=Quaternion((rf[3], rf[0], rf[1], rf[2])),
                              intangenttype="step")
            else:
                k = LMTUniKey(value=Vector((rf[0], rf[1], rf[2])), intangenttype="step")
            self.FrameArray.append(k)

        fl.seek(savepos, fl.SEEK_SET)

    def _decode_one(self, fl, bendian, ver):
        bt = self.BufferType
        bounds = self.Bounds
        if bt == 1:
            return LMTVec3(fl, bendian)
        if bt == 2:
            return LMTVec3Frame(fl, bendian, cmpinp=((fl.tell() + 12) < (self.BufferOffset + self.BufferSize)))
        if bt == 3:
            return LMTVec3Frame(fl, bendian, cmpinp=True)
        if bt == 4:
            if ver > 55:
                return LMTQuatized16Vec3(fl, bendian, bounds=bounds)
            return LMTQuat3Frame(fl, bendian, cmpinp=((fl.tell() + 12) < (self.BufferOffset + self.BufferSize)))
        if bt == 5:
            if ver > 55:
                return LMTQuatized8Vec3(fl, bendian, bounds=bounds)
            return LMTQuadraticVector3(fl, bendian)
        if bt == 6:
            if ver > 50:
                return LMTQuatFramev14(fl, bendian)
            return LMTPolarFrame(fl, bendian)
        if bt == 7:
            return LMTQuatized32Quat(fl, bendian, bounds=bounds)
        if bt == 9:
            return LMTVec3Frame(fl, bendian, cmpinp=((fl.tell() + 12) < (self.BufferOffset + self.BufferSize)))
        if bt == 11:
            return LMTXWQuat(fl, bendian, bounds=bounds)
        if bt == 12:
            return LMTYWQuat(fl, bendian, bounds=bounds)
        if bt == 13:
            return LMTZWQuat(fl, bendian, bounds=bounds)
        if bt == 14:
            return LMTQuatized11Quat(fl, bendian, bounds=bounds)
        if bt == 15:
            return LMTQuatized9Quat(fl, bendian, bounds=bounds)
        print("Unknown buffer format: %s" % bt)
        return None


class LMTAniBlockHeader:
    def __init__(self, reader, bendian, header):
        fl = reader
        self.KeyBuffersRef = []
        if bendian:
            self.Offset = fl.read_be_pointer(header.x64)
            self.BoneCount = fl.read_be_long()
            self.FrameCount = fl.read_be_long()
            self.LoopFrame = fl.read_be_long()
        else:
            self.Offset = fl.read_pointer(header.x64)
            self.BoneCount = fl.read_long(signed=False)
            self.FrameCount = fl.read_long(signed=False)
            self.LoopFrame = fl.read_long(signed=True)
        fl.seek(self.Offset, fl.SEEK_SET)
        # `header` for LMTKeyframeDef must expose `.header` (the LMTHeader).
        wrapper = _BlockWrapper(header, header.x64)
        for _ in range(self.BoneCount):
            self.KeyBuffersRef.append(LMTKeyframeDef(fl, bendian, wrapper))


class _BlockWrapper:
    """Adapter so LMTKeyframeDef sees header.header.Version / .x64."""
    def __init__(self, lmt_header, x64):
        self.header = lmt_header
        self.x64 = x64


class LMTHeader:
    def __init__(self, reader, x64=False):
        self.fl = reader
        self.x64 = x64
        self.BEndian = False
        self.Offsettable = []
        fl = reader
        self.ID = fl.read_long(signed=False)
        if self.ID == 0x4C4D5400:           # big-endian "LMT\0"
            self.BEndian = True
            self.Version = fl.read_be_short(signed=False)
            self.NumBlocks = fl.read_be_short(signed=False)
            if self.Version > 90:
                fl.read_be_pointer(x64)
            for _ in range(self.NumBlocks):
                off = fl.read_be_pointer(x64)
                if off > 0:
                    self.Offsettable.append(off)
        elif self.ID == 0x544D4C:           # little-endian "LMT"
            self.Version = fl.read_short(signed=False)
            self.NumBlocks = fl.read_short(signed=False)
            if self.Version > 90:
                fl.read_pointer(x64)
            for _ in range(self.NumBlocks):
                off = fl.read_pointer(x64)
                if off > 0:
                    self.Offsettable.append(off)
        else:
            raise ValueError("Invalid LMT header")

    def get_animation_from_id(self, idx):
        """idx is 0-based into Offsettable."""
        self.fl.seek(self.Offsettable[idx], self.fl.SEEK_SET)
        return LMTAniBlockHeader(self.fl, self.BEndian, self)
