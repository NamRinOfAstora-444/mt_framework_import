"""
mrl_format.py
Parser for MT Framework .mrl material-library files.

The original MaxScript tool imported neither materials nor textures -- this is
new. Two things are reliably extractable and validated across many real files:

  1. The ordered texture-path table (each path -> a sibling image file).
  2. Each material's albedo texture, matched to the mesh by NAME:
       - the MOD has a material-name table; each mesh selects a name. Each .mrl
         material entry (60-byte array at header field @24, `matCount` @8 long)
         stores the JAMCRC hash of its name at offset +4. Binding is name-hash to
         name-hash, NOT by table position -- the MOD and .mrl order their
         materials independently, so any positional scheme corrupts some models.
       - the albedo value, after the albedo sampler hash (0x1a8dcdc3) in the
         entry's 2nd resource block, selects a texture by its 76-byte unit
         position (value == unit + 1). Verified exact on
         chr000/100/162/209/280/316.
       - outline materials are NAMED with an "ol_" prefix (e.g. "ol__face"); the
         caller renders those as dark, backface-culled inverted hulls.
"""

import struct
import zlib

_TEX_HASH = struct.pack("<I", 0x241f5deb)
_MARKER = b"6wbc"
_MAT_STRIDE = 60          # bytes per material table entry
_TEX_UNIT = 76            # bytes per texture-table "unit"


def mrl_name_hash(name):
    """MT Framework material-name hash: JAMCRC == bitwise-NOT of CRC32.

    The MOD references materials by name; each .mrl material entry stores this
    hash of its name at offset +4. Verified exact on chr000/100/162/209/280/316.
    """
    return (~zlib.crc32(name.encode("ascii", "ignore"))) & 0xFFFFFFFF


def _find_all(data, pat, end=None):
    out = []
    i = 0
    while True:
        j = data.find(pat, i)
        if j < 0 or (end is not None and j >= end):
            break
        out.append(j)
        i = j + 1
    return out


def _texture_offsets(data, mat_off):
    return _find_all(data, _TEX_HASH, mat_off)


def parse_mrl_textures(data):
    """Ordered list of texture base names (no path, no extension)."""
    if data[:4] != b"MRL\x00":
        return []
    try:
        mat_off = struct.unpack_from("<I", data, 24)[0]
    except struct.error:
        mat_off = len(data)
    names = []
    for o in _texture_offsets(data, mat_off):
        seg = data[o + 4:o + 256]
        s = bytearray()
        for b in seg:
            if 32 <= b < 127:
                s.append(b)
            elif len(s) > 4:
                break
            else:
                s = bytearray()
        names.append(s.decode("ascii", "replace").replace("\\", "/").split("/")[-1]
                     if s else "")
    return names


def parse_mrl_albedo_by_hash(data, texture_count):
    """Return {material_name_hash: albedo_tex_index_0based_or_None}.

    Each .mrl material entry (60 bytes at mat_off + slot*60) stores its material
    NAME hash at offset +4 (JAMCRC of the MOD's material name). Meshes select a
    material by NAME via the MOD material-name table, so the binding is by hash,
    not by table position -- on most models the MOD and .mrl order the materials
    independently, which is why any positional scheme corrupts some characters.

    Each material points (slot+48, field [1]) to a resource block of property
    entries; the texture properties carry a VALUE that references a texture by
    its 76-byte unit position in the texture table (value == unit + 1). The
    albedo (base colour) is the property whose value lands on a texture START
    unit; the normal/mask properties point mid-texture, so they are naturally
    excluded. This is read structurally rather than by sampler-hash, because the
    sampler hash varies per shader/version (e.g. chr201 vs chr123 share a MOD
    version but use different sampler hashes). Verified identical to the older
    sampler-based read on chr000/100/162/209/280/316, and resolves the newer
    MOD v230 models (chr201/chr123).
    """
    out = {}
    if data[:4] != b"MRL\x00":
        return out
    try:
        mat_count = struct.unpack_from("<I", data, 8)[0]
        mat_off = struct.unpack_from("<I", data, 24)[0]
    except struct.error:
        return out

    tex_offsets = _texture_offsets(data, mat_off)
    if not tex_offsets:
        return out
    tex_start = tex_offsets[0]
    value_to_tex = {}
    for ti, o in enumerate(tex_offsets):
        unit = int(round((o - tex_start) / float(_TEX_UNIT)))
        value_to_tex[unit + 1] = ti

    # Gather (name_hash, param-block offset) per material slot.
    slots = []
    for slot in range(mat_count):
        off = mat_off + slot * _MAT_STRIDE
        if off + _MAT_STRIDE > len(data):
            break
        try:
            name_hash = struct.unpack_from("<I", data, off + 4)[0]
            o2 = struct.unpack_from("<III", data, off + 48)[1]
        except struct.error:
            continue
        slots.append((name_hash, o2))

    # Bound each param block by the next block's offset (blocks are contiguous),
    # so a scan can't bleed into a neighbouring material.
    o2_sorted = sorted(set(o for _, o in slots if 0 < o < len(data)))

    def block_end(o2):
        for x in o2_sorted:
            if x > o2:
                return x
        return len(data)

    for name_hash, o2 in slots:
        ti = None
        if 0 < o2 < len(data):
            end = min(block_end(o2), len(data))
            # Skip the 12-byte block header; the first property value that lands
            # on a texture start unit is the albedo.
            i = o2 + 0x0C
            while i + 4 <= end:
                val = struct.unpack_from("<I", data, i)[0]
                t = value_to_tex.get(val)
                if t is not None and t < texture_count:
                    ti = t
                    break
                i += 4
        out[name_hash] = ti
    return out


def read_mrl_header(data):
    if len(data) < 16 or data[:4] != b"MRL\x00":
        return None
    ver, matcount, field12 = struct.unpack_from("<III", data, 4)
    return ver, matcount, field12
