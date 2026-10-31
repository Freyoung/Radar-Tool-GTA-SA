"""
GTA SA Radar Tool - Streamlit Edition (v3)
==========================================
Rewrite of the tkinter tool (v6.0 by Tensho Records), improved.

Tabs
----
1. Merge Tiles  : radar00..radar143 (PNG *or TXD*) -> one big map.png (12x12 grid)
2. Split Map    : big map -> 144 tiles as PNG, TXD or both (ZIP), with progress bar
3. PNG -> TXD   : radar*.png -> RenderWare TXD (GTA SA / D3D9), optional zero-padded names
4. TXD -> PNG   : read TXD files (A8R8G8B8, X8R8G8B8, 24/16-bit, PAL8, DXT1/3/5) -> PNG
5. Color Studio : up to 4 stacked color rules + hue/saturation/brightness/contrast,
                  before/after slider, full-res on demand
6. One-Click    : TXD/PNG in -> recolor -> TXD (DXT1 or 32-bit) out, optional merged map
7. Appearance   : black & white themes (Noir / Paper), fonts, corner radius, accent
8. Help

TXD output can be lossless A8R8G8B8 or DXT1 (about 8x smaller, like the game's own files).

Run with:  streamlit run app.py
Needs   :  streamlit, pillow, numpy   (see requirements.txt)
"""

from __future__ import annotations

import inspect
import io
import os
import re
import struct
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ================================
#   Constants
# ================================

APP_TITLE = "GTA SA Radar Tool"
APP_ICON = "🛰️"

DEFAULT_TILE_SIZE = 256
GRID_SIZE = 12
TOTAL_TILES = GRID_SIZE * GRID_SIZE
MAX_TILE_SIZE = 1024                  # per-tile cap
MAX_CANVAS_PIXELS = 40_000_000        # ~160 MB RGBA (safe for streamlit.app; raise if you run locally)
MAX_PREVIEW_DIM = 1600                # preview downscale limit (downloads stay full-res)
LIVE_PREVIEW_DIM = 800                # color-replace live preview size

# --- RenderWare / TXD constants (same bytes as the original tool) ---
RW_VERSION = 0x1803FFFF
SECTION_STRUCT = 0x01
SECTION_EXTENSION = 0x03
SECTION_TEX_NATIVE = 0x15
SECTION_TEX_DICT = 0x16

PLATFORM_D3D9 = 9
RASTER_FORMAT = 0x0500       # rwRASTERFORMAT8888
FILTER_FLAGS = 0x1106
DEPTH = 32
MIPMAP_COUNT = 1
TEX_TYPE = 4
D3D9_FLAGS = 0x01            # bit0 = hasAlpha

D3DFMT_R8G8B8 = 20
D3DFMT_A8R8G8B8 = 21
D3DFMT_X8R8G8B8 = 22
D3DFMT_R5G6B5 = 23
D3DFMT_A1R5G5B5 = 25
FOURCC_DXT1 = 0x31545844
FOURCC_DXT3 = 0x33545844
FOURCC_DXT5 = 0x35545844
FORMAT_LABELS = {
    D3DFMT_R8G8B8: "R8G8B8", D3DFMT_A8R8G8B8: "A8R8G8B8", D3DFMT_X8R8G8B8: "X8R8G8B8",
    D3DFMT_R5G6B5: "R5G6B5", D3DFMT_A1R5G5B5: "A1R5G5B5",
    FOURCC_DXT1: "DXT1", FOURCC_DXT3: "DXT3", FOURCC_DXT5: "DXT5",
}

# radar00.png / radar7.txd ... (matched against the file's base name)
TILE_NAME_RE = re.compile(r"^radar(\d+)\.(png|txd)$", re.IGNORECASE)

FORMATS = ["A8R8G8B8 (lossless)", "DXT1 (8× smaller, opaque)"]

MODE_SHIFT = "Shift (keeps shading)"
MODE_BLEND = "Blend to color (flat)"


# ================================
#   Small helpers
# ================================

def _to_rgba(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGBA" else image.convert("RGBA")


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _zip_bytes(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zf.writestr(name, data)
    return buf.getvalue()


def _thumb(image: Image.Image, max_dim: int = MAX_PREVIEW_DIM) -> Image.Image:
    if max(image.size) <= max_dim:
        return image
    thumb = image.copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return thumb


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)


def _rgb_label(hex_str: str) -> str:
    return "rgb(%d, %d, %d)" % _hex_to_rgb(hex_str)


def _human_size(n: int) -> str:
    return f"{n / 1024:,.0f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:,.1f} MB"


def _stretch_kw(func) -> dict:
    """Full-width kwarg for a Streamlit widget, across Streamlit versions."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params and "stretch" in str(params["width"].annotation):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


# ================================
#   Tile helpers
# ================================

def tile_index(filename: str) -> Optional[int]:
    """'radar013.png' -> 13. None if not a radar tile or out of 0..143."""
    match = TILE_NAME_RE.match(os.path.basename(filename))
    if not match:
        return None
    index = int(match.group(1))
    return index if 0 <= index < TOTAL_TILES else None


def tile_position(index: int, tile_w: int, tile_h: int, flip_vertical: bool) -> tuple[int, int]:
    """Top-left pixel of tile `index` on the merged map (original logic)."""
    col = index % GRID_SIZE
    row = index // GRID_SIZE
    if flip_vertical:
        row = GRID_SIZE - 1 - row
    return col * tile_w, row * tile_h


def canvas_problem(tile_w: int, tile_h: int) -> Optional[str]:
    """Error text if a 12x12 canvas of this tile size is too big, else None."""
    w, h = tile_w * GRID_SIZE, tile_h * GRID_SIZE
    if w * h > MAX_CANVAS_PIXELS:
        return (f"A {w}×{h} map (~{w * h * 4 / 1024 ** 2:,.0f} MB in memory) is over the "
                f"safety limit. Use a smaller tile size.")
    return None


def draw_grid(image: Image.Image, flip_vertical: bool) -> Image.Image:
    """Overlay the 12x12 grid and tile numbers on a preview image."""
    img = image.convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    cw, ch = w / GRID_SIZE, h / GRID_SIZE
    for k in range(GRID_SIZE + 1):
        draw.line([(k * cw, 0), (k * cw, h)], fill=(255, 255, 0, 170), width=1)
        draw.line([(0, k * ch), (w, k * ch)], fill=(255, 255, 0, 170), width=1)
    if cw >= 28:
        for i in range(TOTAL_TILES):
            row = i // GRID_SIZE
            if flip_vertical:
                row = GRID_SIZE - 1 - row
            x, y = (i % GRID_SIZE) * cw + 3, row * ch + 2
            draw.text((x + 1, y + 1), str(i), fill=(0, 0, 0, 255))
            draw.text((x, y), str(i), fill=(255, 255, 0, 255))
    return img


def tile_status_grid(placed: set[int], flip_vertical: bool) -> str:
    """12x12 emoji grid laid out like the merged map (green = present)."""
    rows = []
    for r in range(GRID_SIZE):
        src = GRID_SIZE - 1 - r if flip_vertical else r
        rows.append("".join("🟩" if src * GRID_SIZE + c in placed else "⬜"
                            for c in range(GRID_SIZE)))
    return "\n".join(rows)


# ================================
#   TXD writer
# ================================

def _rw_section(section_type: int, data: bytes) -> bytes:
    return struct.pack("<III", section_type, len(data), RW_VERSION) + data


def safe_texture_name(name: str) -> str:
    """ASCII-only, 31 chars max, no extension - what GTA SA texture names allow."""
    name = re.sub(r"\.(png|txd)$", "", os.path.basename(name), flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    return name[:31] or "texture"


def encode_dxt1(rgb: np.ndarray) -> bytes:
    """Vectorized DXT1 (BC1) encoder: principal-axis endpoints + best-of-4 indices.

    rgb: HxWx3 uint8. Dimensions are padded to multiples of 4 by edge replication.
    """
    h, w = rgb.shape[:2]
    pad_h, pad_w = (-h) % 4, (-w) % 4
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    bh, bw = rgb.shape[0] // 4, rgb.shape[1] // 4
    px = rgb.reshape(bh, 4, bw, 4, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 3).astype(np.float32)

    mean = px.mean(axis=1, keepdims=True)
    cen = px - mean
    cov = np.einsum("bpi,bpj->bij", cen, cen)
    axis = np.linalg.eigh(cov)[1][:, :, -1]                     # principal color axis
    proj = np.einsum("bpi,bi->bp", cen, axis)
    lo = np.clip(mean[:, 0] + axis * proj.min(axis=1)[:, None], 0, 255)
    hi = np.clip(mean[:, 0] + axis * proj.max(axis=1)[:, None], 0, 255)

    def q565(c):
        r = np.rint(c[:, 0] * 31 / 255).astype(np.uint32)
        g = np.rint(c[:, 1] * 63 / 255).astype(np.uint32)
        b = np.rint(c[:, 2] * 31 / 255).astype(np.uint32)
        return (r << 11) | (g << 5) | b

    def expand(v):
        r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
        return np.stack([r * 255 / 31, g * 255 / 63, b * 255 / 31], axis=1).astype(np.float32)

    c0, c1 = q565(hi), q565(lo)
    c0, c1 = np.maximum(c0, c1), np.minimum(c0, c1)             # c0 > c1 -> 4-color mode
    p0, p1 = expand(c0), expand(c1)
    pal = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], axis=1)
    dist = ((px[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)
    idx = dist.argmin(-1).astype(np.uint32)
    idx[c0 == c1] = 0
    bits = np.bitwise_or.reduce(idx << (2 * np.arange(16, dtype=np.uint32)), axis=1)

    out = np.empty(len(c0), dtype=[("c0", "<u2"), ("c1", "<u2"), ("bits", "<u4")])
    out["c0"], out["c1"], out["bits"] = c0, c1, bits
    return out.tobytes()


def make_txd(texture_name: str, image: Image.Image, dxt1: bool = False) -> bytes:
    """Minimal RenderWare TXD (GTA SA, D3D9, no mipmaps).

    dxt1=False : lossless A8R8G8B8 (default, original behaviour)
    dxt1=True  : DXT1 compressed, opaque, about 8x smaller
    """
    img = image.convert("RGBA")
    w, h = img.size
    if dxt1:
        pixel_data = encode_dxt1(np.asarray(img.convert("RGB")))
        raster, d3d_fmt, depth, flags = 0x0200, FOURCC_DXT1, 16, 0x08
    else:
        r, g, b, a = img.split()
        pixel_data = Image.merge("RGBA", (b, g, r, a)).tobytes()   # RGBA -> BGRA
        raster, d3d_fmt, depth, flags = RASTER_FORMAT, D3DFMT_A8R8G8B8, DEPTH, D3D9_FLAGS

    name_bytes = texture_name.encode("ascii")[:31].ljust(32, b"\x00")
    native_struct = (
        struct.pack("<II", PLATFORM_D3D9, FILTER_FLAGS)
        + name_bytes + b"\x00" * 32
        + struct.pack("<IIHHBBBBI", raster, d3d_fmt, w, h,
                      depth, MIPMAP_COUNT, TEX_TYPE, flags, len(pixel_data))
        + pixel_data
    )
    tex_native = _rw_section(
        SECTION_TEX_NATIVE,
        _rw_section(SECTION_STRUCT, native_struct) + _rw_section(SECTION_EXTENSION, b""),
    )
    dict_body = (
        _rw_section(SECTION_STRUCT, struct.pack("<HH", 1, 2))
        + tex_native
        + _rw_section(SECTION_EXTENSION, b"")
    )
    return _rw_section(SECTION_TEX_DICT, dict_body)


def txd_texture_name(filename: str, zero_pad: bool) -> str:
    base = re.sub(r"\.png$", "", os.path.basename(filename), flags=re.IGNORECASE)
    if zero_pad:
        m = re.fullmatch(r"radar(\d+)", base, re.IGNORECASE)
        if m:
            base = f"radar{int(m.group(1)):02d}"
    return safe_texture_name(base)


# ================================
#   TXD reader
# ================================

@dataclass
class TxdTexture:
    name: str
    width: int
    height: int
    fmt: str
    mip_levels: int
    image: Image.Image


def _dds_wrap(fourcc: int, w: int, h: int, payload: bytes) -> bytes:
    """Wrap raw DXT data in a DDS header so Pillow can decode it."""
    header = struct.pack("<4s7I", b"DDS ", 124, 0x81007, h, w, len(payload), 0, 0)
    header += b"\x00" * 44                                       # reserved[11]
    header += struct.pack("<II I 5I", 32, 4, fourcc, 0, 0, 0, 0, 0)  # pixel format
    header += struct.pack("<5I", 0x1000, 0, 0, 0, 0)             # caps
    return header + payload


def _decode_mip0(d3d_format: int, w: int, h: int, payload: bytes,
                 palette: Optional[bytes]) -> Image.Image:
    if palette is not None:
        idx = np.frombuffer(payload, np.uint8, w * h).reshape(h, w)
        pal = np.frombuffer(palette, np.uint8).reshape(256, 4)[:, [2, 1, 0, 3]]
        return Image.fromarray(np.ascontiguousarray(pal[idx]))
    if d3d_format in (D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8):
        arr = np.frombuffer(payload, np.uint8, w * h * 4).reshape(h, w, 4)[..., [2, 1, 0, 3]].copy()
        if d3d_format == D3DFMT_X8R8G8B8:
            arr[..., 3] = 255
        return Image.fromarray(arr)
    if d3d_format == D3DFMT_R8G8B8:
        arr = np.frombuffer(payload, np.uint8, w * h * 3).reshape(h, w, 3)[..., ::-1]
        return Image.fromarray(np.ascontiguousarray(arr)).convert("RGBA")
    if d3d_format in (D3DFMT_R5G6B5, D3DFMT_A1R5G5B5):
        v = np.frombuffer(payload, "<u2", w * h).reshape(h, w).astype(np.uint32)
        if d3d_format == D3DFMT_R5G6B5:
            r, g, b, a = (v >> 11) & 31, (v >> 5) & 63, v & 31, np.full_like(v, 255)
            r, g, b = r * 255 // 31, g * 255 // 63, b * 255 // 31
        else:
            a = (v >> 15) * 255
            r, g, b = ((v >> 10) & 31) * 255 // 31, ((v >> 5) & 31) * 255 // 31, (v & 31) * 255 // 31
        return Image.fromarray(np.dstack([r, g, b, a]).astype(np.uint8))
    if d3d_format in (FOURCC_DXT1, FOURCC_DXT3, FOURCC_DXT5):
        block = 8 if d3d_format == FOURCC_DXT1 else 16
        need = ((w + 3) // 4) * ((h + 3) // 4) * block
        if len(payload) < need:
            raise ValueError("compressed data is truncated")
        dds = _dds_wrap(d3d_format, w, h, payload[:need])
        return Image.open(io.BytesIO(dds)).convert("RGBA")
    raise ValueError(f"unsupported pixel format {d3d_format:#x}")


def _parse_native(data: bytes, start: int) -> TxdTexture:
    sec_type, _, _ = struct.unpack_from("<III", data, start)
    if sec_type != SECTION_STRUCT:
        raise ValueError("texture has no struct section")
    p = start + 12
    if p + 92 > len(data):
        raise ValueError("file is truncated")
    platform = struct.unpack_from("<I", data, p)[0]
    if platform != PLATFORM_D3D9:
        raise ValueError(f"platform {platform} is not D3D9 (only GTA SA PC textures are supported)")
    name = data[p + 8:p + 40].split(b"\x00")[0].decode("ascii", "replace")
    raster, d3d_format, w, h, _depth, mips, _rtype, _flags = struct.unpack_from("<IIHHBBBB", data, p + 72)
    q = p + 88
    palette = None
    if raster & 0x2000:                       # PAL8
        palette = data[q:q + 1024]
        q += 1024
    elif raster & 0x4000:                     # PAL4
        raise ValueError("4-bit palette textures are not supported")
    if q + 4 > len(data):
        raise ValueError("file is truncated")
    size = struct.unpack_from("<I", data, q)[0]
    payload = data[q + 4:q + 4 + size]
    if len(payload) < size:
        raise ValueError("pixel data is truncated")
    image = _decode_mip0(d3d_format, w, h, payload, palette)
    fmt = "PAL8" if palette is not None else FORMAT_LABELS.get(d3d_format, hex(d3d_format))
    return TxdTexture(name, w, h, fmt, max(1, mips), image)


def parse_txd(data: bytes) -> list[TxdTexture]:
    """Decode every texture (first mip level) in a GTA SA TXD."""
    if len(data) < 28 or struct.unpack_from("<I", data, 0)[0] != SECTION_TEX_DICT:
        raise ValueError("not a TXD file")
    end = min(len(data), 12 + struct.unpack_from("<I", data, 4)[0])
    pos = 12
    st_type, st_size, _ = struct.unpack_from("<III", data, pos)
    if st_type != SECTION_STRUCT:
        raise ValueError("damaged TXD header")
    count = struct.unpack_from("<H", data, pos + 12)[0]
    pos += 12 + st_size
    textures: list[TxdTexture] = []
    while pos + 12 <= end and len(textures) < count:
        sec_type, size, _ = struct.unpack_from("<III", data, pos)
        if sec_type == SECTION_TEX_NATIVE:
            textures.append(_parse_native(data, pos + 12))
        pos += 12 + size
    if not textures:
        raise ValueError("no textures found")
    return textures


# ================================
#   Feature 1 - Merge
# ================================

@dataclass
class MergeReport:
    placed: set = field(default_factory=set)
    ignored: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)
    resized: int = 0


def _load_tile(upload) -> Image.Image:
    raw = upload.getvalue()
    if upload.name.lower().endswith(".txd"):
        return _to_rgba(parse_txd(raw)[0].image)
    return _to_rgba(Image.open(io.BytesIO(raw)))


def merge_uploads(uploads: list, tile_w: int, tile_h: int, flip_vertical: bool,
                  background: tuple = (0, 0, 0, 0)) -> tuple[Image.Image, MergeReport]:
    """Stitch uploaded tiles one at a time (keeps memory low)."""
    canvas = Image.new("RGBA", (tile_w * GRID_SIZE, tile_h * GRID_SIZE), background)
    report = MergeReport()
    for upload in uploads:
        index = tile_index(upload.name)
        if index is None:
            report.ignored.append(upload.name)
            continue
        try:
            img = _load_tile(upload)
        except Exception as exc:
            report.ignored.append(f"{upload.name} ({exc})")
            continue
        if index in report.placed:
            report.duplicates.append(upload.name)
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            report.resized += 1
        canvas.paste(img, tile_position(index, tile_w, tile_h, flip_vertical))
        report.placed.add(index)
    return canvas, report


# ================================
#   Feature 2 - Split
# ================================

def split_to_zip(map_image: Image.Image, tile_w: int, tile_h: int, flip_vertical: bool,
                 make_png: bool, make_txd_files: bool,
                 progress: Optional[Callable[[int, int], None]] = None,
                 dxt1: bool = False) -> bytes:
    """Cut the map into 144 tiles and zip them as PNG and/or TXD."""
    map_image = _to_rgba(map_image)
    target = (tile_w * GRID_SIZE, tile_h * GRID_SIZE)
    if map_image.size != target:
        map_image = map_image.resize(target, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(TOTAL_TILES):
            x, y = tile_position(i, tile_w, tile_h, flip_vertical)
            tile = map_image.crop((x, y, x + tile_w, y + tile_h))
            name = f"radar{i:02d}"
            if make_png:
                zf.writestr(f"{name}.png", _png_bytes(tile))
            if make_txd_files:
                zf.writestr(f"{name}.txd", make_txd(name, tile, dxt1))
            if progress:
                progress(i + 1, TOTAL_TILES)
    return buf.getvalue()


# ================================
#   Feature 3 / 4 - PNG <-> TXD
# ================================

def convert_pngs_to_txd(uploads: list, zero_pad: bool,
                        progress: Optional[Callable[[int, int], None]] = None,
                        dxt1: bool = False):
    """Returns (zip entries, table rows, error messages)."""
    results: list[tuple[str, bytes]] = []
    rows: list[dict] = []
    errors: list[str] = []
    used: set[str] = set()
    for n, upload in enumerate(uploads, 1):
        try:
            name = txd_texture_name(upload.name, zero_pad)
            if name.lower() in used:
                raise ValueError(f"duplicate texture name '{name}' - skipped")
            img = Image.open(io.BytesIO(upload.getvalue()))
            data = make_txd(name, img, dxt1)
            used.add(name.lower())
            results.append((name + ".txd", data))
            rows.append({"Source": upload.name, "Texture": name,
                         "Size": f"{img.size[0]}×{img.size[1]}", "TXD": _human_size(len(data))})
        except Exception as exc:
            errors.append(f"{upload.name}: {exc}")
        if progress:
            progress(n, len(uploads))
    return results, rows, errors


def extract_txds(uploads: list):
    """Returns (zip entries, table rows, previews, errors)."""
    files: list[tuple[str, bytes]] = []
    rows: list[dict] = []
    previews: list[tuple[str, Image.Image]] = []
    errors: list[str] = []
    used: set[str] = set()
    for upload in uploads:
        stem = os.path.splitext(os.path.basename(upload.name))[0]
        try:
            for tex in parse_txd(upload.getvalue()):
                fname = f"{tex.name}.png"
                if fname.lower() in used:
                    fname = f"{stem}_{tex.name}.png"
                used.add(fname.lower())
                files.append((fname, _png_bytes(tex.image)))
                rows.append({"TXD": upload.name, "Texture": tex.name,
                             "Size": f"{tex.width}×{tex.height}", "Format": tex.fmt,
                             "Mips": tex.mip_levels})
                if len(previews) < 8:
                    previews.append((tex.name, _thumb(tex.image, 256)))
        except Exception as exc:
            errors.append(f"{upload.name}: {exc}")
    return files, rows, previews, errors


# ================================
#   Feature 5 - Color Replace
# ================================

def recolor_image(image: Image.Image, target_hex: str, replace_hex: str,
                  tolerance: int, mode: str) -> Image.Image:
    """Soft color replace, vectorized with NumPy and processed in row chunks.

    mask = clamp(1 - distance(pixel, target) / tolerance, 0, 1)
    SHIFT : out = pixel + (replace - target) * mask     (original behaviour)
    BLEND : out = pixel + (replace - pixel)  * mask     (pulls toward a flat color)
    """
    arr = np.asarray(_to_rgba(image), dtype=np.uint8)
    out = arr.copy()
    t = np.array(_hex_to_rgb(target_hex), np.float32)
    r = np.array(_hex_to_rgb(replace_hex), np.float32)
    tol = float(max(1e-6, tolerance))
    for y0 in range(0, arr.shape[0], 256):
        rgb = arr[y0:y0 + 256, :, :3].astype(np.float32)
        dist = np.sqrt(((rgb - t) ** 2).sum(axis=-1))
        mask = np.clip(1.0 - dist / tol, 0.0, 1.0)[..., None]
        res = rgb + ((r - t) if mode == MODE_SHIFT else (r - rgb)) * mask
        out[y0:y0 + 256, :, :3] = np.clip(res + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def adjust_image(image: Image.Image, hue: int = 0, sat: int = 0, bri: int = 0, con: int = 0) -> Image.Image:
    """Global hue shift (deg), saturation / brightness / contrast (%), alpha preserved."""
    if not (hue or sat or bri or con):
        return image
    alpha = image.getchannel("A")
    rgb = image.convert("RGB")
    clamp = lambda v: min(255, max(0, int(v + 0.5)))
    if hue or sat or bri:
        h, sch, v = rgb.convert("HSV").split()
        if hue:
            shift = round(hue * 255 / 360)
            h = h.point([(i + shift) % 256 for i in range(256)])
        if sat:
            sch = sch.point([clamp(i * (1 + sat / 100)) for i in range(256)])
        if bri:
            v = v.point([clamp(i * (1 + bri / 100)) for i in range(256)])
        rgb = Image.merge("HSV", (h, sch, v)).convert("RGB")
    if con:
        k = 1 + con / 100
        rgb = rgb.point([clamp((i - 128) * k + 128) for i in range(256)] * 3)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def process_image(image: Image.Image, rules: tuple, adj: tuple) -> Image.Image:
    """Apply stacked color rules (target, replace, tolerance, mode), then global adjustments."""
    img = _to_rgba(image)
    for target, replace, tol, mode in rules:
        if target.lower() != replace.lower():
            img = recolor_image(img, target, replace, tol, mode)
    return adjust_image(img, *adj)


@st.cache_data(show_spinner=False, max_entries=4)
def process_png(image_bytes: bytes, rules: tuple, adj: tuple, max_dim: int = 0) -> bytes:
    image = _to_rgba(Image.open(io.BytesIO(image_bytes)))
    if max_dim:
        image = _thumb(image, max_dim)
    return _png_bytes(process_image(image, rules, adj))


def compare_image(original: Image.Image, result: Image.Image, pos: int) -> Image.Image:
    """Before/after composite: original on the left of the divider, result on the right."""
    w, h = result.size
    original = original.convert("RGBA")
    if original.size != (w, h):
        original = original.resize((w, h))
    out = result.convert("RGBA")
    x = int(w * pos / 100)
    if x > 0:
        out.paste(original.crop((0, 0, x, h)), (0, 0))
    if 0 < x < w:
        ImageDraw.Draw(out).line([(x, 0), (x, h)], fill=(255, 255, 255, 255), width=3)
    return out


@st.cache_data(show_spinner=False, max_entries=2)
def preview_png(image_bytes: bytes, max_dim: int) -> bytes:
    return _png_bytes(_thumb(_to_rgba(Image.open(io.BytesIO(image_bytes))), max_dim))


@st.cache_data(show_spinner=False, max_entries=2)
def dominant_colors(image_bytes: bytes, n: int = 8) -> list[str]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((256, 256))
    q = img.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    counts = sorted(q.getcolors() or [], reverse=True)
    return ["#%02x%02x%02x" % tuple(pal[i * 3:i * 3 + 3]) for _, i in counts]


def _pick(prefix: str, i: int) -> None:
    value = st.session_state.get(f"{prefix}_pick{i}")
    if value and value != "—":
        st.session_state[f"{prefix}_t{i}"] = value


def _reset_adjust() -> None:
    for k in ("adj_hue", "adj_sat", "adj_bri", "adj_con"):
        st.session_state[k] = 0


def rules_ui(prefix: str, colors: list[str]) -> tuple:
    """Up to 4 stacked color rules; returns a hashable tuple of (target, replace, tol, mode)."""
    n = st.radio("Color rules", [1, 2, 3, 4], horizontal=True, key=f"{prefix}_n")
    rules = []
    for i in range(n):
        st.session_state.setdefault(f"{prefix}_t{i}", "#888888" if i == 0 else "#ffffff")
        st.session_state.setdefault(f"{prefix}_r{i}", "#ff0000" if i == 0 else "#ffffff")
        with st.expander(f"Rule {i + 1}", expanded=(i == 0)):
            c1, c2 = st.columns(2)
            c1.color_picker("🎯 Target", key=f"{prefix}_t{i}")
            c2.color_picker("🎨 Replace with", key=f"{prefix}_r{i}")
            if colors:
                st.selectbox("Pick target from the image", ["—"] + colors, key=f"{prefix}_pick{i}",
                             on_change=_pick, args=(prefix, i))
            tol = st.slider("Tolerance", 1, 150, 40, key=f"{prefix}_tol{i}",
                            help="1 = near-exact matches only. 150 = very wide, soft blend.")
            mode = st.radio("Mode", [MODE_SHIFT, MODE_BLEND], horizontal=True, key=f"{prefix}_mode{i}",
                            help="Shift keeps shading and detail. Blend pulls pixels to one flat color.")
        rules.append((st.session_state[f"{prefix}_t{i}"], st.session_state[f"{prefix}_r{i}"], tol, mode))
    return tuple(rules)


# ================================
#   Theme (black & white, customizable)
# ================================

THEMES = {
    "Noir":  dict(bg="#0a0a0a", panel="#151515", fg="#f4f4f2", muted="#8d8d8d", border="#2c2c2c"),
    "Paper": dict(bg="#fafaf8", panel="#ffffff", fg="#0a0a0a", muted="#6b6b6b", border="#d9d9d4"),
}
FONTS = {
    "Inter - modern": ("Inter:wght@400;500;600;700", "'Inter', system-ui, sans-serif"),
    "Playfair Display - classic serif": ("Playfair+Display:wght@400;600;700", "'Playfair Display', Georgia, serif"),
    "JetBrains Mono - technical": ("JetBrains+Mono:wght@400;600", "'JetBrains Mono', ui-monospace, monospace"),
    "Helvetica - system": ("", "'Helvetica Neue', Helvetica, Arial, sans-serif"),
}

CSS_TEMPLATE = r"""
%IMPORTS%
:root{--bg:%BG%;--panel:%PANEL%;--fg:%FG%;--muted:%MUTED%;--border:%BORDER%;--accent:%ACCENT%;--on-accent:%ONACC%;--r:%R%px}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:var(--bg)!important}
html,body,.stApp{font-family:%BODY%}
.stApp p,.stApp label,.stApp li,.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stSliderThumbValue"],.stApp [data-testid="stTickBarMin"],.stApp [data-testid="stTickBarMax"]{color:var(--fg)!important;font-family:%BODY%}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:var(--fg)!important;font-family:%HEAD%!important;font-weight:600;letter-spacing:-.01em}
.stApp h1{%CAPS%}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] *{color:var(--muted)!important}
.stApp hr{border-color:var(--border)!important}
.stApp a{color:var(--fg)!important;text-decoration:underline}
.stApp code{background:var(--panel)!important;color:var(--fg)!important;border:1px solid var(--border);border-radius:6px}
.stApp pre{background:var(--panel)!important;border:1px solid var(--border);border-radius:var(--r)}
/* tabs = the menu */
.stApp [data-baseweb="tab-list"]{gap:.25rem;border-bottom:1px solid var(--border)}
.stApp button[data-baseweb="tab"]{background:transparent!important}
.stApp button[data-baseweb="tab"] p{font-size:.78rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)!important}
.stApp button[data-baseweb="tab"][aria-selected="true"] p{color:var(--fg)!important;font-weight:600}
.stApp [data-baseweb="tab-highlight"]{background:var(--accent)!important;height:2px!important}
.stApp [data-baseweb="tab-border"]{background:transparent!important}
/* buttons */
.stApp button[data-testid^="stBaseButton"],.stApp button[kind]{border-radius:var(--r)!important;border:1px solid var(--fg)!important;background:transparent!important;transition:all .15s ease}
.stApp button[data-testid^="stBaseButton"] *,.stApp button[kind] *{color:var(--fg)!important}
.stApp button[data-testid^="stBaseButton"]:hover,.stApp button[kind]:hover{background:var(--fg)!important}
.stApp button[data-testid^="stBaseButton"]:hover *,.stApp button[kind]:hover *{color:var(--bg)!important}
.stApp button[data-testid="stBaseButton-primary"],.stApp button[kind="primary"]{background:var(--accent)!important;border-color:var(--accent)!important}
.stApp button[data-testid="stBaseButton-primary"] *,.stApp button[kind="primary"] *{color:var(--on-accent)!important}
.stApp button[data-testid="stBaseButton-primary"]:hover,.stApp button[kind="primary"]:hover{background:var(--accent)!important;opacity:.85}
.stApp button[data-testid="stBaseButton-primary"]:hover *,.stApp button[kind="primary"]:hover *{color:var(--on-accent)!important}
.stApp button:disabled{opacity:.35!important}
/* inputs */
.stApp [data-baseweb="input"],.stApp [data-baseweb="textarea"],.stApp [data-baseweb="select"]>div{background:var(--panel)!important;border-color:var(--border)!important;border-radius:var(--r)!important}
.stApp input,.stApp textarea,.stApp [data-baseweb="select"] *{color:var(--fg)!important}
[data-baseweb="popover"] *,[data-baseweb="menu"]{background:var(--panel)!important;color:var(--fg)!important}
.stApp [data-baseweb="slider"] [role="slider"]{background:var(--accent)!important;border-color:var(--accent)!important}
.stApp [data-baseweb="progress-bar"] > div > div{background:var(--accent)!important}
.stApp [data-testid="stFileUploaderDropzone"]{background:var(--panel)!important;border:1px dashed var(--muted)!important;border-radius:var(--r)!important}
.stApp [data-testid="stFileUploaderDropzone"] *{color:var(--fg)!important}
.stApp [data-testid="stExpander"] details{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important}
.stApp [data-testid="stExpander"] summary *{color:var(--fg)!important}
.stApp [data-testid="stAlert"]{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important}
.stApp [data-testid="stAlert"] *{color:var(--fg)!important}
.stApp [data-testid="stImage"] img{border-radius:var(--r);border:1px solid var(--border)}
footer,[data-testid="stToolbar"]{visibility:hidden}
.block-container{padding-top:2rem;max-width:1200px}
"""


def build_css() -> str:
    ss = st.session_state
    theme = THEMES.get(ss.get("ui_preset", "Noir"), THEMES["Noir"])
    body_q, body_stack = FONTS.get(ss.get("ui_font", "Inter - modern"), FONTS["Inter - modern"])
    head_q, head_stack = FONTS.get(ss.get("ui_head_font", "Playfair Display - classic serif"),
                                   FONTS["Playfair Display - classic serif"])
    accent = ss.get("ui_accent", "#ffffff") if ss.get("ui_custom_accent") else theme["fg"]
    r, g, b = _hex_to_rgb(accent)
    on_accent = "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#ffffff"
    imports = "".join(
        f"@import url('https://fonts.googleapis.com/css2?family={q}&display=swap');"
        for q in dict.fromkeys([body_q, head_q]) if q)
    values = {
        "IMPORTS": imports, "BG": theme["bg"], "PANEL": theme["panel"], "FG": theme["fg"],
        "MUTED": theme["muted"], "BORDER": theme["border"], "ACCENT": accent, "ONACC": on_accent,
        "R": str(ss.get("ui_radius", 10)), "BODY": body_stack, "HEAD": head_stack,
        "CAPS": "text-transform:uppercase;letter-spacing:.12em;font-size:1.9rem" if ss.get("ui_caps", True) else "",
    }
    css = CSS_TEMPLATE
    for key, val in values.items():
        css = css.replace(f"%{key}%", val)
    return css


# ================================
#   UI
# ================================

HELP_TEXT = """
### 🛰️ About GTA SA radar files
The radar is built from **144 square tiles** in a **12×12 grid**, named
`radar00` … `radar143`, left-to-right, top-to-bottom. In the game each tile is
its own `radarNN.txd` holding one texture `radarNN`.

### Fastest way: ⚡ One-Click
Set your colors in **Color Studio**, then drop the game's `radarNN.txd` files in **One-Click**.
You get back recolored TXD files (and optionally a merged map) - no tab hopping.

### TXD formats
- **A8R8G8B8**: lossless, about 256 KB per 256×256 tile.
- **DXT1**: about 8× smaller (like the game's own files), opaque only, slight quality loss.
  Test one tile in the game before converting all 144.

### Typical manual workflow
1. **TXD → PNG**: extract the game's original `radarNN.txd` files.
2. **Merge Tiles**: drop the PNGs *or the TXDs directly* to get one big map.
3. Edit the map in any image editor (or use **Color Replace**).
4. **Split Map** with output **TXD** (or PNG + TXD) and put the files back in the game.

### Notes
- **Flip Vertical** reverses the row order. Use the same setting when splitting
  that you used when merging.
- Tile size defaults to 256 (game standard). Tiles of another size are resized.
- Texture names are limited to 31 ASCII characters. *Zero-pad* turns `radar5` into `radar07`-style names.
- TXD reading supports A8R8G8B8 / X8R8G8B8 / R8G8B8 / 565 / 1555 / PAL8 / DXT1 / DXT3 / DXT5
  (first mip level). Writing produces A8R8G8B8 or DXT1.
- **Privacy:** files are processed in memory by the machine running this app. If you run it
  locally nothing leaves your PC; if it is hosted, uploads go to that server.

*Streamlit edition - based on RadarTool v6.0 by Tensho Records.*
"""


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    BTN, DL, IMG = _stretch_kw(st.button), _stretch_kw(st.download_button), _stretch_kw(st.image)
    for key, default in (("ui_preset", "Noir"), ("ui_font", "Inter - modern"),
                         ("ui_head_font", "Playfair Display - classic serif"),
                         ("ui_radius", 10), ("ui_caps", True), ("ui_custom_accent", False),
                         ("ui_accent", "#ffffff")):
        st.session_state.setdefault(key, default)
    st.markdown(f"<style>{build_css()}</style>", unsafe_allow_html=True)

    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("One-click · Merge · Split · Convert · Extract · Color studio - GTA San Andreas radar maps")

    tab_one, tab_merge, tab_split, tab_txd, tab_extract, tab_color, tab_style, tab_help = st.tabs(
        ["⚡ One-Click", "🧩 Merge", "✂️ Split", "📦 PNG → TXD", "📤 TXD → PNG",
         "🎨 Color Studio", "🎛️ Appearance", "❓ Help"])

    # ---------- Merge ----------
    with tab_merge:
        st.markdown("Upload radar tiles (`radar00` … `radar143`, **PNG or TXD**) and stitch them "
                    "into one **map.png**.")
        uploads = st.file_uploader("Radar tiles", type=["png", "txd"],
                                   accept_multiple_files=True, key="merge_uploads")
        c1, c2, c3 = st.columns(3)
        tile_w = c1.number_input("Tile width (px)", 1, MAX_TILE_SIZE, DEFAULT_TILE_SIZE, key="merge_w")
        tile_h = c2.number_input("Tile height (px)", 1, MAX_TILE_SIZE, DEFAULT_TILE_SIZE, key="merge_h")
        fill = c3.selectbox("Missing tiles", ["Transparent", "Black", "White"], key="merge_fill")
        o1, o2 = st.columns(2)
        flip = o1.checkbox("🔃 Flip Vertical", key="merge_flip",
                           help="Mirror the row order - use if the map is upside-down in game.")
        show_grid = o2.checkbox("🔢 Show tile grid on preview", key="merge_grid")

        problem = canvas_problem(int(tile_w), int(tile_h))
        if problem:
            st.error(problem)

        if st.button("🧩 Merge Tiles", type="primary", disabled=not uploads or bool(problem), **BTN):
            try:
                bg = {"Transparent": (0, 0, 0, 0), "Black": (0, 0, 0, 255),
                      "White": (255, 255, 255, 255)}[fill]
                with st.spinner("Merging tiles…"):
                    canvas, rep = merge_uploads(uploads, int(tile_w), int(tile_h), flip, bg)
                if not rep.placed:
                    st.error("No usable radar tiles found. Names must look like `radar00.png` … `radar143.png`.")
                else:
                    warnings = []
                    if rep.ignored:
                        warnings.append("Ignored: " + ", ".join(rep.ignored))
                    if rep.duplicates:
                        warnings.append("Duplicate tile numbers (last one wins): " + ", ".join(rep.duplicates))
                    if rep.resized:
                        warnings.append(f"{rep.resized} tile(s) were resized to {tile_w}×{tile_h}.")
                    st.session_state.merge_result = {
                        "png": _png_bytes(canvas), "thumb": _png_bytes(_thumb(canvas)),
                        "size": canvas.size, "placed": rep.placed, "flip": flip, "warnings": warnings}
            except Exception as exc:
                st.error(f"Merge failed: {exc}")
        elif not uploads:
            st.info("Upload at least one radar tile to get started.")

        res = st.session_state.get("merge_result")
        if res:
            st.success(f"map.png built ({res['size'][0]}×{res['size'][1]}) from "
                       f"{len(res['placed'])}/{TOTAL_TILES} tiles.")
            for w in res["warnings"]:
                st.warning(w)
            preview = Image.open(io.BytesIO(res["thumb"]))
            st.image(draw_grid(preview, res["flip"]) if show_grid else preview, **IMG)
            if len(res["placed"]) < TOTAL_TILES:
                missing = [str(i) for i in range(TOTAL_TILES) if i not in res["placed"]]
                with st.expander(f"{len(missing)} missing tiles"):
                    st.code(tile_status_grid(res["placed"], res["flip"]), language=None)
                    st.caption("Missing: " + ", ".join(missing))
            st.download_button("⬇️ Download map.png", res["png"], "map.png", "image/png",
                               key="merge_download", **DL)

    # ---------- Split ----------
    with tab_split:
        st.markdown("Slice one large map into exactly **144 tiles** (`radar00` … `radar143`), as a ZIP.")
        map_upload = st.file_uploader("Map image", type=["png", "jpg", "jpeg", "bmp", "webp"],
                                      key="split_upload")
        c1, c2, c3 = st.columns(3)
        s_w = c1.number_input("Tile width (px)", 1, MAX_TILE_SIZE, DEFAULT_TILE_SIZE, key="split_w")
        s_h = c2.number_input("Tile height (px)", 1, MAX_TILE_SIZE, DEFAULT_TILE_SIZE, key="split_h")
        out_fmt = c3.radio("Output", ["PNG", "TXD", "PNG + TXD"], horizontal=True, key="split_fmt")
        split_fmt = st.radio("TXD format", FORMATS, horizontal=True, key="split_txdfmt",
                             help="DXT1 is far smaller and drops transparency. Only used for TXD output.")
        s_flip = st.checkbox("🔃 Flip Vertical", key="split_flip",
                             help="Exact inverse of 'Flip Vertical' when merging.")

        map_img = None
        s_problem = canvas_problem(int(s_w), int(s_h))
        if s_problem:
            st.error(s_problem)
        elif map_upload:
            try:
                map_img = Image.open(io.BytesIO(map_upload.getvalue()))
                mw, mh = map_img.size
                st.caption(f"Loaded **{map_upload.name}** - {mw}×{mh} px.")
                if mw * mh > MAX_CANVAS_PIXELS:
                    st.error("This image is too large to process safely.")
                    map_img = None
                else:
                    expected = (int(s_w) * GRID_SIZE, int(s_h) * GRID_SIZE)
                    if map_img.size != expected:
                        st.info(f"The map will be resized to {expected[0]}×{expected[1]} (12 × tile size).")
                    if abs(mw / mh - 1) > 0.01:
                        st.warning("The map is not square - it will be stretched to a square.")
            except Exception as exc:
                st.error(f"Could not read image: {exc}")
                map_img = None

        if st.button("✂️ Split Map", type="primary", disabled=map_img is None, **BTN):
            bar = st.progress(0.0, text="Starting…")
            try:
                data = split_to_zip(
                    map_img, int(s_w), int(s_h), s_flip,
                    make_png=out_fmt in ("PNG", "PNG + TXD"),
                    make_txd_files=out_fmt in ("TXD", "PNG + TXD"),
                    dxt1=split_fmt.startswith("DXT1"),
                    progress=lambda i, n: bar.progress(i / n, text=f"Tile {i}/{n}"))
                st.session_state.split_result = {"zip": data, "info": f"{s_w}×{s_h} · {out_fmt}"}
            except Exception as exc:
                st.error(f"Split failed: {exc}")
            bar.empty()

        sres = st.session_state.get("split_result")
        if sres:
            st.success(f"All {TOTAL_TILES} tiles exported ({sres['info']}, {_human_size(len(sres['zip']))}).")
            st.download_button("⬇️ Download radar_tiles.zip", sres["zip"], "radar_tiles.zip",
                               "application/zip", key="split_download", **DL)

    # ---------- PNG -> TXD ----------
    with tab_txd:
        st.markdown("Convert PNG files into **RenderWare TXD** textures for GTA San Andreas "
                    "(D3D9, A8R8G8B8).")
        txd_uploads = st.file_uploader("PNG files", type=["png"], accept_multiple_files=True,
                                       key="txd_uploads")
        zero_pad = st.checkbox("Zero-pad tile numbers (radar5 → radar05)", value=True, key="txd_pad")
        txd_fmt = st.radio("TXD format", FORMATS, horizontal=True, key="txd_fmt",
                           help="DXT1 is far smaller and drops transparency.")
        if st.button("📦 Convert to TXD", type="primary", disabled=not txd_uploads, **BTN):
            bar = st.progress(0.0, text="Converting…")
            results, rows, errors = convert_pngs_to_txd(
                txd_uploads, zero_pad, lambda i, n: bar.progress(i / n, text=f"{i}/{n}"),
                dxt1=txd_fmt.startswith("DXT1"))
            bar.empty()
            st.session_state.txd_result = {"zip": _zip_bytes(results) if results else None,
                                           "rows": rows, "errors": errors}
        tres = st.session_state.get("txd_result")
        if tres:
            if tres["zip"]:
                st.success(f"Converted {len(tres['rows'])} PNG file(s) to TXD.")
                st.dataframe(tres["rows"], hide_index=True)
                st.download_button("⬇️ Download txd_files.zip", tres["zip"], "txd_files.zip",
                                   "application/zip", key="txd_download", **DL)
            if tres["errors"]:
                st.error("Some files could not be converted:\n\n" + "\n".join(f"- {e}" for e in tres["errors"]))
        elif not txd_uploads:
            st.info("Upload one or more PNG files to convert.")

    # ---------- TXD -> PNG ----------
    with tab_extract:
        st.markdown("Extract the textures from **TXD** files (for example the game's original "
                    "`radarNN.txd`) into PNG.")
        ex_uploads = st.file_uploader("TXD files", type=["txd"], accept_multiple_files=True,
                                      key="extract_uploads")
        if st.button("📤 Extract to PNG", type="primary", disabled=not ex_uploads, **BTN):
            with st.spinner("Reading TXD files…"):
                files, rows, previews, errors = extract_txds(ex_uploads)
            st.session_state.extract_result = {"zip": _zip_bytes(files) if files else None,
                                               "rows": rows, "previews": previews, "errors": errors}
        eres = st.session_state.get("extract_result")
        if eres:
            if eres["zip"]:
                st.success(f"Extracted {len(eres['rows'])} texture(s).")
                st.dataframe(eres["rows"], hide_index=True)
                cols = st.columns(4)
                for i, (name, img) in enumerate(eres["previews"]):
                    with cols[i % 4]:
                        st.image(img, caption=name, **IMG)
                st.download_button("⬇️ Download png_files.zip", eres["zip"], "png_files.zip",
                                   "application/zip", key="extract_download", **DL)
            if eres["errors"]:
                st.error("Some files could not be read:\n\n" + "\n".join(f"- {e}" for e in eres["errors"]))
        elif not ex_uploads:
            st.info("Upload one or more TXD files.")

    # ---------- Color Studio ----------
    with tab_color:
        st.markdown("Stack up to **4 color rules**, fine-tune hue / saturation / brightness / contrast, "
                    "and slide to compare before and after.")
        color_upload = st.file_uploader("Map image", type=["png"], key="color_upload")

        image_bytes, colors = None, []
        if color_upload:
            image_bytes = color_upload.getvalue()
            try:
                size = Image.open(io.BytesIO(image_bytes)).size
                if size[0] * size[1] > MAX_CANVAS_PIXELS:
                    st.error("This image is too large to process safely.")
                    image_bytes = None
                else:
                    st.caption(f"Loaded **{color_upload.name}** - {size[0]}×{size[1]} px.")
            except Exception as exc:
                st.error(f"Could not read image: {exc}")
                image_bytes = None
        if image_bytes:
            colors = dominant_colors(image_bytes)
            st.caption("Dominant colors: " + "  ".join(colors))
            for col, hx in zip(st.columns(len(colors)), colors):
                col.markdown(f'<div style="background:{hx};height:22px;border-radius:6px;'
                             f'border:1px solid #8886"></div>', unsafe_allow_html=True)

        rules = rules_ui("cr", colors)
        with st.expander("🎚️ Global adjustments"):
            a1, a2 = st.columns(2)
            hue = a1.slider("Hue shift (°)", -180, 180, 0, key="adj_hue")
            sat = a2.slider("Saturation (%)", -100, 100, 0, key="adj_sat")
            bri = a1.slider("Brightness (%)", -100, 100, 0, key="adj_bri")
            con = a2.slider("Contrast (%)", -100, 100, 0, key="adj_con")
            st.button("Reset adjustments", on_click=_reset_adjust)
        adj = (hue, sat, bri, con)
        st.session_state.color_settings = (rules, adj)      # used by the One-Click tab

        if image_bytes:
            before = Image.open(io.BytesIO(preview_png(image_bytes, LIVE_PREVIEW_DIM)))
            after = Image.open(io.BytesIO(process_png(image_bytes, rules, adj, LIVE_PREVIEW_DIM)))
            pos = st.slider("◀ Original  ·  Recolored ▶", 0, 100, 50, key="cmp_pos")
            st.image(compare_image(before, after, pos), **IMG)

            key = (color_upload.name, len(image_bytes), rules, adj)
            if st.button("🖼️ Render full resolution", **BTN):
                with st.spinner("Rendering full-resolution image…"):
                    st.session_state.color_full = {"key": key,
                                                   "png": process_png(image_bytes, rules, adj, 0)}
            full = st.session_state.get("color_full")
            if full and full["key"] == key:
                st.download_button("⬇️ Download map_recolored.png", full["png"],
                                   "map_recolored.png", "image/png", key="color_download", **DL)
            elif full:
                st.caption("Settings changed - render full resolution again to download.")
        else:
            st.info("Upload a map image to preview. Your rules here also drive the One-Click tab.")

    # ---------- One-Click ----------
    with tab_one:
        st.markdown("Drop the game's **radar TXD** (or PNG) files. The recipe from the "
                    "**Color Studio** tab is applied to every tile and you get ready-to-use TXD files back.")
        rules_now, adj_now = st.session_state.get("color_settings", ((), (0, 0, 0, 0)))
        n_rules = sum(1 for r in rules_now if r[0].lower() != r[1].lower())
        recipe = f"{n_rules} color rule(s)"
        recipe += (f" · hue {adj_now[0]}° · sat {adj_now[1]}% · bright {adj_now[2]}% · contrast {adj_now[3]}%"
                   if any(adj_now) else " · no global adjustments")
        st.caption("Current recipe: " + recipe)
        if not n_rules and not any(adj_now):
            st.info("No recipe set yet - tiles will just be re-encoded (handy to convert to DXT1).")

        one_files = st.file_uploader("Radar TXD / PNG files", type=["txd", "png"],
                                     accept_multiple_files=True, key="one_uploads")
        o1, o2 = st.columns(2)
        one_fmt = o1.radio("Output TXD format", FORMATS, key="one_fmt")
        want_map = o2.checkbox("Also build merged map.png", key="one_map")
        one_flip = o2.checkbox("🔃 Flip Vertical (map only)", key="one_flip")

        if st.button("⚡ Run", type="primary", disabled=not one_files, **BTN):
            bar = st.progress(0.0, text="Working…")
            files, rows, errors, used = [], [], [], set()
            first, canvas, tile_size, placed = None, None, None, 0
            for n, up in enumerate(one_files, 1):
                try:
                    src = _load_tile(up)
                    out = process_image(src, rules_now, adj_now)
                    name = txd_texture_name(os.path.splitext(os.path.basename(up.name))[0] + ".png", True)
                    if name.lower() in used:
                        raise ValueError(f"duplicate name '{name}' - skipped")
                    data = make_txd(name, out, one_fmt.startswith("DXT1"))
                    used.add(name.lower())
                    files.append((name + ".txd", data))
                    rows.append({"File": up.name, "Texture": name, "Size": f"{out.size[0]}×{out.size[1]}",
                                 "TXD": _human_size(len(data))})
                    if first is None:
                        first = (_thumb(src, 256), _thumb(out, 256))
                    idx = tile_index(up.name)
                    if want_map and idx is not None:
                        if canvas is None:
                            tile_size = out.size
                            if canvas_problem(*tile_size):
                                want_map = False
                                errors.append("Merged map skipped: " + canvas_problem(*tile_size))
                            else:
                                canvas = Image.new("RGBA", (tile_size[0] * GRID_SIZE, tile_size[1] * GRID_SIZE))
                        if canvas is not None:
                            tile = out if out.size == tile_size else out.resize(tile_size, Image.Resampling.LANCZOS)
                            canvas.paste(tile, tile_position(idx, *tile_size, one_flip))
                            placed += 1
                except Exception as exc:
                    errors.append(f"{up.name}: {exc}")
                bar.progress(n / len(one_files), text=f"{n}/{len(one_files)}")
            bar.empty()
            st.session_state.one_result = {
                "zip": _zip_bytes(files) if files else None, "rows": rows, "errors": errors,
                "first": first, "map": _png_bytes(canvas) if canvas is not None else None, "placed": placed}

        ores = st.session_state.get("one_result")
        if ores:
            if ores["zip"]:
                st.success(f"{len(ores['rows'])} TXD file(s) ready ({_human_size(len(ores['zip']))} zipped).")
                if ores["first"]:
                    lc, rc = st.columns(2)
                    lc.image(ores["first"][0], caption="Before (first tile)", **IMG)
                    rc.image(ores["first"][1], caption="After", **IMG)
                st.dataframe(ores["rows"], hide_index=True)
                st.download_button("⬇️ Download radar_txd.zip", ores["zip"], "radar_txd.zip",
                                   "application/zip", key="one_dl_zip", **DL)
                if ores["map"]:
                    st.download_button(f"⬇️ Download map.png ({ores['placed']}/{TOTAL_TILES} tiles)",
                                       ores["map"], "map.png", "image/png", key="one_dl_map", **DL)
            if ores["errors"]:
                st.error("Problems:\n\n" + "\n".join(f"- {e}" for e in ores["errors"]))

    # ---------- Appearance ----------
    with tab_style:
        st.markdown("Make it yours. Changes apply instantly.")
        st.radio("Theme", list(THEMES), horizontal=True, key="ui_preset",
                 help="Noir = white on black. Paper = black on white.")
        f1, f2 = st.columns(2)
        f1.selectbox("Body font", list(FONTS), key="ui_font")
        f2.selectbox("Heading font", list(FONTS), key="ui_head_font")
        st.slider("Corner radius", 0, 24, key="ui_radius")
        st.checkbox("Uppercase, spaced title", key="ui_caps")
        st.checkbox("Custom accent color (buttons & highlights)", key="ui_custom_accent")
        if st.session_state.ui_custom_accent:
            st.color_picker("Accent", key="ui_accent")
        st.caption("Fonts load from Google Fonts, so they need an internet connection.")

    # ---------- Help ----------
    with tab_help:
        st.markdown(HELP_TEXT)

    st.divider()
    st.caption("🛰️ GTA SA Radar Tool v3 · 12×12 grid · 144 tiles · Streamlit + Pillow + NumPy")


if st.runtime.exists():
    main()
