"""Font discovery and validation for Ukrainian character support."""

from pathlib import Path

from fontTools.ttLib import TTFont

from ukr_synth.config import UKR_REQUIRED_CHARS
from ukr_synth.logger import get_logger

logger = get_logger(__name__)

FONT_EXTENSIONS = (".ttf", ".otf")


def _font_has_glyph(font_tt: TTFont, char: str) -> bool:
    """Check if the font has a glyph for the given character (no fallback)."""
    code_point = ord(char)
    for table in font_tt["cmap"].tables:
        if code_point in table.cmap:
            return True
    return False


def _font_supports_chars(font_path: str, chars: set[str]) -> bool:
    try:
        font_tt = TTFont(font_path)
        try:
            return all(_font_has_glyph(font_tt, ch) for ch in chars)
        finally:
            font_tt.close()
    except Exception:
        return False


def discover_fonts(fonts_dir: str | Path) -> list[Path]:
    """List all .ttf/.otf font files in the given directory."""
    fonts_dir = Path(fonts_dir)
    if not fonts_dir.is_dir():
        return []
    return sorted(p for p in fonts_dir.iterdir() if p.suffix.lower() in FONT_EXTENSIONS)


def validate_font(
    font_path: str | Path, required_chars: str = UKR_REQUIRED_CHARS
) -> bool:
    """Check whether a font file supports all required Ukrainian characters.

    Uses fontTools to read the font cmap directly. This avoids PIL's fallback
    behavior (squares, emojis, skipped glyphs) when a character is missing.
    """
    try:
        font_tt = TTFont(str(font_path))
    except Exception:
        return False

    try:
        for char in required_chars:
            if not _font_has_glyph(font_tt, char):
                return False
        return True
    finally:
        font_tt.close()


def get_valid_fonts(
    fonts_dir: str | Path,
    required_chars: str = UKR_REQUIRED_CHARS,
) -> list[Path]:
    """Return fonts from ``fonts_dir`` that support all required characters."""
    valid: list[Path] = []
    fonts = discover_fonts(fonts_dir)
    logger.info(f"Found {len(fonts)} font(s) in '{fonts_dir}' directory")
    for font_path in fonts:
        if validate_font(font_path, required_chars):
            valid.append(font_path)
        else:
            logger.info(f"Invalid font: {font_path}")
    return valid


def get_fonts_for_text(
    valid_fonts: list[Path],
    text: str,
) -> list[Path]:
    """Filter already-validated fonts to those that can render *text*."""
    required = {ch for ch in text if ch in UKR_REQUIRED_CHARS}
    if not required:
        return list(valid_fonts)
    return [p for p in valid_fonts if _font_supports_chars(str(p), required)]
