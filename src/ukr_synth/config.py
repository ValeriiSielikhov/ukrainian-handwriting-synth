"""Configuration constants for Ukrainian synthetic handwriting generation."""

from pathlib import Path

# Ukrainian Cyrillic alphabet (33 letters, upper + lower)
UKR_UPPER = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
UKR_LOWER = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
# English Latin alphabet (A-Z, a-z)
ENG_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ENG_LOWER = "abcdefghijklmnopqrstuvwxyz"
UKR_DIGITS = "0123456789"
UKR_PUNCTUATION = '.!"%(),-?:; '
UKR_SPECIAL = "@«»—–'№#₴$€£*"

UKR_ALLOWED_SYMBOLS: str = (
    UKR_UPPER + UKR_LOWER + ENG_UPPER + ENG_LOWER + UKR_DIGITS + UKR_PUNCTUATION + UKR_SPECIAL
)
UKR_ALLOWED_SYMBOLS_SET: set[str] = set(UKR_ALLOWED_SYMBOLS)

# Minimum set of characters a font must support to be considered valid
UKR_REQUIRED_CHARS: str = UKR_UPPER + UKR_LOWER


_global_skipped_letters_set: set[str] = set()


def is_text_allowed(text: str) -> bool:
    """Check if text contains only allowed symbols (Ukrainian, English, digits, punctuation)."""
    for char in text:
        if char not in UKR_ALLOWED_SYMBOLS_SET:
            _global_skipped_letters_set.add(char)
            return False
    return True


# Ink/pen colors for realistic handwriting (RGB)
INK_COLORS: list[tuple[int, int, int]] = [
    (15, 15, 15),  # black
    (40, 40, 40),  # dark grey
    (25, 25, 112),  # dark blue (midnight)
    (0, 51, 153),  # blue
    (30, 60, 120),  # navy
    (72, 61, 139),  # violet/purple
    (65, 50, 120),  # blue-violet
]
# Max per-channel random offset applied on top of palette color
INK_COLOR_JITTER: int = 10

# Page/canvas background colors (RGB) for plain backgrounds
PAGE_COLORS: list[tuple[int, int, int]] = [
    (255, 255, 255),  # white
    (250, 250, 245),  # off-white
    (252, 248, 227),  # cream
    (255, 250, 235),  # light yellow
    (240, 240, 240),  # light grey
]

# Default directory for texture backgrounds (lined, grid, kraft)
BACKGROUNDS_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "background"

# Default rendering configuration
DEFAULT_CONFIG: dict = {
    # Font size in px (will be randomized per sample)
    "font_size_min": 40,
    "font_size_max": 85,
    # Page / canvas color in RGB
    "page_color": (255, 255, 255),
    # Text color in RGB
    "text_color": (0, 0, 0),
    # Margin around cropped text in px
    "margin": (15, 15),
    # Skew angle range in radians (-0.5 ≈ -28.65°, 0.5 ≈ 28.65°)
    "skew_min": -0.5,
    "skew_max": 0.5,
}
