"""Configuration constants for Ukrainian synthetic handwriting generation."""

# Ukrainian Cyrillic alphabet (33 letters, upper + lower)
UKR_UPPER = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
UKR_LOWER = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UKR_DIGITS = "0123456789"
UKR_PUNCTUATION = '.!"%(),-?:; '
UKR_SPECIAL = "«»—–'№"

UKR_ALLOWED_SYMBOLS: str = (
    UKR_UPPER + UKR_LOWER + UKR_DIGITS + UKR_PUNCTUATION + UKR_SPECIAL
)
UKR_ALLOWED_SYMBOLS_SET: set[str] = set(UKR_ALLOWED_SYMBOLS)

# Minimum set of characters a font must support to be considered valid
UKR_REQUIRED_CHARS: str = UKR_UPPER + UKR_LOWER

# Default rendering configuration
DEFAULT_CONFIG: dict = {
    # Font size in px (will be randomized per sample)
    "font_size_min": 30,
    "font_size_max": 100,
    # Page / canvas color in RGB
    "page_color": (255, 255, 255),
    # Text color in RGB
    "text_color": (0, 0, 0),
    # Margin around cropped text in px
    "margin": (10, 10),
    # Skew angle range in radians (-1.0 ≈ -57°, 1.0 ≈ 57°)
    "skew_min": -1.0,  #
    "skew_max": 1.0,
}
