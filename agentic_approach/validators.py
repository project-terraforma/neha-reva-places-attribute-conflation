"""
Phone and website validation. Uses reference validators when available,
with fallbacks for phonenumbers (libphonenumber) and requests-based website checks.

Used by evidence utilities (extract_fields) to validate phone numbers extracted
from web pages and to verify that website URLs are reachable before fetching.
"""

import re
from typing import Tuple

# Try project's phonenumber_validator first (may have Overture-specific logic)
try:
    from phonenumber_validator import validate_phone_number as _validate_phone
    from phonenumber_validator import try_with_region as _try_with_region
    _HAS_PHONE_VALIDATOR = True
except ImportError:
    _HAS_PHONE_VALIDATOR = False

# Try project's website_validator (may have custom reachability checks)
try:
    from website_validator import verify_website as _verify_website
    _HAS_WEBSITE_VALIDATOR = True
except ImportError:
    _HAS_WEBSITE_VALIDATOR = False

# Fallback: Google libphonenumber for parsing/validating international numbers
try:
    import phonenumbers
    _HAS_PHONENUMBERS = True
except ImportError:
    _HAS_PHONENUMBERS = False

# Regex for basic format check when no phone library is available
_PHONE_PATTERN = re.compile(r"^\+?[\d\s\-\.\(\)]{10,20}$")


def validate_phone_number(phone: str | None, region: str = "US") -> Tuple[bool, str | None]:
    """
    Validate phone number. Returns (is_valid, normalized_e164_or_none).
    Region hint (e.g. "US", "CA") helps parse numbers without country code.
    """
    if not phone or not isinstance(phone, str):
        return False, None
    s = phone.strip()
    if not s:
        return False, None

    # Prefer project validator if available
    if _HAS_PHONE_VALIDATOR:
        ok, extra = _validate_phone(s)
        return (bool(ok), extra if isinstance(extra, str) else None)

    # Use libphonenumber for proper parsing and E.164 normalization
    if _HAS_PHONENUMBERS:
        try:
            parsed = phonenumbers.parse(s, region)
            if phonenumbers.is_valid_number(parsed):
                return True, phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except phonenumbers.NumberParseException:
            pass
        return False, None

    # Last resort: basic format check (10+ digits, reasonable chars)
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 10 and _PHONE_PATTERN.match(s):
        return True, s
    return False, None


def try_with_region(phone: str, region: str) -> Tuple[bool, str | None]:
    """
    Try validating phone with region hint. Used when parsing numbers from
    addresses (e.g. US/CA/GB) to add country code if missing.
    Returns (ok, e164).
    """
    if _HAS_PHONE_VALIDATOR:
        return _try_with_region(phone, region)
    ok, e164 = validate_phone_number(phone, region)
    return ok, e164


def verify_website(url: str | None) -> Tuple[bool, str | None]:
    """
    Verify website is reachable. Returns (is_valid, final_url_or_none).
    Follows redirects; returns the final URL after redirect chain.
    """
    if not url or not isinstance(url, str):
        return False, None
    s = url.strip()
    if not s or not s.startswith(("http://", "https://")):
        return False, None

    if _HAS_WEBSITE_VALIDATOR:
        return _verify_website(s)

    # Fallback: simple HTTP GET; 2xx/3xx = valid
    try:
        import requests
        r = requests.get(s, timeout=10, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        return 200 <= r.status_code < 400, r.url
    except Exception:
        return False, None
