"""Small shared helpers with no intra-app dependencies."""

from datetime import datetime, timezone

# Upper bound for a SQLite INTEGER column (signed 64-bit). Values beyond this
# raise OverflowError at insert time, so IDs are range-checked before use.
MAX_SQLITE_INT = 2 ** 63 - 1


def utcnow():
    """Naive UTC timestamp.

    ``datetime.utcnow()`` is deprecated from Python 3.12 onward, but the
    existing columns store naive datetimes and are compared against naive
    values elsewhere. Stripping the tzinfo keeps that arithmetic working while
    dropping the deprecated call.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_int(value, default=None, minimum=None, maximum=None):
    """Best-effort int coercion that never raises.

    Returns ``default`` when *value* is missing, non-numeric, or a float-ish
    string such as ``"1.5"``. Bounds are clamped rather than rejected so that
    pagination degrades to a valid page instead of erroring.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def parse_item_id(value):
    """Validate a movie/TV id from request JSON.

    Returns ``None`` when the value is absent, non-numeric, or outside the
    range SQLite can store, so callers can answer 400 instead of raising.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0 or parsed > MAX_SQLITE_INT:
        return None
    return parsed
