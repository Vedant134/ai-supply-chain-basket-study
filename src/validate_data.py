"""Small, dependency-free quality check for long-form adjusted price rows."""

from collections import Counter
from math import isfinite


def validate_rows(rows, expected_sessions, symbols):
    """Report missing, duplicated, and invalid date-symbol observations.

    Expected sessions are supplied by the caller so exchange holidays and weekends
    are never inferred from calendar days.
    """
    counts = Counter()
    invalid = set()
    for row in rows:
        key = f"{row['date']} {row['symbol']}"
        counts[key] += 1
        try:
            price = float(row['adjusted_close'])
            if not isfinite(price) or price <= 0:
                invalid.add(key)
        except (TypeError, ValueError):
            invalid.add(key)

    expected = {f"{date} {symbol}" for date in expected_sessions for symbol in symbols}
    return {
        "missing": sorted(expected - counts.keys()),
        "duplicates": sorted(key for key, count in counts.items() if count > 1),
        "invalid_prices": sorted(invalid),
    }
