"""Fetch real data from Yahoo Finance and write manifest."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class FetchError(RuntimeError):
    """Network or fetch error."""
    pass


BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_one(
    symbol: str,
    interval: str,
    period1: int,
    period2: int,
    data_dir: str | Path,
) -> dict:
    """Fetch one symbol from Yahoo Finance and write to data_dir/raw/<symbol>_<interval>.json.

    Args:
        symbol: stock symbol (e.g., "SPY", "^GSPC", "^VIX")
        interval: interval string ("1d", "1h", "5m")
        period1: start time (epoch seconds)
        period2: end time (epoch seconds)
        data_dir: root data directory

    Returns:
        Manifest entry dict with keys: symbol, interval, url, period1, period2, sha256, bytes, rows, first, last

    Raises:
        FetchError: if network error or HTTP error occurs
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Build request URL
    url_encoded_symbol = urllib.parse.quote(symbol, safe="")
    request_url = f"{BASE_URL}{url_encoded_symbol}?period1={period1}&period2={period2}&interval={interval}"

    # Create request with User-Agent
    req = urllib.request.Request(request_url)
    req.add_header("User-Agent", USER_AGENT)

    try:
        response = urllib.request.urlopen(req)
        response_bytes = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        raise FetchError(f"Failed to fetch {symbol}: {e}") from e
    except Exception as e:
        raise FetchError(f"Unexpected error fetching {symbol}: {e}") from e

    # Parse response to get row count and first/last timestamps
    try:
        data = json.loads(response_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise FetchError(f"Failed to parse response for {symbol}: {e}") from e

    # Count retained rows (load_chart_json logic)
    try:
        from rough_hedge.realdata import load_chart_json

        # Write bytes first
        output_file = raw_dir / f"{symbol}_{interval}.json"
        output_file.write_bytes(response_bytes)

        # Load to get row count and timestamps
        df = load_chart_json(output_file)
        n_rows = len(df)
        # Format timestamps as ISO 8601 with 'Z' suffix for UTC
        if len(df) > 0:
            first_ts = df.index[0].strftime("%Y-%m-%dT%H:%M:%SZ")
            last_ts = df.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            first_ts = ""
            last_ts = ""
    except Exception as e:
        raise FetchError(f"Failed to process {symbol}: {e}") from e

    # Compute manifest entry
    sha256 = hashlib.sha256(response_bytes).hexdigest()
    entry = {
        "symbol": symbol,
        "interval": interval,
        "url": f"{BASE_URL}{url_encoded_symbol}" if url_encoded_symbol != symbol else f"{BASE_URL}{symbol}",
        "period1": int(period1),
        "period2": int(period2),
        "sha256": sha256,
        "bytes": len(response_bytes),
        "rows": n_rows,
        "first": first_ts,
        "last": last_ts,
    }

    return entry


def write_manifest(entries: list[dict], data_dir: str | Path) -> Path:
    """Write manifest.json file.

    Args:
        entries: list of manifest entry dicts
        data_dir: root data directory

    Returns:
        Path to written MANIFEST.json
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "MANIFEST.json"

    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)

    return manifest_path


if __name__ == "__main__":
    import sys
    import time
    from datetime import datetime, timedelta

    dry_run = "--dry-run" in sys.argv

    def to_epoch(year, month, day):
        return int(datetime(year, month, day).timestamp())

    # Now rounded down to the hour
    now = datetime.utcnow()
    period2_intraday = int((now.replace(minute=0, second=0, microsecond=0)).timestamp())

    # Fetch specs: 8 daily + 2 intraday (1h and 5m)
    specs = [
        # Daily: 2010-01-01 to 2026-01-01
        ("^GSPC", "1d", to_epoch(2010, 1, 1), to_epoch(2026, 1, 1)),
        ("SPY", "1d", to_epoch(2010, 1, 1), to_epoch(2026, 1, 1)),
        ("^VIX", "1d", to_epoch(2010, 1, 1), to_epoch(2026, 1, 1)),
        ("^VIX9D", "1d", to_epoch(2010, 1, 1), to_epoch(2026, 1, 1)),
        ("^VIX3M", "1d", to_epoch(2010, 1, 1), to_epoch(2026, 1, 1)),
        ("^VIX6M", "1d", to_epoch(2010, 1, 1), to_epoch(2026, 1, 1)),
        ("^VVIX", "1d", to_epoch(2010, 1, 1), to_epoch(2026, 1, 1)),
        ("^SKEW", "1d", to_epoch(2010, 1, 1), to_epoch(2026, 1, 1)),
        # Intraday 1h: 728 days back from now
        ("SPY", "1h", period2_intraday - 728 * 86400, period2_intraday),
        ("^GSPC", "1h", period2_intraday - 728 * 86400, period2_intraday),
        # Intraday 5m: 58 days back from now
        ("SPY", "5m", period2_intraday - 58 * 86400, period2_intraday),
        ("^GSPC", "5m", period2_intraday - 58 * 86400, period2_intraday),
    ]

    data_dir = Path(__file__).parent.parent / "data"

    if dry_run:
        print("Dry run: planned URLs and periods:")
        print("=" * 100)
        for symbol, interval, period1, period2 in specs:
            url_encoded = urllib.parse.quote(symbol, safe="")
            url = f"{BASE_URL}{url_encoded}?period1={period1}&period2={period2}&interval={interval}"
            print(f"{symbol:10} {interval:5} p1={period1:10} p2={period2:10}")
            print(f"  {url}\n")
        sys.exit(0)

    # Real fetch
    print(f"Fetching {len(specs)} symbols...")
    entries = []
    for i, (symbol, interval, period1, period2) in enumerate(specs, 1):
        try:
            print(f"[{i}/{len(specs)}] Fetching {symbol} {interval}...", end=" ", flush=True)
            entry = fetch_one(symbol, interval, period1, period2, data_dir)
            entries.append(entry)
            print(f"✓ {entry['rows']} rows")
            time.sleep(1)
        except Exception as e:
            print(f"✗ ERROR: {e}")
            sys.exit(1)

    manifest_path = write_manifest(entries, data_dir)
    print(f"Manifest written to {manifest_path}")
    from rough_hedge.realdata import verify_manifest
    problems = verify_manifest(data_dir)
    print("✓ All verified" if not problems else f"✗ {len(problems)} problems")
