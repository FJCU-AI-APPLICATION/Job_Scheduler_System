"""Download INRC-I instances on first use.

Pulls the official PATAT mirror's instance archive, then extracts the
nested sprint zip into data/benchmarks/inrc1/. The repo intentionally
ships only a single bundled fixture so non-bench developers don't pay
the download cost on a normal install.

Usage:
    python scripts/fetch_inrc1.py
"""

from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

CANDIDATE_URLS = [
    "https://patat.cs.kuleuven.be/sites/patat.cs.kuleuven.be/files/inrc1/instances.zip",
]
TARGET_DIR = Path("data/benchmarks/inrc1")
SPRINT_INNER_PREFIX = "INRC-I - instances/sprint/sprint.zip"


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for url in CANDIDATE_URLS:
        try:
            print(f"Attempting download from {url}")
            data = urllib.request.urlopen(url, timeout=60).read()
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  FAILED: {e}")
            continue

        try:
            outer = zipfile.ZipFile(io.BytesIO(data))
            sprint_zip_bytes = outer.read(SPRINT_INNER_PREFIX)
            inner = zipfile.ZipFile(io.BytesIO(sprint_zip_bytes))
            extracted = 0
            for member in inner.namelist():
                if member.endswith(".xml"):
                    inner.extract(member, TARGET_DIR)
                    extracted += 1
            print(f"  Extracted {extracted} sprint instances to {TARGET_DIR}")
            return 0
        except (zipfile.BadZipFile, KeyError) as e:
            print(f"  FAILED to extract sprint instances: {e}")

    print(
        "\nAll mirrors failed. To populate the benchmark manually:\n"
        "  1. Obtain the INRC-I 'instances.zip' from Haspeslagh et al. 2014\n"
        "     (Annals of OR 218(1), https://doi.org/10.1007/s10479-014-1683-6)\n"
        f"  2. Extract sprint01.xml..sprint10.xml into {TARGET_DIR}/\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
