#!/usr/bin/env python3
"""Download a frozen five-graph G-set Max-Cut benchmark, with literature optima.

Usage:
    python3 download_gset.py
    python3 download_gset.py --output data/gset
    python3 download_gset.py --ids G11 G57 --output data/gset
    python3 download_gset.py --verify-only --output data/gset
    python3 download_gset.py --force --output data/gset

No third-party packages required. The optimal values below are *reported by*
Glass & Feder (2026), arXiv:2606.28478v1, Table 1. This script does not
independently reproduce the mathematical upper-bound proofs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

STANFORD_BASE = "https://web.stanford.edu/~yyye/yyye/Gset"
PAPER_URL = "https://arxiv.org/html/2606.28478"
PAPER_ID = "arXiv:2606.28478v1"
PAPER_TITLE = "Maximum Cut Algorithms and Upper Bounds for Planar and Toroidal Graphs"
PAPER_AUTHORS = ["Mark Glass", "Meir Feder"]

# Official G-set instance sizes; reported optimal cuts and upper bounds are
# copied from Table 1 of Glass & Feder (arXiv:2606.28478v1, 2026-06-26).
# Both are equal for these five graphs, giving literature-level certification.
BENCHMARKS = {
    "G11": {"vertices": 800, "edges": 1600, "optimal_cut": 564, "upper_bound": 564},
    "G57": {"vertices": 5000, "edges": 10000, "optimal_cut": 3494, "upper_bound": 3494},
    "G62": {"vertices": 7000, "edges": 14000, "optimal_cut": 4872, "upper_bound": 4872},
    "G72": {
        "vertices": 10000,
        "edges": 20000,
        "optimal_cut": 7008,
        "upper_bound": 7008,
    },
    "G81": {
        "vertices": 20000,
        "edges": 40000,
        "optimal_cut": 14060,
        "upper_bound": 14060,
    },
}
MAX_BYTES = 16 * 1024 * 1024


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_and_validate_gset(data: bytes, name: str) -> tuple[int, int]:
    """Reject damaged / unexpected files without changing their original bytes."""
    try:
        lines = [
            line.strip() for line in data.decode("ascii").splitlines() if line.strip()
        ]
    except UnicodeError as exc:
        raise ValueError(f"{name}: G-set must be ASCII text") from exc
    if not lines:
        raise ValueError(f"{name}: empty file")
    try:
        n, m = (int(v) for v in lines[0].split())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: invalid N M header") from exc

    expected = BENCHMARKS[name]
    if (n, m) != (expected["vertices"], expected["edges"]):
        raise ValueError(
            f"{name}: expected N,M=({expected['vertices']},{expected['edges']}), got ({n},{m})"
        )
    if len(lines) != m + 1:
        raise ValueError(f"{name}: expected {m} edge lines, got {len(lines) - 1}")

    seen: set[tuple[int, int]] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            u, v, weight = (int(token) for token in line.split())
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}: invalid edge at line {line_number}") from exc
        if u < 1 or v < 1 or u > n or v > n or u == v:
            raise ValueError(
                f"{name}: invalid endpoint/self-loop at line {line_number}"
            )
        if weight not in (-1, +1):
            raise ValueError(
                f"{name}: unexpected edge weight {weight} at line {line_number}"
            )
        edge = (min(u, v), max(u, v))
        if edge in seen:
            raise ValueError(
                f"{name}: duplicate undirected edge {edge} at line {line_number}"
            )
        seen.add(edge)
    return n, m


def fetch_bytes(url: str, retries: int, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "ising-solver-gset-downloader/1.0"})
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                # Limit download size in case of incorrect redirects / remote errors.
                payload = response.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                raise ValueError(f"response exceeds {MAX_BYTES} bytes: {url}")
            return payload
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(
                    f"download failed after {retries} attempts: {url}: {exc}"
                ) from exc
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as fp:
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def existing_manifest(output: Path) -> dict:
    path = output / "manifest.json"
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in obj.get("instances", [])}


def write_metadata(output: Path, entries: dict[str, dict]) -> None:
    records = [entries[name] for name in BENCHMARKS if name in entries]
    manifest = {
        "schema_version": 1,
        "dataset": "Stanford G-set / Max-Cut / selected proven-optimal toroidal cases",
        "generated_at_utc": now_utc(),
        "notes": "Reported proof from a research preprint; not independently certified by this downloader.",
        "reference": {
            "paper": PAPER_TITLE,
            "authors": PAPER_AUTHORS,
            "arxiv": PAPER_ID,
            "url": PAPER_URL,
            "table": "Table 1",
            "date": "2026-06-26",
        },
        "instances": records,
    }
    atomic_write(
        output / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )

    columns = [
        "id",
        "vertices",
        "edges",
        "optimal_cut",
        "upper_bound",
        "proof_status",
        "proof_reference",
        "graph_source_url",
        "sha256",
        "downloaded_at_utc",
        "validated_at_utc",
        "file",
    ]
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for record in records:
        writer.writerow({col: record.get(col, "") for col in columns})
    atomic_write(output / "optima.csv", buf.getvalue().encode("utf-8"))


def run() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gset"),
        help="Download directory (default: data/gset)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        choices=list(BENCHMARKS),
        default=list(BENCHMARKS),
        help="Subset to process, default all five",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate cached files without network access",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if valid files are already cached",
    )
    parser.add_argument(
        "--timeout", type=float, default=20.0, help="HTTP timeout in seconds"
    )
    parser.add_argument("--retries", type=int, default=3, help="Max download attempts")
    args = parser.parse_args()
    if args.verify_only and args.force:
        parser.error("--verify-only and --force cannot be used together")
    if args.timeout <= 0 or args.retries <= 0:
        parser.error("--timeout and --retries must be positive")

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    prior = existing_manifest(output)
    merged = dict(prior)
    errors = []
    for name in dict.fromkeys(args.ids):
        path = output / name
        url = f"{STANFORD_BASE}/{name}"
        try:
            if path.exists() and not args.force:
                payload = path.read_bytes()
                status = "cached"
            elif args.verify_only:
                raise FileNotFoundError(f"missing cached graph: {path}")
            else:
                payload = fetch_bytes(url, retries=args.retries, timeout=args.timeout)
                status = "downloaded"
            n, m = parse_and_validate_gset(payload, name)
            sha256 = hashlib.sha256(payload).hexdigest()
            if (
                path.exists()
                and not args.force
                and name in prior
                and prior[name].get("sha256") != sha256
            ):
                raise ValueError(
                    "SHA-256 differs from saved manifest; inspect data or use --force"
                )
            if status == "downloaded":
                atomic_write(path, payload)  # Write only after complete validation.

            ref = BENCHMARKS[name]
            cached_download_time = prior.get(name, {}).get("downloaded_at_utc")
            record = {
                "id": name,
                "file": name,
                "vertices": n,
                "edges": m,
                "optimal_cut": ref["optimal_cut"],
                "upper_bound": ref["upper_bound"],
                "proof_status": "reported_proven_optimal_in_paper",
                "proof_reference": f"{PAPER_ID}, Table 1",
                "graph_source_url": url,
                "sha256": sha256,
                "downloaded_at_utc": (
                    cached_download_time
                    if status == "cached"
                    else now_utc()
                    if status == "downloaded"
                    else None
                ),
                "validated_at_utc": now_utc(),
            }
            merged[name] = record
            print(
                f"[OK] {name:4} {status:10} N={n:<6} M={m:<6} optimum={ref['optimal_cut']:<6} sha256={sha256[:12]}..."
            )
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{name}: {exc}")
            print(f"[ERROR] {name}: {exc}", file=sys.stderr)

    # Keep prior entries when using --ids, but never rewrite meta files after failed verification:
    # their existing records might be stale until the user resolves the mismatch.
    if errors:
        print(
            "Failure: manifest not rewritten; correct errors and rerun.",
            file=sys.stderr,
        )
        return 1
    write_metadata(output, merged)
    print(f"\nSaved {output / 'manifest.json'} and {output / 'optima.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
