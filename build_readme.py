#!/usr/bin/env python3
"""Assemble README.md from docs/sections/ + paper_data.json.

Same pattern as the paper build system:
1. Load all .md files from docs/sections/ in order
2. Load paper_data.json
3. Replace {{key}} placeholders
4. Write README.md
"""

import json
import re
import sys
from pathlib import Path

SECTIONS_DIR = Path(__file__).parent / "docs" / "sections"
DATA_FILE = Path(__file__).parent / "paper_data.json"
OUTPUT_FILE = Path(__file__).parent / "README.md"


def build():
    print(f"Loading {DATA_FILE}...", flush=True)
    try:
        data = json.load(open(DATA_FILE))
    except FileNotFoundError:
        print(f"ERROR: {DATA_FILE} not found. Run build_data.py first.", flush=True)
        sys.exit(1)

    sections = sorted(SECTIONS_DIR.glob("*.md"))
    if not sections:
        print(f"ERROR: No .md files in {SECTIONS_DIR}/", flush=True)
        sys.exit(1)

    print(f"Found {len(sections)} sections:", flush=True)
    for s in sections:
        print(f"  {s.name}", flush=True)

    parts = [s.read_text() for s in sections]
    full_text = "\n\n".join(parts)

    missing = []
    used_keys = set()

    def replace_var(match):
        key = match.group(1).strip()
        if key in data:
            used_keys.add(key)
            return str(data[key])
        missing.append(key)
        return match.group(0)

    result = re.sub(r'\{\{(\w+)\}\}', replace_var, full_text)

    if missing:
        unique_missing = sorted(set(missing))
        print(f"\nWARNING: {len(unique_missing)} unresolved keys:", flush=True)
        for k in unique_missing:
            print(f"  - {k}", flush=True)

    OUTPUT_FILE.write_text(result)
    n_lines = result.count('\n')
    print(f"\nWrote {OUTPUT_FILE}", flush=True)
    print(f"  {n_lines:,} lines, {len(result):,} chars", flush=True)
    print(f"  {len(used_keys)} data keys used", flush=True)

    remaining = re.findall(r'\{\{(\w+)\}\}', result)
    if remaining:
        print(f"  {len(set(remaining))} unresolved placeholders remain", flush=True)
    else:
        print("  All placeholders resolved", flush=True)


if __name__ == "__main__":
    build()
