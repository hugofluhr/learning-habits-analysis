#!/usr/bin/env python3
"""Extract base64-encoded PNG outputs from executed Jupyter notebooks
and save them to presentation_assets/ for the Marp slide deck.

Each entry maps a target filename to (notebook, cell_index, output_index).
output_index selects among multiple image outputs in the same cell (0-based).
"""

import base64
import json
import os
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "presentation_assets"

# (target_filename, notebook, cell_index, output_index_within_cell)
FIGURES = [
    ("19-rsa-5term-bar.png",              "rsa_roi_results.ipynb",              11, 0),
    ("20-rsa-shuffled.png",               "rsa_roi_results.ipynb",              19, 0),
    ("21-rsa-interaction.png",            "rsa_roi_results.ipynb",              28, 0),
    ("22-rsa-interaction-validation.png", "rsa_roi_results.ipynb",              30, 0),
    ("23-freq-decoding-roi.png",          "frequency_decoding_results.ipynb",    6, 0),
    ("24-freq-searchlight-tmap.png",      "frequency_searchlight_results.ipynb", 8, 0),
    ("25-freq-searchlight-roi.png",       "frequency_searchlight_results.ipynb", 13, 0),
    ("26-rsa-design-counterbalance.png",  "rsa_design_checks.ipynb",             7, 0),
    ("27-rsa-sl-fdr-maps.png",            "rsa_searchlight_results.ipynb",       7, 0),
    ("28-rsa-sl-roi-bar.png",             "rsa_searchlight_results.ipynb",      12, 0),
    ("29-rsa-sl-vs-freqdecode.png",       "rsa_searchlight_results.ipynb",      14, 0),
    ("30-rsa-sl-interaction.png",         "rsa_searchlight_results.ipynb",      21, 0),
]


def extract_image(nb_path: Path, cell_idx: int, output_idx: int) -> bytes:
    """Return raw PNG bytes from the given cell's display output."""
    with open(nb_path) as f:
        nb = json.load(f)
    cell = nb["cells"][cell_idx]
    img_outputs = [
        o for o in cell.get("outputs", [])
        if o.get("output_type") in ("display_data", "execute_result")
        and "image/png" in o.get("data", {})
    ]
    if not img_outputs:
        raise ValueError(f"{nb_path.name} cell {cell_idx}: no image/png outputs found")
    if output_idx >= len(img_outputs):
        raise IndexError(
            f"{nb_path.name} cell {cell_idx}: requested output {output_idx} "
            f"but only {len(img_outputs)} image outputs exist"
        )
    b64 = img_outputs[output_idx]["data"]["image/png"]
    # Handle both single-string and list-of-lines encodings
    if isinstance(b64, list):
        b64 = "".join(b64)
    return base64.b64decode(b64)


def main():
    nb_dir = Path(__file__).parent
    ASSETS_DIR.mkdir(exist_ok=True)

    for target, nb_name, cell_idx, out_idx in FIGURES:
        nb_path = nb_dir / nb_name
        print(f"Extracting {target} from {nb_name} cell {cell_idx} output {out_idx}...")
        try:
            data = extract_image(nb_path, cell_idx, out_idx)
        except (ValueError, IndexError, KeyError) as e:
            print(f"  ERROR: {e}")
            continue

        out_path = ASSETS_DIR / target
        out_path.write_bytes(data)
        size_kb = len(data) / 1024
        print(f"  -> {out_path.name}  ({size_kb:.1f} KB)")
        if size_kb < 5:
            print(f"  WARNING: file is only {size_kb:.1f} KB — may be a thumbnail or empty")

    print(f"\nDone. Files in {ASSETS_DIR}/:")
    for p in sorted(ASSETS_DIR.glob("*.png")):
        print(f"  {p.name:45s}  {p.stat().st_size/1024:7.1f} KB")


if __name__ == "__main__":
    main()
