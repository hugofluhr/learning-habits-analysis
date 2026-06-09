#!/usr/bin/env python3
"""
Build visual cortex mask from Harvard-Oxford atlas — run once locally, then scp to cluster.

Usage
-----
python multivariate/build_visual_cortex_mask.py --output-dir /tmp/decoding
# Saves: <output-dir>/visual_cortex_mask.nii.gz  (MNI152 2mm space)
"""

import argparse
import numpy as np
from pathlib import Path
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn import image

VISUAL_KEYWORDS = [
    'Occipital', 'Lingual', 'Cuneal', 'Intracalcarine',
    'Fusiform', 'Temporal Occipital',
]


def build_mask(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'visual_cortex_mask.nii.gz'

    atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_data = atlas.maps.get_fdata()

    vis_data = np.zeros(atlas_data.shape, dtype=np.uint8)
    print("Regions included:")
    for i, label in enumerate(atlas.labels):
        if any(k in label for k in VISUAL_KEYWORDS):
            vis_data[atlas_data == i] = 1
            print(f"  {i:3d}  {label}")

    vis_mask = image.new_img_like(atlas.maps, vis_data)
    vis_mask.to_filename(str(out_path))
    print(f"\nTotal voxels: {int(vis_data.sum()):,}")
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build visual cortex mask from Harvard-Oxford atlas.")
    parser.add_argument("--output-dir", required=True, help="Directory to save visual_cortex_mask.nii.gz")
    args = parser.parse_args()
    build_mask(args.output_dir)


if __name__ == "__main__":
    main()
