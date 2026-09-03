"""
MNI coordinate -> anatomical label lookup.

Replaces eyeballing MNI coordinates against a viewer for searchlight/cluster
peak tables. Primary atlas is Harvard-Oxford (cortical + subcortical,
maxprob-thr25-2mm) — standard choice for readable fMRI cluster-table labels
(e.g. "Inferior Frontal Gyrus, pars opercularis" rather than AAL's
"Frontal_Inf_Oper_R"). AAL (SPM12 version) is included as a cross-check,
matching the atlas already used elsewhere in this repo for anatomical ROI
masks (notebooks/roi/ROI_masks.ipynb: parietal/putamen AAL masks).

Both atlases are already cached locally (~/nilearn_data/fsl,
~/nilearn_data/aal_SPM12) from prior sessions — fetch_atlas_harvard_oxford
and fetch_atlas_aal(version='SPM12') resolve offline. Do NOT call
fetch_atlas_aal() without version='SPM12' — the default version (AAL3v2)
is not cached and requires an internet fetch that fails in this environment
(SSL error against www.gin.cnrs.fr, confirmed 2026-09-03).

Usage
-----
from utils.atlas import label_coordinates

peaks = [(-48, -81, -1), (42, -54, -18.5), (45, 3, 34)]
labels = label_coordinates(peaks)
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_aal, fetch_atlas_harvard_oxford


@lru_cache(maxsize=1)
def _load_atlases():
    cort = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    aal = fetch_atlas_aal(version="SPM12")

    cort_data = cort.maps.get_fdata().astype(int)
    sub_data = sub.maps.get_fdata().astype(int)
    aal_img = aal.maps if not isinstance(aal.maps, str) else __import__("nibabel").load(aal.maps)
    aal_data = aal_img.get_fdata().astype(int)
    aal_value_to_name = {int(idx): name for name, idx in zip(aal.labels, aal.indices)}

    return {
        "cort_data": cort_data, "cort_labels": list(cort.labels), "cort_affine": cort.maps.affine,
        "sub_data": sub_data, "sub_labels": list(sub.labels), "sub_affine": sub.maps.affine,
        "aal_data": aal_data, "aal_value_to_name": aal_value_to_name, "aal_affine": aal_img.affine,
    }


def _nearest_nonbackground(data, affine, xyz, max_radius_mm=15, step_mm=2):
    """Return (label_index, distance_mm) of the nearest non-zero voxel to xyz.

    Searches an expanding cube of voxel offsets (in atlas voxel units,
    assumed ~2mm isotropic) up to max_radius_mm. Used when a peak coordinate
    falls in unlabeled space (atlas gaps at thr25 boundaries, ventricle,
    or genuine white matter for the cortical-only map).
    """
    inv_affine = np.linalg.inv(affine)
    ijk = np.round(inv_affine @ np.array([*xyz, 1]))[:3].astype(int)

    if np.all((0 <= ijk) & (ijk < np.array(data.shape))):
        val = data[tuple(ijk)]
        if val != 0:
            return val, 0.0

    voxel_size = np.abs(np.diag(affine)[:3]).mean()
    max_offset = int(np.ceil(max_radius_mm / voxel_size))
    best = (0, np.inf)
    for dx in range(-max_offset, max_offset + 1):
        for dy in range(-max_offset, max_offset + 1):
            for dz in range(-max_offset, max_offset + 1):
                dist_vox = np.sqrt(dx**2 + dy**2 + dz**2)
                dist_mm = dist_vox * voxel_size
                if dist_mm > max_radius_mm or dist_mm >= best[1]:
                    continue
                voxel = ijk + np.array([dx, dy, dz])
                if not np.all((0 <= voxel) & (voxel < np.array(data.shape))):
                    continue
                val = data[tuple(voxel)]
                if val != 0:
                    best = (val, dist_mm)
    return best if best[1] != np.inf else (0, np.nan)


def label_coordinates(coords, max_radius_mm=15):
    """Label a list of MNI coordinates against Harvard-Oxford (+ AAL cross-check).

    Parameters
    ----------
    coords : iterable of (x, y, z) tuples, MNI mm.
    max_radius_mm : float
        Search radius for the nearest-non-background fallback when a
        coordinate lands on an atlas gap (default 15mm).

    Returns
    -------
    pd.DataFrame with columns:
        x, y, z, harvard_oxford, ho_distance_mm, aal, aal_distance_mm
    harvard_oxford prefers the subcortical atlas when both cortical and
    subcortical agree on non-background (i.e. subcortical takes precedence
    for striatum/thalamus/hippocampus/amygdala, matching standard practice
    of reporting the more specific structure).
    """
    atlases = _load_atlases()
    rows = []
    for x, y, z in coords:
        xyz = (float(x), float(y), float(z))

        sub_val, sub_dist = _nearest_nonbackground(
            atlases["sub_data"], atlases["sub_affine"], xyz, max_radius_mm)
        cort_val, cort_dist = _nearest_nonbackground(
            atlases["cort_data"], atlases["cort_affine"], xyz, max_radius_mm)
        aal_val, aal_dist = _nearest_nonbackground(
            atlases["aal_data"], atlases["aal_affine"], xyz, max_radius_mm)

        # Prefer subcortical only for actual subcortical/non-cortex structures
        # (skip "Left/Right Cerebral Cortex" and "Cerebral White Matter", which
        # are uninformative next to a real cortical gyrus label).
        sub_name = atlases["sub_labels"][sub_val] if sub_val else None
        cort_name = atlases["cort_labels"][cort_val] if cort_val else None
        is_generic_sub = sub_name is not None and (
            "Cerebral Cortex" in sub_name or "Cerebral White Matter" in sub_name
            or "Ventricle" in sub_name)

        if sub_name is not None and not is_generic_sub:
            ho_label, ho_dist = sub_name, sub_dist
        elif cort_name is not None:
            ho_label, ho_dist = cort_name, cort_dist
        elif sub_name is not None:
            ho_label, ho_dist = sub_name, sub_dist
        else:
            ho_label, ho_dist = "no label found", np.nan

        aal_name = atlases["aal_value_to_name"].get(int(aal_val), "no label found") if aal_val else "no label found"

        rows.append({
            "x": x, "y": y, "z": z,
            "harvard_oxford": ho_label, "ho_distance_mm": round(ho_dist, 1),
            "aal": aal_name, "aal_distance_mm": round(aal_dist, 1),
        })
    return pd.DataFrame(rows)
