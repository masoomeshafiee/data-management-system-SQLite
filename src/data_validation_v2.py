import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from lab_db_app.config import  EMAIL_RE

def _is_number(x: Any) -> bool:
    if x in ("", None):
        return True
    try:
        float(x)
        return True
    except Exception:
        return False

def validate_manifest(
    manifest: Dict[str, Any],
    *,
    allowed_capture_types: set,
    allowed_organisms: set,
    condition_units: set,
    mask_types: set,
    supported_exts: set,
    require_path_exists: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    
    """
    Returns:
      exp_issues: list[str]  -> BLOCKERS
      file_issues: list[dict] -> SKIPPABLE (if allow_partial_files=True)
    """
    exp = manifest.get("experiment") or {}
    g = manifest.get("global_defaults") or {}
    files = manifest.get("files_resolved") or []

    exp_issues: List[str] = []
    file_issues: List[Dict[str, Any]] = []

    # -------------------
    # Experiment-level (blocker)
    # -------------------
    date = exp.get("date")
    if not date:
        exp_issues.append("Experiment: missing date (YYYY-MM-DD).")
    else:
        try:
            datetime.strptime(str(date), "%Y-%m-%d")
        except ValueError:
            exp_issues.append(f"Experiment: invalid date format '{date}', expected YYYY-MM-DD.")

    replicate = exp.get("replicate")
    if replicate in ("", None):
        exp_issues.append("Experiment: missing replicate.")
    else:
        try:
            r = int(replicate)
            if r < 1:
                exp_issues.append("Experiment: replicate must be >= 1.")
        except Exception:
            exp_issues.append(f"Experiment: replicate is not an integer ('{replicate}').")

    organism = (exp.get("organism") or "").strip()
    if not organism:
        exp_issues.append("Experiment: missing organism.")
    elif organism not in allowed_organisms:
        exp_issues.append(f"Experiment: organism '{organism}' not in allowed list: {sorted(allowed_organisms)}.")

    capture_type = (exp.get("capture_type") or "").strip()
    if not capture_type:
        exp_issues.append("Experiment: missing capture_type.")
    elif capture_type not in allowed_capture_types:
        exp_issues.append(f"Experiment: capture_type '{capture_type}' must be one of {sorted(allowed_capture_types)}.")

    # Email (optional but validate if present)
    email = (g.get("user_email") or "").strip()
    if email and not EMAIL_RE.match(email):
        exp_issues.append(f"Global defaults: invalid user_email '{email}'.")

    # Units sanity (only if values provided)
    if exp.get("concentration_unit"):
        u = str(exp.get("concentration_unit")).strip()
        if u and u not in condition_units:
            exp_issues.append(f"Experiment: concentration_unit '{u}' not in {sorted(condition_units)}.")

    # numeric sanity (only if set)
    for k in ["exposure_time", "time_interval", "concentration_value", "dye_concentration_value",
              "laser_wavelength", "laser_intensity", "pixel_size"]:
        if k in exp and exp[k] not in ("", None) and not _is_number(exp[k]):
            exp_issues.append(f"Experiment: '{k}' must be numeric (got '{exp[k]}').")

    if exp.get("camera_binning") not in ("", None):
        try:
            int(exp["camera_binning"])
        except Exception:
            exp_issues.append(f"Experiment: camera_binning must be integer (got '{exp.get('camera_binning')}').")


    # -------------------
    # File-level (skippable)
    # -------------------
    for i, f in enumerate(files):
        issues: List[str] = []
        dt = f.get("data_type")
        path = f.get("path")
        fname = f.get("file_name")
        ext = f.get("ext")

        # basics
        if not dt:
            issues.append("Missing data_type.")
        if not fname:
            issues.append("Missing file_name.")
        if not path:
            issues.append("Missing path.")
        if ext and str(ext).lower() not in supported_exts:
            issues.append(f"Unsupported extension '{ext}'.")

        if require_path_exists and path and (not os.path.exists(path)):
            issues.append(f"Path does not exist on server: {path}")

        # type-specific
        if dt == "tracking":
            if f.get("threshold") in ("", None):
                issues.append("Tracking: missing threshold.")
            elif not _is_number(f.get("threshold")):
                issues.append(f"Tracking: threshold not numeric ('{f.get('threshold')}').")
            for k in ["linking_distance", "gap_closing_distance"]:
                if f.get(k) not in ("", None) and not _is_number(f.get(k)):
                    issues.append(f"Tracking: {k} not numeric ('{f.get(k)}').")
            if f.get("max_frame_gap") not in ("", None):
                try:
                    int(f.get("max_frame_gap"))
                except Exception:
                    issues.append(f"Tracking: max_frame_gap not integer ('{f.get('max_frame_gap')}').")

        if dt == "mask":
            mt = (f.get("mask_type") or "").strip()
            if not mt:
                issues.append("Mask: missing mask_type.")
            elif mt not in mask_types:
                issues.append(f"Mask: mask_type '{mt}' not in allowed list: {sorted(mask_types)}.")
            sm = (f.get("segmentation_method") or "").strip()
            if not sm:
                issues.append("Mask: missing segmentation_method.")

        # analysis types: no extra requirements right now

        if issues:
            file_issues.append({
                "index": i,
                "file_name": fname,
                "data_type": dt,
                "path": path,
                "issues": issues,
            })

    return exp_issues, file_issues