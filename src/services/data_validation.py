import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from config import EMAIL_RE, CONDITION_UNITS, CAPTURE_TYPES, ALLOWED_ORGANISMS, MASK_TYPES, SUPPORTED_EXTS


def _is_number(x: Any) -> bool:
    if x in ("", None):
        return True
    try:
        float(x)
        return True
    except Exception:
        return False
    
def validate_manifest_v1(
    manifest: Dict[str, Any],
    *,
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
    elif organism not in ALLOWED_ORGANISMS:
        exp_issues.append(f"Experiment: organism '{organism}' not in allowed list: {sorted(ALLOWED_ORGANISMS)}.")

    capture_type = (exp.get("capture_type") or "").strip()
    if not capture_type:
        exp_issues.append("Experiment: missing capture_type.")
    elif capture_type not in CAPTURE_TYPES:
        exp_issues.append(f"Experiment: capture_type '{capture_type}' must be one of {sorted(CAPTURE_TYPES)}.")

    # Email (optional but validate if present)
    email = (g.get("user_email") or "").strip()
    if email and not EMAIL_RE.match(email):
        exp_issues.append(f"Global defaults: invalid user_email '{email}'.")

    # Units sanity (only if values provided)
    if exp.get("concentration_unit"):
        u = str(exp.get("concentration_unit")).strip()
        if u and u not in CONDITION_UNITS:
            exp_issues.append(f"Experiment: concentration_unit '{u}' not in {sorted(CONDITION_UNITS)}.")

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
        if ext and str(ext).lower() not in SUPPORTED_EXTS:
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
            elif mt not in MASK_TYPES:
                issues.append(f"Mask: mask_type '{mt}' not in allowed list: {sorted(MASK_TYPES)}.")
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



def _is_number(x: Any) -> bool:
    if x in ("", None):
        return True
    try:
        float(x)
        return True
    except Exception:
        return False


def _validate_one_experiment(exp: Dict[str, Any], g: Dict[str, Any], group_name: str) -> List[str]:
    issues: List[str] = []
    prefix = f"Experiment group '{group_name}':"

    date = exp.get("date")
    if not date:
        issues.append(f"{prefix} missing date (YYYY-MM-DD).")
    else:
        try:
            datetime.strptime(str(date), "%Y-%m-%d")
        except ValueError:
            issues.append(f"{prefix} invalid date format '{date}', expected YYYY-MM-DD.")

    replicate = exp.get("replicate")
    if replicate in ("", None):
        issues.append(f"{prefix} missing replicate.")
    else:
        try:
            r = int(replicate)
            if r < 1:
                issues.append(f"{prefix} replicate must be >= 1.")
        except Exception:
            issues.append(f"{prefix} replicate is not an integer ('{replicate}').")

    organism = (exp.get("organism") or "").strip()
    if not organism:
        issues.append(f"{prefix} missing organism.")
    elif organism not in ALLOWED_ORGANISMS:
        issues.append(f"{prefix} organism '{organism}' not in allowed list: {sorted(ALLOWED_ORGANISMS)}.")

    capture_type = (exp.get("capture_type") or "").strip()
    if not capture_type:
        issues.append(f"{prefix} missing capture_type.")
    elif capture_type not in CAPTURE_TYPES:
        issues.append(f"{prefix} capture_type '{capture_type}' must be one of {sorted(CAPTURE_TYPES)}.")

    if exp.get("concentration_unit"):
        u = str(exp.get("concentration_unit")).strip()
        if u and u not in CONDITION_UNITS:
            issues.append(f"{prefix} concentration_unit '{u}' not in {sorted(CONDITION_UNITS)}.")

    for k in [
        "exposure_time",
        "time_interval",
        "concentration_value",
        "dye_concentration_value",
        "laser_wavelength",
        "laser_intensity",
        "pixel_size",
    ]:
        if k in exp and exp[k] not in ("", None) and not _is_number(exp[k]):
            issues.append(f"{prefix} '{k}' must be numeric (got '{exp[k]}').")

    if exp.get("camera_binning") not in ("", None):
        try:
            int(exp["camera_binning"])
        except Exception:
            issues.append(f"{prefix} camera_binning must be integer (got '{exp.get('camera_binning')}').")

    # Global email stays global
    email = (g.get("user_email") or "").strip()
    if email and not EMAIL_RE.match(email):
        issues.append(f"Global defaults: invalid user_email '{email}'.")

    return issues


def validate_manifest(
    manifest: Dict[str, Any]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns:
      exp_issues: list[str]  -> BLOCKERS
      file_issues: list[dict] -> SKIPPABLE
    """
    g = manifest.get("global_defaults") or {}
    exp_by_group = manifest.get("experiment_metadata_by_group") or {}
    files = manifest.get("files_resolved") or []

    exp_issues: List[str] = []
    file_issues: List[Dict[str, Any]] = []

    if not exp_by_group:
        exp_issues.append("Manifest missing 'experiment_metadata_by_group'.")

    if not files:
        exp_issues.append("Manifest has no 'files_resolved' rows.")
        return exp_issues, file_issues

    # Collect files by group
    files_by_group = defaultdict(list)
    for f in files:
        group = f.get("experiment_group")
        if not group:
            exp_issues.append(f"File '{f.get('file_name')}' is missing experiment_group.")
            continue
        files_by_group[group].append(f)

    # Check that every file group has metadata
    file_groups = set(files_by_group.keys())
    metadata_groups = set(exp_by_group.keys())

    missing_metadata_groups = sorted(file_groups - metadata_groups)
    #extra_metadata_groups = sorted(metadata_groups - file_groups)

    for group in missing_metadata_groups:
        exp_issues.append(f"Experiment metadata missing for group '{group}'.")

    #for group in extra_metadata_groups:
        #exp_issues.append(f"Experiment metadata exists for unused group '{group}'.")

    # Validate each experiment group
    for group in sorted(file_groups & metadata_groups):
        exp = exp_by_group.get(group) or {}
        exp_issues.extend(_validate_one_experiment(exp, g, group))

    # File-level validation
    for i, f in enumerate(files):
        issues: List[str] = []
        dt = f.get("data_type")
        path = f.get("path")
        fname = f.get("file_name")
        ext = f.get("ext")
        group = f.get("experiment_group")

        if not group:
            issues.append("Missing experiment_group.")
        elif group not in exp_by_group:
            issues.append(f"Unknown experiment_group '{group}'.")

        if not dt:
            issues.append("Missing data_type.")
        if not fname:
            issues.append("Missing file_name.")
        if not path:
            issues.append("Missing path.")
        if ext and str(ext).lower() not in SUPPORTED_EXTS:
            issues.append(f"Unsupported extension '{ext}'.")

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
            elif mt not in MASK_TYPES:
                issues.append(f"Mask: mask_type '{mt}' not in allowed list: {sorted(MASK_TYPES)}.")

            sm = (f.get("segmentation_method") or "").strip()
            if not sm:
                issues.append("Mask: missing segmentation_method.")

        if issues:
            file_issues.append({
                "index": i,
                "file_name": fname,
                "data_type": dt,
                "path": path,
                "experiment_group": group,
                "issues": issues,
            })

    return exp_issues, file_issues

