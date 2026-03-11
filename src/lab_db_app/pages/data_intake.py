from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import streamlit as st


# ----------------------------
# Config / constants
# ----------------------------
SUPPORTED_EXTS = {".csv", ".tif", ".tiff",".TIF", ".nd", ".npy", ".png", ".txt", ".json", ".mat", ".pickle", ".svg", ".pdf", ".xlsx"}
CAPTURE_TYPES = ["confocal", "fast", "long"]
ALLOWED_ORGANISMS = ["human", "yeast", "E.coli"]
MASK_TYPES = ["cell", "nucleus", "nucleus-g1", "membrane", "cytoplasm"]
DYE_CONCENTRATION_UNITS = ["pM","nM", "uM", "mM", "M", "N/A"]
CONDITION_UNITS = ["N/A","nM", "uM", "mM", "M", "%", "mJ/m2", "mJ/cm2", "J/cm2", "J/m2"]
ROLE_OPTIONS = ["unassigned", "raw", "mask", "tracking", "analysis", "batch_analysis", "plot", "config", "ignore"]


# Optional folder-name hints (NOT required)
FOLDER_HINTS = {
    "mask": "mask",
    "masks": "mask",
    "seg": "mask",
    "segmentation": "mask",
    "track": "tracking",
    "tracks": "tracking",
    "spot": "tracking",
    "spots": "tracking",
    "analysis": "analysis",
    "results": "analysis",
    "raw": "raw",
    "video": "raw",
    "images": "raw",
    "projection": "raw"
}


# ----------------------------
# Helpers
# ----------------------------
def is_hidden_or_system_file(p: Path) -> bool:
    name = p.name
    if name.startswith("._") or name.startswith("."):
        return True
    return False


def safe_stat(p: Path) -> Tuple[Optional[int], Optional[float]]:
    try:
        st_ = p.stat()
        return int(st_.st_size), float(st_.st_mtime)
    except Exception:
        return None, None


def extract_field_of_view(file_name: str) -> str:
    """
    Optional. Best-effort only.
    If no match, return empty string.
    """
    image_match = re.search(r"([0-9]+)_w.*\.(TIF|tif|png|npy)$", file_name)
    if image_match:
        return image_match.group(1)

    nd_match = re.search(r"([0-9]+)\.nd$", file_name)
    if nd_match:
        return nd_match.group(1)

    stream_match = re.search(r"Stream(\d+)", file_name, flags=re.IGNORECASE)
    if stream_match:
        return stream_match.group(1)

    snap_match = re.search(r"Snap(\d+)", file_name, flags=re.IGNORECASE)
    if snap_match:
        return snap_match.group(1)

    return ""

def _clean_str(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _none_if_blank(x: Any) -> Optional[str]:
    s = _clean_str(x)
    return s if s else None

def _none_if_zero_float(x: float) -> Optional[float]:
    return None if x == 0.0 else float(x)


def _none_if_zero_int(x: int) -> Optional[int]:
    return None if x == 0 else int(x)

def suggest_role_from_folder(relative_folder: str) -> str:
    parts = [p.lower() for p in Path(relative_folder).parts]
    for part in reversed(parts):
        if part in FOLDER_HINTS:
            return FOLDER_HINTS[part]
    return "unassigned"


def scan_experiment_folder(root_folder: Path) -> pd.DataFrame:
    rows: List[dict] = []
    root_folder = root_folder.resolve()

    for p in root_folder.rglob("*"):
        if not p.is_file():
            continue
        if is_hidden_or_system_file(p):
            continue

        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue

        rel_folder = str(p.parent.relative_to(root_folder))
        size_bytes, mtime = safe_stat(p)

        rows.append(
            {
                "relative_folder": "." if rel_folder == "" else rel_folder,
                "file_name": p.name,
                "field_of_view": extract_field_of_view(p.name),
                "ext": ext,
                "full_path": str(p),
                "size_mb": None if size_bytes is None else round(size_bytes / (1024 * 1024), 3),
                "modified": None if mtime is None else datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                "suggested_data_type": suggest_role_from_folder(rel_folder),
                "data_type": "unassigned",  # user-assigned / rule-assigned
                "file_overrides_json": "",

                # Raw overrides --> for the future if needed. 

                # Tracking overrides
                "ov_threshold": None,
                "ov_linking_distance": None,
                "ov_gap_closing_distance": None,
                "ov_max_frame_gap": None,

                # Mask overrides
                "ov_segmentation_method": "",
                "ov_segmentation_parameters": "",
                "ov_mask_type": "",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Initial role = suggested_role (but user can override)
    df["data_type"] = df["suggested_data_type"].where(df["suggested_data_type"].isin(ROLE_OPTIONS), "unassigned")

    # Stable ordering
    df = df.sort_values(["relative_folder", "file_name"]).reset_index(drop=True)
    return df



def apply_folder_rule(df: pd.DataFrame, folder: str, role: str) -> pd.DataFrame:
    if role not in ROLE_OPTIONS:
        return df
    out = df.copy()
    out.loc[out["relative_folder"] == folder, "data_type"] = role
    return out


def apply_ext_rule(df: pd.DataFrame, ext: str, role: str) -> pd.DataFrame:
    if role not in ROLE_OPTIONS:
        return df
    out = df.copy()
    out.loc[out["ext"] == ext, "data_type"] = role
    return out


def validate_manifest_df(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    if df.empty:
        issues.append("No files found (or none match supported extensions).")
        return issues

    unassigned = df[df["data_type"] == "unassigned"]
    if not unassigned.empty:
        issues.append(f"{len(unassigned)} file(s) are still unassigned. Assign a data type or set to ignore.")

    # Duplicates by full_path
    dup = df["full_path"].duplicated().sum()
    if dup:
        issues.append(f"Duplicate paths detected ({dup}). This usually indicates scan issues or symlinks.")

    # Validate overrides JSON if present
    bad_json = []
    for i, v in df["file_overrides_json"].fillna("").items():
        if str(v).strip() == "":
            continue
        try:
            json.loads(v)
        except Exception:
            bad_json.append(i)
    if bad_json:
        issues.append(f"{len(bad_json)} file(s) have invalid JSON in file_overrides_json (rows: {bad_json[:10]}...).")

    return issues


# -----------------------------
# Data classes for metadata
# -----------------------------
@dataclass
class GlobalDefaults:
    user_name: str
    user_last_name: str
    user_email: str
    fluorescent_dye: str
    dye_concentration_value: Optional[float]
    dye_concentration_unit: str
    objective_magnification: Optional[float]
    laser_wavelength: Optional[float]
    laser_intensity: Optional[float]
    camera_binning: Optional[int]
    pixel_size: Optional[float]

@dataclass
class ExperimentMetadata:
    # keep minimal for now; we will connect to DB lookups later
    date: str
    replicate: Optional[int]
    organism: str
    protein: str
    strain: str
    
    capture_type: str
    exposure_time: Optional[float]
    time_interval: Optional[float]

    condition_name: str
    concentration_value: Optional[float]
    concentration_unit: str

    is_valid: bool
    comment: Optional[str]

    experiment_path: str

@dataclass
class TypeDefaults:
    # Masks
    segmentation_method: str
    segmentation_parameters: str
    mask_type: str

    # Tracking
    linking_distance: Optional[float]
    gap_closing_distance: Optional[float]
    max_frame_gap: Optional[int]
    threshold: Optional[float]
    trackmate_settings_json: Optional[dict]  # loaded from uploaded JSON (optional)



# ----------------------------
# Resolution logic
# Precedence:
# Per-file overrides > Per-type defaults > Per-experiment overrides > Global defaults
# ----------------------------
def resolve_file_metadata(
    *,
    global_defaults: Dict[str, Any],
    experiment_meta: Dict[str, Any],
    type_defaults: Dict[str, Dict[str, Any]],
    file_row: pd.Series,
) -> Dict[str, Any]:
    """
    Precedence:
    per-file overrides > per-type defaults > per-experiment > global
    """

    dt = file_row["data_type"]
    resolved: Dict[str, Any] = {}

    # 1) global
    resolved.update({k: v for k, v in global_defaults.items() if v not in ("", None)})

    # 2) experiment
    resolved.update({k: v for k, v in experiment_meta.items() if v not in ("", None)})

    # 3) per-type
    td = type_defaults.get(dt, {})
    resolved.update({k: v for k, v in td.items() if v not in ("", None)})

    # 4) derived per-file (highest)
    resolved.update(
        {
            "path": file_row["full_path"],
            "relative_folder": file_row["relative_folder"],
            "file_name": file_row["file_name"],
            "ext": file_row["ext"],
            "data_type": dt,
            "field_of_view": _none_if_blank(file_row.get("field_of_view", "")),
        }
    )
    # raw file overrides:
    # ------------------------------------------
    # add later if needed 
    # ------------------------------------------
    # Tracking overrides
    if file_row.get("ov_threshold") is not None:
        resolved["threshold"] = float(file_row["ov_threshold"])
    if file_row.get("ov_linking_distance") is not None:
        resolved["linking_distance"] = float(file_row["ov_linking_distance"])
    if file_row.get("ov_gap_closing_distance") is not None:
        resolved["gap_closing_distance"] = float(file_row["ov_gap_closing_distance"])
    if file_row.get("ov_max_frame_gap") is not None:
        resolved["max_frame_gap"] = int(file_row["ov_max_frame_gap"])

    # Mask overrides
    if _none_if_blank(file_row.get("ov_segmentation_method", "")):
        resolved["segmentation_method"] = _clean_str(file_row["ov_segmentation_method"])
    if _none_if_blank(file_row.get("ov_segmentation_parameters", "")):
        resolved["segmentation_parameters"] = _clean_str(file_row["ov_segmentation_parameters"])
    if _none_if_blank(file_row.get("ov_mask_type", "")):
        resolved["mask_type"] = _clean_str(file_row["ov_mask_type"])


    return resolved


def build_manifest(*,
    global_defaults: dict,
    experiment_meta: dict,
    type_defaults: Dict[str, dict],
    df: pd.DataFrame,
) -> dict:
    files_raw = []
    files_resolved = []

    for _, r in df.iterrows():
        if r["data_type"] == "ignore":
            continue
        files_raw.append(
            {
                "path": r["full_path"],
                "relative_folder": r["relative_folder"],
                "file_name": r["file_name"],
                "field_of_view": _none_if_blank(r.get("field_of_view", "")),
                "ext": r["ext"],
                "data_type": r["data_type"],
                "overrides": {
                    "ov_threshold": r.get("ov_threshold"),
                    "ov_linking_distance": r.get("ov_linking_distance"),
                    "ov_gap_closing_distance": r.get("ov_gap_closing_distance"),
                    "ov_max_frame_gap": r.get("ov_max_frame_gap"),
                    "ov_segmentation_method": _none_if_blank(r.get("ov_segmentation_method", "")),
                    "ov_segmentation_parameters": _none_if_blank(r.get("ov_segmentation_parameters", "")),
                    "ov_mask_type": _none_if_blank(r.get("ov_mask_type", "")),},
            }
        )
        files_resolved.append(
            resolve_file_metadata(
                global_defaults=global_defaults,
                experiment_meta=experiment_meta,
                type_defaults=type_defaults,
                file_row=r,
            )
        )

    return {
        "global_defaults": global_defaults,
        "experiment": experiment_meta,
        "type_defaults": type_defaults,
        "files_raw": files_raw,
        "files_resolved": files_resolved,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Experiment Intake", layout="wide")
st.title("Experiment Intake")
st.markdown('<p style="font-size:16px;">Scan a folder and any subfolders, classify files (raw/mask/tracking/analysis), and export a clean manifest for DB insertion.</p>', unsafe_allow_html=True)


with st.expander("Supported file extensions", expanded=False):
    
    st.write(", ".join(sorted(SUPPORTED_EXTS)))

# ------------------------------------
# --- Step 1: Select folder ---
# ------------------------------------
st.header("1) Select experiment folder ")
folder_path = st.text_input(
    "Experiment folder path (local or mounted drive)",
    placeholder="/Users/masoomeshafiee/Desktop/test/masks",
)

scan = st.button("Scan folder", type="primary", disabled=not folder_path.strip())

if scan:
    root = Path(folder_path.strip())
    if not root.exists() or not root.is_dir():
        st.error("Folder path does not exist or is not a directory.")
        st.stop()

    df = scan_experiment_folder(root)
    st.session_state["intake_root"] = str(root.resolve())
    st.session_state["intake_df"] = df

# Load existing scan from session
df: pd.DataFrame = st.session_state.get("intake_df", pd.DataFrame())
root_str: Optional[str] = st.session_state.get("intake_root")

if df.empty:
    st.info("Scan a folder to begin.")
    st.stop()

st.success(f"Found {len(df)} file(s) under: {root_str}")

# ----------------------------------------------
# --- Step 2: Classification rules ---
# ----------------------------------------------
st.header("2) Classify files in bulk")

colA, colB = st.columns(2)

with colA:
    st.subheader("Assign by folder")
    st.caption("Selecting **.** assigns the chosen data type to all files in the current directory. "
    "For files outside this directory, the app attempts to infer the data type "
    "using config-based type hints and filenames. If it cannot determine the type, "
    "the file remains unassigned for manual review.")

    folders = sorted(df["relative_folder"].unique().tolist())
    selected_folder = st.selectbox("Folder", folders)
    folder_role = st.selectbox("Set data type for the file in the selected folder", ROLE_OPTIONS, index=ROLE_OPTIONS.index("raw") if "raw" in ROLE_OPTIONS else 0)
    if st.button("Apply folder rule"):
        df = apply_folder_rule(df, selected_folder, folder_role)
        st.session_state["intake_df"] = df

with colB:
    st.subheader("Assign by extension")
    st.caption("Assigns the selected data type to all files with the chosen extension "
    "in the specified directory and its subdirectories. "
    "For other files, the app attempts to infer the data type using "
    "config-based type hints and filenames. If it cannot determine the type, "
    "the file remains unassigned for manual review.")

    exts = sorted(df["ext"].unique().tolist())
    selected_ext = st.selectbox("Extension", exts)
    ext_role = st.selectbox("Set data type for this extension", ROLE_OPTIONS, index=ROLE_OPTIONS.index("tracking") if "tracking" in ROLE_OPTIONS else 0)
    if st.button("Apply extension rule"):
        df = apply_ext_rule(df, selected_ext, ext_role)
        st.session_state["intake_df"] = df

st.divider()

# --------------------------------------------
# --- Step 3: Per-file overrides ---
# --------------------------------------------
st.header("3) Review & override (only if needed)")
st.markdown(
    '<p style="font-size:16px;">Edit <b>data_type</b>. Use <b>Advanced</b> only if you need per-file overrides '
    '(e.g., a specific tracking threshold for one file).</p>',
    unsafe_allow_html=True,
)
# Filters
fcol1, fcol2, fcol3 = st.columns([1.2, 1.2, 1.6])

with fcol1:
    show_only_unassigned = st.checkbox("Show only unassigned files", value=True)

with fcol2:
    # Filter by data_type (role)
    dt_values = ["All"] + [x for x in ROLE_OPTIONS if x not in ("unassigned", "ignore")]
    filter_data_type = st.selectbox("Filter by data type", dt_values, index=0)

with fcol3:
    advanced = st.checkbox("Advanced: per-file overrides", value=False)

# Build view dataframe
view_df = df.copy()

if show_only_unassigned:
    view_df = view_df[view_df["data_type"] == "unassigned"]

if filter_data_type != "All":
    view_df = view_df[view_df["data_type"] == filter_data_type]


# number of rows found after filtering:
st.markdown(f"Number of records found based on the filtering criteria: `{len(view_df.index)}`")


# Base columns always visible/editable
base_cols = ["relative_folder", "file_name", "field_of_view", "ext", "data_type", "full_path"]

# Override columns depend on filter + advanced toggle
tracking_override_cols = ["ov_threshold", "ov_linking_distance", "ov_gap_closing_distance", "ov_max_frame_gap"]
mask_override_cols = ["ov_segmentation_method", "ov_segmentation_parameters", "ov_mask_type"]

override_cols = []
if advanced:
    if filter_data_type == "tracking":
        override_cols = tracking_override_cols
    elif filter_data_type == "mask":
        override_cols = mask_override_cols
    elif filter_data_type == "All":
        # If they really want advanced for all files, show both sets.
        # (Most users won’t do this; but it keeps power-users happy.)
        override_cols = tracking_override_cols + mask_override_cols
    else:
        # For analysis/raw/plot/etc there are no defined per-file overrides in v1
        override_cols = []

cols_to_show = base_cols + override_cols

st.write("Tip: aim for 0 unassigned files.")
st.caption("If Advanced overrides are enabled, make sure you filtered to tracking or mask to see only relevant fields.")

edited_df = st.data_editor(
    view_df[cols_to_show],
    use_container_width=True,
    hide_index=True,
    disabled=["relative_folder", "file_name", "field_of_view", "ext", "full_path"],
    column_config={
        "data_type": st.column_config.SelectboxColumn("data_type", options=ROLE_OPTIONS),
        # You can optionally add column configs for numeric fields:
        # "ov_threshold": st.column_config.NumberColumn("ov_threshold", step=0.1),
    },
    key="role_editor",
)

# Apply edits back to df.
# Important: data_editor edits are relative to view_df indices, so we update by index.
# Because view_df is a filtered view of df, indices must stay aligned with df indices.
# This works because we didn't reset_index() on view_df.
# if "role_editor" in st.session_state and isinstance(st.session_state["role_editor"], dict):
#     edited = st.session_state["role_editor"].get("edited_rows", {})
#     if edited:
#         for idx_str, changes in edited.items():
#             idx = int(idx_str)

#             if "data_type" in changes:
#                 df.loc[idx, "data_type"] = changes["data_type"]

#             # Apply only visible override fields
#             for k in override_cols:
#                 if k in changes:
#                     df.loc[idx, k] = changes[k]

#         st.session_state["intake_df"] = df
for col in (["data_type"] + override_cols):
    if col in edited_df.columns:
        df.loc[edited_df.index, col] = edited_df[col]

st.divider()

# ----------------------------------------------
# --- Step 4: Metadata defaults & scope ---
# ----------------------------------------------

st.header("4) Optional Metadata Defaults (Can be modified later in the workflow)")

scope = st.radio(
    "Which metadata level are you focusing on right now?",
    options=["Global defaults", "Experiment-level metadata", "Per-type defaults"],
    horizontal=True,
)


# ---- Global defaults form ----

if scope == "Global defaults":
    with st.form("global_defaults_form", clear_on_submit=False):
        st.subheader("Global defaults (optional)")
        c1, c2, c3 = st.columns(3)

        with c1:
            user_name = st.text_input("user_name")
            user_last_name = st.text_input("user_last_name")
            user_email = st.text_input("user_email")

        with c2:
            fluorescent_dye = st.text_input("fluorescent_dye")
            dye_concentration_value = st.number_input("dye_concentration_value", value=0.0, step=0.1)
            dye_concentration_unit = st.selectbox("dye_concentration_unit", DYE_CONCENTRATION_UNITS, index=DYE_CONCENTRATION_UNITS.index("nM") if "nM" in DYE_CONCENTRATION_UNITS else 0)
            

        with c3:
            objective_magnification = st.number_input("objective_magnification", value=0.0, step=1.0)
            laser_wavelength = st.number_input("laser_wavelength", value=0.0, step=1.0)
            laser_intensity = st.number_input("laser_intensity", value=0.0, step=1.0)
            camera_binning = st.number_input("camera_binning", value=1, step=1)
            pixel_size = st.number_input("pixel_size", value=0.0, step=0.01)

        submitted = st.form_submit_button("Save global metadata defaults")
        if submitted:
            st.session_state["defaults_global"] = {
                "user_name": _clean_str(user_name),
                "user_last_name": _clean_str(user_last_name),
                "user_email": _clean_str(user_email),
                "fluorescent_dye": _clean_str(fluorescent_dye),
                "dye_concentration_value": _none_if_zero_float(float(dye_concentration_value)),
                "dye_concentration_unit": _clean_str(dye_concentration_unit),
                "objective_magnification": _none_if_zero_float(float(objective_magnification)),
                "laser_wavelength": _none_if_zero_float(float(laser_wavelength)),
                "laser_intensity": _none_if_zero_float(float(laser_intensity)),
                "camera_binning": _none_if_zero_int(int(camera_binning)),
                "pixel_size": _none_if_zero_float(float(pixel_size)),
            }
            st.success("Saved global defaults for this intake session.")
# ---- Experiment-level metadata + override-global expander ----
if scope == "Experiment-level metadata":
    g = st.session_state.get("defaults_global", {})

    # Prefill from global defaults (but we will store only overrides if toggle enabled)
    g_fluo = g.get("fluorescent_dye", "")
    g_dye_val = g.get("dye_concentration_value", 0.0) or 0.0
    g_dye_unit = g.get("dye_concentration_unit", "")
    g_obj = g.get("objective_magnification", 0.0) or 0.0
    g_wl = g.get("laser_wavelength", 0.0) or 0.0
    g_int = g.get("laser_intensity", 0.0) or 0.0
    g_bin = g.get("camera_binning", 0) or 0
    g_px = g.get("pixel_size", 0.0) or 0.0

    with st.form("exp_meta_form", clear_on_submit=False):
        st.subheader("Experiment-level metadata (applies to all files in this intake)")
        c1, c2, c3 = st.columns(3)

        with c1:
            date = st.text_input("date (YYYY-MM-DD)", value=datetime.now().date().isoformat())
            replicate = st.number_input("replicate", min_value=1, step=1, value=1)
            is_valid = st.checkbox("is_valid", value=True)

        with c2:
            #organism = st.text_input("organism", value="")
            organism = st.selectbox("organism", ALLOWED_ORGANISMS, index=ALLOWED_ORGANISMS.index("yeast") if "yeast" in ALLOWED_ORGANISMS else 0)
            protein = st.text_input("protein", value="")
            strain = st.text_input("strain", value="")
            

        with c3:
            #capture_type = st.text_input("capture_type (fast/long/confocal…)", value="")
            
            capture_type = st.selectbox("capture_type", CAPTURE_TYPES, index=CAPTURE_TYPES.index("long") if "long" in CAPTURE_TYPES else 0)
            exposure_time = st.number_input("exposure_time (s)", value=0.0, step=0.01)
            time_interval = st.number_input("time_interval (s)", value=0.0, step=0.01)

        st.markdown("**Condition (optional, if untreated, select N/A):**")

        c4, c5, c6 = st.columns(3)
        with c4:
            condition_name = st.text_input("condition_name", value="untreated")

        with c5:
            is_na = st.checkbox("concentration value is not applicaple")
            if is_na:
                concentration_value = 0
            else:
                concentration_value = st.number_input("concentration_value", value=0.0, step=0.1)
        with c6:
            #concentration_unit = st.text_input("concentration_unit", value="")
            concentration_unit = st.selectbox("cconcentration_unit", CONDITION_UNITS, index=CONDITION_UNITS.index("uM") if "uM" in CAPTURE_TYPES else 0)
            

    
        # ---- override global microscopy settings (per experiment) ----

        with st.expander("Microscopy settings (override global defaults for this experiment)", expanded=False):
            enable_override = st.checkbox("Enable override for this experiment", value=False)

            fluorescent_dye = st.text_input("fluorescent_dye", value=g_fluo)
            dye_concentration_value = st.number_input("dye_concentration_value", value=float(g_dye_val), step=0.1)
            dye_concentration_unit = st.text_input("dye_concentration_unit", value=g_dye_unit)

            objective_magnification = st.number_input("objective_magnification (ex.:100)", value=float(g_obj), step=1.0)
            laser_wavelength = st.number_input("laser_wavelength", value=float(g_wl), step=1.0)
            laser_intensity = st.number_input("laser_intensity", value=float(g_int), step=1.0)
            camera_binning = st.number_input("camera_binning", value=int(g_bin), step=1)
            pixel_size = st.number_input("pixel_size", value=float(g_px), step=0.01)

        comment = st.text_area("comment (optional)", height=80)

        submitted = st.form_submit_button("Save experiment metadata")

    if submitted:
        exp_meta: Dict[str, Any] = {
            "date": _clean_str(date),
            "replicate": int(replicate) if replicate else None,

            "organism": _clean_str(organism),
            "protein": _clean_str(protein),
            "strain": _clean_str(strain),
            
            "capture_type": _clean_str(capture_type),
            "exposure_time": _none_if_zero_float(float(exposure_time)),
            "time_interval": _none_if_zero_float(float(time_interval)),

            "condition_name": _clean_str(condition_name),
            "concentration_value": _none_if_zero_float(float(concentration_value)),
            "concentration_unit": _clean_str(concentration_unit),

            "is_valid": bool(is_valid),
            "comment": _none_if_blank(comment),
            "experiment_path": str(root_str) if root_str else "",
        }
        # Only store override fields if toggle enabled AND value differs from global
        if enable_override:
            def maybe_set_override(key: str, val: Any, global_val: Any):
                # treat "" and None as "not set"
                if val in ("", None):
                    return
                if global_val in ("", None) and val not in ("", None):
                    exp_meta[key] = val
                    return
                # numeric comparisons
                if isinstance(val, (int, float)) and isinstance(global_val, (int, float)):
                    if float(val) != float(global_val):
                        exp_meta[key] = val
                    return
                if str(val) != str(global_val):
                    exp_meta[key] = val

            maybe_set_override("fluorescent_dye", _clean_str(fluorescent_dye), g_fluo)
            maybe_set_override("dye_concentration_value", _none_if_zero_float(float(dye_concentration_value)), g.get("dye_concentration_value"))
            maybe_set_override("dye_concentration_unit", _clean_str(dye_concentration_unit), g_dye_unit)
            maybe_set_override("objective_magnification", _none_if_zero_float(float(objective_magnification)), g.get("objective_magnification"))
            maybe_set_override("laser_wavelength", _none_if_zero_float(float(laser_wavelength)), g.get("laser_wavelength"))
            maybe_set_override("laser_intensity", _none_if_zero_float(float(laser_intensity)), g.get("laser_intensity"))
            maybe_set_override("camera_binning", _none_if_zero_int(int(camera_binning)), g.get("camera_binning"))
            maybe_set_override("pixel_size", _none_if_zero_float(float(pixel_size)), g.get("pixel_size"))

        st.session_state["meta_experiment"] = exp_meta
        st.success("Saved experiment metadata for this intake session.")


# ---- Per-type defaults (raw/mask/tracking) ----
if scope == "Per-type defaults":
    st.subheader("Per-type defaults")

    if "defaults_by_type" not in st.session_state:
        st.session_state["defaults_by_type"] = {}

    tabs = st.tabs(["raw", "mask", "tracking"])


    with tabs[0]:
        st.markdown("### Raw files defaults, to be developed")
        # to be developed 
    with tabs[1]:
        st.markdown("### Mask defaults")
        with st.form("mask_defaults_form", clear_on_submit=False):
            segmentation_method = st.text_input("segmentation_method(ex. Cellpose)", value="")
            #mask_type = st.text_input("mask_type (ex.:nuclear, cell)", value="")
            mask_type = st.selectbox("mask_type", MASK_TYPES, index=MASK_TYPES.index("nucleus") if "nucleus" in MASK_TYPES else 0)
            

            st.markdown("**Optional: Upload segmentation parameters JSON file**")
            uploaded = st.file_uploader("Segmentation parameters JSON", type=["json"])

            submitted_mask = st.form_submit_button("Save mask defaults")

        if submitted_mask:
            segmentation_params_json_obj = None
            if uploaded is not None:
                try:
                    segmentation_params_json_obj = json.loads(uploaded.read().decode("utf-8"))
                except Exception as e:
                    st.error(f"Could not parse uploaded JSON: {e}")
                    st.stop()

            st.session_state["defaults_by_type"]["mask"] = {
                "segmentation_method": _clean_str(segmentation_method),
                "mask_type": _clean_str(mask_type),
                "segmentation_parameters": segmentation_params_json_obj
            }
            st.success("Saved mask defaults.")

    with tabs[2]:
        st.markdown("### Tracking defaults")
        with st.form("tracking_defaults_form", clear_on_submit=False):

            threshold = st.number_input("threshold", value=0.0, step=1.0)
            linking_distance = st.number_input("linking_distance", value=0.0, step=1.0)
            gap_closing_distance = st.number_input("gap_closing_distance", value=0.0, step=1.0)
            max_frame_gap = st.number_input("max_frame_gap", value=-1, step=1)

            st.markdown("**Optional: Upload TrackMate settings JSON**")
            uploaded = st.file_uploader("TrackMate JSON", type=["json"])

            submitted_tracking = st.form_submit_button("Save tracking defaults")

        if submitted_tracking:
            trackmate_json_obj = None
            if uploaded is not None:
                try:
                    trackmate_json_obj = json.loads(uploaded.read().decode("utf-8"))
                except Exception as e:
                    st.error(f"Could not parse uploaded JSON: {e}")
                    st.stop()

            st.session_state["defaults_by_type"]["tracking"] = {
                "threshold": _none_if_zero_float(float(threshold)),
                "linking_distance": _none_if_zero_float(float(linking_distance)),
                "gap_closing_distance": _none_if_zero_float(float(gap_closing_distance)),
                "max_frame_gap": None if max_frame_gap == -1 else int(max_frame_gap),
                "trackmate_settings_json": trackmate_json_obj,
                }
            st.success("Saved tracking defaults.")

st.divider()

# ---------------------------------------------
# --- Step 5: Validate + export manifest ---
# ---------------------------------------------

st.header("5) Validate & export manifest")

# 1) Basic classification checks (assignment-level)
issues = validate_manifest_df(df)

defaults_global = st.session_state.get("defaults_global", {})
meta_experiment = st.session_state.get("meta_experiment", {})
defaults_by_type = st.session_state.get("defaults_by_type", {})

if not meta_experiment:
    st.warning("Experiment-level metadata not set (required).")

can_build_manifest = (not issues) and bool(meta_experiment)

# Optional: path existence validation for export
require_path_exists_export = st.checkbox(
    "Require file paths to exist (for validation)",
    value=False,
    help="Turn this on only if Streamlit is running on the same machine/mount as the experiment files.",
)

manifest = None
exp_issues = []
file_issues = []
invalid_files_df = pd.DataFrame()

if can_build_manifest:
    manifest = build_manifest(
        global_defaults=defaults_global,
        experiment_meta=meta_experiment,
        type_defaults=defaults_by_type,
        df=df,
    )

    from data_validation_v2 import validate_manifest
    #import data_validation_v2 as dv

    exp_issues, file_issues = validate_manifest(
        manifest,
        allowed_capture_types=CAPTURE_TYPES,
        allowed_organisms=ALLOWED_ORGANISMS,
        dye_units=DYE_CONCENTRATION_UNITS,
        condition_units=CONDITION_UNITS,
        mask_types=MASK_TYPES,
        supported_exts=SUPPORTED_EXTS,
        require_path_exists=require_path_exists_export,
    )

    if file_issues:
        invalid_files_df = pd.DataFrame([{
            "file_name": x.get("file_name"),
            "data_type": x.get("data_type"),
            "path": x.get("path"),
            "issues": "; ".join(x.get("issues", [])),
        } for x in file_issues])

# --- Show assignment-level issues first ---
if issues:
    st.error("Fix these file classification issues:")
    for x in issues:
        st.write(f"- {x}")
else:
    st.success("File classification looks good.")

# --- Show experiment-level blockers ---
if can_build_manifest and exp_issues:
    
    st.error("Experiment-level validation failed (must fix before export/insert):")
    for x in exp_issues:
        st.write(f"- {x}")
else:
        st.success("Experiment-level metadata validated.")

# --- Show file-level invalids (soft blockers) ---
if can_build_manifest and (not invalid_files_df.empty):
    with st.expander(f"File rows with validation issues ({len(invalid_files_df)})"):
        st.dataframe(invalid_files_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download invalid file rows (CSV)",
            data=invalid_files_df.to_csv(index=False),
            file_name="manifest_invalid_file_rows.csv",
            mime="text/csv",
        )
st.subheader("Summary")

col1, col2, col3 = st.columns(3)

# --- File type distribution ---
with col1:
    st.markdown("**File types**")
    st.write(df["data_type"].value_counts(dropna=False))

# --- Validation summary ---
with col2:
    st.markdown("**Validation overview**")
    st.write({
        "experiment_issues": len(exp_issues),
        "file_issues": len(file_issues),
    })
can_export = bool(manifest) and (not issues) and (not exp_issues) and (not file_issues)
# --- Export readiness ---
with col3:
    st.markdown("**Status**")
    if can_export:
        st.success("Ready to export")
    else:
        st.warning("Not ready")

if manifest:
    with st.expander("Preview resolved manifest "):
        resolved_df = pd.json_normalize(manifest["files_resolved"])
        # also show the validation status for each file in the preview
        if file_issues:
            issues_by_path = {x.get("path"): x.get("issues", []) for x in file_issues}
            resolved_df["validation_issues"] = resolved_df["path"].map(issues_by_path).apply(lambda x: "; ".join(x) if isinstance(x, list) else "")
        st.dataframe(resolved_df, use_container_width=True, hide_index=True)
# Export allowed if:
# - classification ok
# - experiment metadata ok
# - experiment-level validation ok
# (file-level issues do NOT block export)


if can_export:
    st.success("Manifest is valid to export.")

    manifest_json = json.dumps(manifest, indent=2, default=str)
    st.download_button(
        "Download resolved manifest.json",
        data=manifest_json,
        file_name="experiment_manifest_resolved.json",
        mime="application/json",
        type="primary",
    )

    #resolved_df = pd.json_normalize(manifest["files_resolved"])
    st.download_button(
        "Download resolved_files.csv",
        data=resolved_df.to_csv(index=False),
        file_name="experiment_files_resolved.csv",
        mime="text/csv",
    )
else:
    st.info("Fix the issues above to enable export.")

    
# -----------------------------------------------------
# ------ Step 6) Insert into database --------
# -----------------------------------------------------

st.header("6) Insert into database (optional)")

# --- DB path ---
DB_PATH = st.secrets.get("DB_PATH", "")

db_path_input = st.text_input(
    "SQLite DB path",
    value=DB_PATH,
    placeholder="/path/to/your.db",
    help="Path to the SQLite database file on the machine running Streamlit.",
)

# ------------- Select manifest source -----------------
source = st.radio(
    "Manifest source",
    options=["Use current session manifest", "Upload manifest.json"],
    horizontal=True,
)
manifest_candidate = None
if source == "Use current session manifest":
    if can_export:
        manifest_candidate = manifest
        st.success("Using the current session manifest.")
    else:
        st.warning("Complete Step 5 first (no valid manifest in session).")
else:
    uploaded_manifest = st.file_uploader("Upload experiment_manifest_resolved.json", type=["json"])
    if uploaded_manifest is not None:
        try:
            manifest_candidate = json.loads(uploaded_manifest.read().decode("utf-8"))
            st.success("Manifest uploaded and parsed successfully.")
        except Exception as e:
            st.error(f"Could not parse uploaded manifest JSON: {e}")
            st.stop()

# ------------- Select insertion mode -----------------
# “Strict (skip duplicates)”
# “Upsert (update duplicates)” (later)

mode = st.selectbox(
    "Duplicate handling mode",
    options=["strict", "upsert"],
    index=0,
    help=(
        "strict: do not modify existing rows; duplicates are skipped and reported.\n upsert: update existing rows when the same file already exists (based on the defined uniqueness rules)."
    ),
)
allow_partial = st.checkbox("Allow partial insert (skip invalid files)", value=True)
require_path_exists = st.checkbox("Require file paths to exist on server", value=True)


colA, colB = st.columns([1, 1])
with colA:
    insert_disabled = (not db_path_input.strip()) or (manifest_candidate is None)
    if st.button("Insert into DB", type="primary", disabled=insert_disabled):
        try:
            from insert import insert_manifest  # adjust import if needed

            report = insert_manifest(
                manifest_candidate,
                db_path_input.strip(),
                allow_partial_files=allow_partial,
                require_path_exists=require_path_exists,
                duplicate_mode=mode,
            )
            st.session_state["last_insert_report"] = report

        except Exception as e:
            st.error(f"DB insertion failed: {e}")
            st.stop()

with colB:
    if st.button("Clear last report"):
        st.session_state.pop("last_insert_report", None)

report = st.session_state.get("last_insert_report")

if report:
    status = report.get("status")
    if status == "ok":
        st.success(f"Inserted. Experiment ID: {report.get('experiment_id')}")
    elif status == "failed_validation":
        st.error("Validation failed. Nothing was inserted.")
    else:
        st.error(f"Insertion failed: {status}")

counts = report.get("counts", {}) or {}
inserted_counts = counts.get("inserted", {}) or {}
updated_counts = counts.get("updated", {}) or {}

st.subheader("Inserted counts")
st.dataframe(pd.DataFrame([inserted_counts]), use_container_width=True, hide_index=True)

st.subheader("Updated counts")
st.dataframe(pd.DataFrame([updated_counts]), use_container_width=True, hide_index=True)

# If validation failed, show reasons
if status == "failed_validation":
    v = report.get("validation", {}) or {}
    exp_issues = v.get("experiment_issues", []) or []
    if exp_issues:
        st.subheader("Validation issues")
        for issue in exp_issues:
            st.write(f"- {issue}")

skipped = report.get("skipped", []) or []
if skipped:
    with st.expander(f"Skipped rows ({len(skipped)})"):
        rows = []
        for s in skipped:
            ctx = s.get("context", {}) or {}
            rows.append({
                "file_name": ctx.get("file_name"),
                "data_type": ctx.get("data_type"),
                "reason": s.get("reason"),
                "existing_id": s.get("existing_id"),
                "table": s.get("table"),
                "path": ctx.get("path"),
            })
        skipped_df = pd.DataFrame(rows)
        st.dataframe(skipped_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download skipped rows (CSV)",
            data=skipped_df.to_csv(index=False),
            file_name="db_skipped_rows.csv",
            mime="text/csv",
        )
else:
    st.caption("No skipped rows reported.")

report_json = json.dumps(report, indent=2, default=str)
st.download_button(
    "Download insert report (JSON)",
    data=report_json,
    file_name="db_insert_report.json",
    mime="application/json",
)
