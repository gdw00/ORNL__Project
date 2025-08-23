#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from config import Config
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import netCDF4 as nc


def build_restart_kdtree(ds_restart: nc.Dataset) -> Tuple[cKDTree, np.ndarray]:
    """Build a KDTree and coordinates array from restart file grid cells."""
    gridcell_lat = ds_restart.variables["grid1d_lat"][:]
    gridcell_lon = ds_restart.variables["grid1d_lon"][:]
    coords = np.vstack((gridcell_lat, gridcell_lon)).T
    tree = cKDTree(coords)
    return tree, coords


def build_column_index_map(ds_restart: nc.Dataset) -> Dict[int, np.ndarray]:
    """Build a mapping from grid cell ID to column indices."""
    cols1d_gridcell_index = ds_restart.variables["cols1d_gridcell_index"][:]
    unique_ids = np.unique(cols1d_gridcell_index)
    mapping: Dict[int, np.ndarray] = {}
    for grid_id in unique_ids:
        mapping[int(grid_id)] = np.where(cols1d_gridcell_index == grid_id)[0]
    return mapping


def ensure_vars_exist(ds: nc.Dataset, var_names: List[str]) -> List[str]:
    """Filter a list of variable names, keeping only those that exist in the dataset."""
    existing = []
    for name in var_names:
        if name in ds.variables:
            existing.append(name)
    return existing


def build_pft_index_map(ds_restart: nc.Dataset) -> Dict[int, np.ndarray]:
    """Build a mapping from grid cell ID to PFT indices."""
    pfts1d_gridcell_index = ds_restart.variables["pfts1d_gridcell_index"][:]
    unique_ids = np.unique(pfts1d_gridcell_index)
    mapping: Dict[int, np.ndarray] = {}
    for grid_id in unique_ids:
        mapping[int(grid_id)] = np.where(pfts1d_gridcell_index == grid_id)[0]
    return mapping


def extract_col1d_x(ds_restart: nc.Dataset, var_name: str, col_indices: np.ndarray) -> List[float]:
    """Extract 1D column-level values (X) from the restart file."""
    if col_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][col_indices]
    return values.astype(float).tolist()


def extract_col1d_y(ds_r_list: List[nc.Dataset], var_name: str, col_indices: np.ndarray) -> List[float]:
    """Extract and average 1D column-level values (Y) from a list of files."""
    if col_indices.size == 0:
        return []
    slices: List[np.ndarray] = []
    for ds_r in ds_r_list:
        values = ds_r.variables[var_name][col_indices]
        slices.append(np.asarray(values, dtype=float))
    stacked = np.stack(slices, axis=0)  # (n_files, n_cols_for_cell)
    avg = np.mean(stacked, axis=0)
    return avg.tolist()


def extract_col2d_x(ds_restart: nc.Dataset, var_name: str, col_indices: np.ndarray) -> List[List[float]]:
    """Extract 2D column-level values (X) from the restart file."""
    if col_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][col_indices, :]
    return np.asarray(values, dtype=float).tolist()


def extract_col2d_y(ds_r_list: List[nc.Dataset], var_name: str, col_indices: np.ndarray) -> List[List[float]]:
    """Extract and average 2D column-level values (Y) from a list of files."""
    if col_indices.size == 0:
        return []
    slices: List[np.ndarray] = []
    for ds_r in ds_r_list:
        values = ds_r.variables[var_name][col_indices, :]
        slices.append(np.asarray(values, dtype=float))
    stacked = np.stack(slices, axis=0)  # (n_files, n_cols_for_cell, n_layers)
    avg = np.mean(stacked, axis=0)
    return avg.tolist()


def extract_pft1d_x(ds_restart: nc.Dataset, var_name: str, pft_indices: np.ndarray) -> List[float]:
    """Extract 1D PFT-level values (X) from the restart file."""
    if pft_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][pft_indices]
    return np.asarray(values, dtype=float).tolist()


def extract_pft1d_y(ds_r_list: List[nc.Dataset], var_name: str, pft_indices: np.ndarray) -> List[float]:
    """Extract and average 1D PFT-level values (Y) from a list of files."""
    if pft_indices.size == 0:
        return []
    slices: List[np.ndarray] = []
    for ds_r in ds_r_list:
        values = ds_r.variables[var_name][pft_indices]
        slices.append(np.asarray(values, dtype=float))
    stacked = np.stack(slices, axis=0)  # (n_files, n_pfts_for_cell)
    avg = np.mean(stacked, axis=0)
    return avg.tolist()


def extract_pft2d_x(ds_restart: nc.Dataset, var_name: str, pft_indices: np.ndarray) -> List[List[float]]:
    """Extract 2D PFT-level values (X) from the restart file."""
    if pft_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][pft_indices, :]
    return np.asarray(values, dtype=float).tolist()


def extract_pft2d_y(ds_r_list: List[nc.Dataset], var_name: str, pft_indices: np.ndarray) -> List[List[float]]:
    """Extract and average 2D PFT-level values (Y) from a list of files."""
    if pft_indices.size == 0:
        return []
    slices: List[np.ndarray] = []
    for ds_r in ds_r_list:
        values = ds_r.variables[var_name][pft_indices, :]
        slices.append(np.asarray(values, dtype=float))
    stacked = np.stack(slices, axis=0)  # (n_files, n_pfts_for_cell, n_layers)
    avg = np.mean(stacked, axis=0)
    return avg.tolist()


def augment_dataframe_with_pools(
    df: pd.DataFrame,
    ds_restart: nc.Dataset,
    ds_r_list: List[nc.Dataset],
    restart_tree: cKDTree,
    restart_coords: np.ndarray,
    col_index_map: Dict[int, np.ndarray],
    pool_vars: List[str],
) -> pd.DataFrame:
    """
    Augments the DataFrame with pool variables from restart and Y files.
    """
    # Filter out data rows without coordinates
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("DataFrame is missing Latitude/Longitude columns and cannot be mapped to the restart grid.")

    # Only keep variables that exist in the dataset
    pool_vars_existing = ensure_vars_exist(ds_restart, pool_vars)
    if not pool_vars_existing:
        raise ValueError(f"No target variables found in the restart file: {pool_vars}")

    # Also require that these variables exist in the Y-side files
    pool_vars_final: List[str] = []
    for v in pool_vars_existing:
        if all(v in ds_r.variables for ds_r in ds_r_list):
            pool_vars_final.append(v)
        else:
            print(f"[Warning] Variable {v} is missing in some Y files, skipping this variable.")

    if not pool_vars_final:
        raise ValueError("Target variables do not exist in the Y file group.")

    latitudes = df["Latitude"].to_numpy()
    longitudes = df["Longitude"].to_numpy()
    query_coords = np.vstack((latitudes, longitudes)).T

    # Nearest neighbor grid cell (restart grid) index
    _, nearest_restart_indices = restart_tree.query(query_coords, k=1)

    # Append four variables and their Y_ counterparts for each observation row
    results_x: Dict[str, List[List[float]]] = {v: [] for v in pool_vars_final}
    results_y: Dict[str, List[List[float]]] = {f"Y_{v}": [] for v in pool_vars_final}

    for row_idx, restart_idx in enumerate(nearest_restart_indices):
        gridcell_id = int(restart_idx) + 1  # Index in the file usually starts from 1
        col_indices = col_index_map.get(gridcell_id, np.array([], dtype=int))

        for v in pool_vars_final:
            x_vals = extract_col1d_x(ds_restart, v, col_indices)
            y_vals = extract_col1d_y(ds_r_list, v, col_indices)

            results_x[v].append(x_vals)
            results_y[f"Y_{v}"].append(y_vals)

        if (row_idx + 1) % 1000 == 0:
            print(f"   Processed {row_idx + 1} / {len(df)} rows ...")

    # Write to DataFrame
    for v in pool_vars_final:
        df[v] = results_x[v]
        df[f"Y_{v}"] = results_y[f"Y_{v}"]

    return df


def augment_dataframe_with_vars(
    df: pd.DataFrame,
    ds_restart: nc.Dataset,
    ds_special_p_restart: nc.Dataset,
    ds_r_list: List[nc.Dataset],
    restart_tree: cKDTree,
    restart_coords: np.ndarray,
    col_index_map: Dict[int, np.ndarray],
    pft_index_map: Dict[int, np.ndarray],
    vars_1d: List[str],
    vars_2d: List[str],
    special_p_vars: List[str],
) -> pd.DataFrame:
    """
    Appends column-level 1D/2D variables from restart and Y files (names from config/txt) to the DataFrame.
    - For 1D variables: extracts values of shape [n_cols]; for Y: averages over file dimension to get [n_cols]
    - For 2D variables: extracts values of shape [n_cols, n_layers]; for Y: averages over file dimension to get [n_cols, n_layers]
    Returns the augmented DataFrame.
    """
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("DataFrame is missing Latitude/Longitude columns and cannot be mapped to the restart grid.")

    # Filter for variable existence (requires existence in restart and all Y files)
    vars_1d_existing = ensure_vars_exist(ds_restart, vars_1d)
    vars_2d_existing = ensure_vars_exist(ds_restart, vars_2d)

    final_1d: List[str] = [v for v in vars_1d_existing if all(v in ds_r.variables for ds_r in ds_r_list)]
    final_2d: List[str] = [v for v in vars_2d_existing if all(v in ds_r.variables for ds_r in ds_r_list)]

    if not final_1d and not final_2d:
        print("Warning: Target 1D/2D variables do not exist in the restart or Y file group.")
        return df

    # ========================== New print logic starts ==========================
    print("\n" + "="*70)
    print("🔬 Variable Classification Report (based on NetCDF dimension auto-detection)")
    print("="*70)

    classified_pft_1d, classified_col_1d = [], []
    classified_pft_2d, classified_col_2d = [], []

    # Pre-scan and classify 1D variables
    for v in final_1d:
        var_obj = ds_restart.variables[v]
        if "pft" in tuple(var_obj.dimensions):
            classified_pft_1d.append(v)
        else:
            classified_col_1d.append(v)

    # Pre-scan and classify 2D variables
    for v in final_2d:
        ds_for_x = ds_special_p_restart if v in special_p_vars else ds_restart
        var_obj = ds_for_x.variables[v]
        if "pft" in tuple(var_obj.dimensions):
            classified_pft_2d.append(v)
        else:
            classified_col_2d.append(v)

    # Print formatted report
    print("▶️   Identified as [PFT Variables] (1D):")
    print(f"    {classified_pft_1d if classified_pft_1d else '(None)'}")

    print("\n▶️   Identified as [Column Variables] (1D):")
    print(f"    {classified_col_1d if classified_col_1d else '(None)'}")

    print("\n▶️   Identified as [PFT Variables] (2D):")
    print(f"    {classified_pft_2d if classified_pft_2d else '(None)'}")

    print("\n▶️   Identified as [Column Variables] (2D):")
    print(f"    {classified_col_2d if classified_col_2d else '(None)'}")

    print("="*70 + "\n")
    # ========================== New print logic ends ==========================

    final_1d: List[str] = []
    for v in vars_1d_existing:
        if all(v in ds_r.variables for ds_r in ds_r_list):
            final_1d.append(v)
        else:
            print(f"[Warning] Variable {v} is missing in some Y files, skipping this variable.")

    final_2d: List[str] = []
    for v in vars_2d_existing:
        if all(v in ds_r.variables for ds_r in ds_r_list):
            final_2d.append(v)
        else:
            print(f"[Warning] Variable {v} is missing in some Y files, skipping this variable.")

    if not final_1d and not final_2d:
        print("Warning: Target 1D/2D variables do not exist in the restart or Y file group.")
        return df

    print("\n" + "="*50)
    print(f"[Confirm] Augmenting variables for batch data. Below is the source file confirmation for X values:")
    
    default_x_source_file = ds_restart.filepath()
    special_x_source_file = ds_special_p_restart.filepath()
    
    if final_2d:
        print("\n--- 2D Variable Source ---")
        for var_name in sorted(final_2d):
            if var_name in special_p_vars:
                print(f"  -> ⭐ Variable '{var_name}': Will be read from special file \n\t    '{special_x_source_file}'")
            else:
                print(f"  ->   Variable '{var_name}': Will be read from default file \n\t    '{default_x_source_file}'")
    
    if final_1d:
        print("\n--- 1D Variable Source ---")
        for var_name in sorted(final_1d):
            print(f"  ->   Variable '{var_name}': Will be read from default file \n\t    '{default_x_source_file}'")

    print("="*50 + "\n")

    latitudes = df["Latitude"].to_numpy()
    longitudes = df["Longitude"].to_numpy()
    query_coords = np.vstack((latitudes, longitudes)).T

    # Nearest neighbor grid cell (restart grid) index
    _, nearest_restart_indices = restart_tree.query(query_coords, k=1)

    # Result containers
    results_x_1d: Dict[str, List[List[float]]] = {v: [] for v in final_1d}
    results_y_1d: Dict[str, List[List[float]]] = {f"Y_{v}": [] for v in final_1d}
    results_x_2d: Dict[str, List[List[List[float]]]] = {v: [] for v in final_2d}
    results_y_2d: Dict[str, List[List[List[float]]]] = {f"Y_{v}": [] for v in final_2d}

    for row_idx, restart_idx in enumerate(nearest_restart_indices):
        gridcell_id = int(restart_idx) + 1  # Index in the file usually starts from 1
        col_indices = col_index_map.get(gridcell_id, np.array([], dtype=int))
        pft_indices = pft_index_map.get(gridcell_id, np.array([], dtype=int))

        for v in final_1d:
            var_obj = ds_restart.variables[v]
            dims = tuple(var_obj.dimensions)
            if "pft" in dims:
                x_vals = extract_pft1d_x(ds_restart, v, pft_indices)
                y_vals = extract_pft1d_y(ds_r_list, v, pft_indices)
            else:
                x_vals = extract_col1d_x(ds_restart, v, col_indices)
                y_vals = extract_col1d_y(ds_r_list, v, col_indices)
            results_x_1d[v].append(x_vals)
            results_y_1d[f"Y_{v}"].append(y_vals)

        for v in final_2d:
            # Determine which file to read the X value from based on whether it's a special P variable
            if v in special_p_vars:
                ds_for_x = ds_special_p_restart
            else:
                ds_for_x = ds_restart
            var_obj = ds_for_x.variables[v]  # Use the correct ds object
            dims = tuple(var_obj.dimensions)
            
            if "pft" in dims:
                x_vals_2d = extract_pft2d_x(ds_for_x, v, pft_indices)
                y_vals_2d = extract_pft2d_y(ds_r_list, v, pft_indices)
            else:
                x_vals_2d = extract_col2d_x(ds_for_x, v, col_indices)
                y_vals_2d = extract_col2d_y(ds_r_list, v, col_indices)
            
            results_x_2d[v].append(x_vals_2d)
            results_y_2d[f"Y_{v}"].append(y_vals_2d)

        if (row_idx + 1) % 1000 == 0:
            print(f"   Processed {row_idx + 1} / {len(df)} rows ...")

    # Write to DataFrame
    for v in final_1d:
        df[v] = results_x_1d[v]
        df[f"Y_{v}"] = results_y_1d[f"Y_{v}"]
    for v in final_2d:
        df[v] = results_x_2d[v]
        df[f"Y_{v}"] = results_y_2d[f"Y_{v}"]

    return df


def main():
    parser = argparse.ArgumentParser(
        description=
        "Augments existing batched training data with newly classified RESTART variables and corresponding Y_ columns (variable names from config/txt config)"
    )
    parser.add_argument(
        "--input_glob",
        default=Config.INPUT_GLOB,
        help="Glob path pattern for input pkl batch files (defaults to original data path and pattern)"
    )
    parser.add_argument(
        "--output_dir",
        default=Config.OUTPUT_DIR,
        help="Output directory (defaults to augmented data directory)"
    )

    args = parser.parse_args()

    # Data file paths (hardcoded for fixed paths, no dependency on config for independent run)
    file_path10 = \
        "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/20250117_trendytest_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc"

    file_path17 = \
        "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0701-01-01-00000.nc"
    file_path18 = \
        "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0721-01-01-00000.nc"
    file_path19 = \
        "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0741-01-01-00000.nc"
    file_path20 = \
        "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0761-01-01-00000.nc"
    file_path21 = \
        "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc"

    # Read new classifications from config and map them to three RESTART groups:
    # - RESTART_PFT_VARS        <- dataset_new_1D_PFT_VARIABLES
    # - RESTART_COL_1D_VARS     <- dataset_new_RESTART_COL_1D_VARS
    # - RESTART_COL_2D_VARS     <- dataset_new_Water_variables + dataset_new_2D_VARIABLES
    restart_pft_vars = list(dict.fromkeys(Config.dataset_new_1D_PFT_VARIABLES))
    restart_col_1d_vars = list(dict.fromkeys(Config.dataset_new_RESTART_COL_1D_VARS))
    restart_col_2d_vars = list(dict.fromkeys(list(Config.dataset_new_Water_variables) + list(Config.dataset_new_2D_VARIABLES)))

    # Map to the 1D/2D inputs required by the augmentation function
    configured_vars_1d = list(dict.fromkeys(restart_pft_vars + restart_col_1d_vars))
    configured_vars_2d = list(dict.fromkeys(restart_col_2d_vars + Config.SPECIAL_P_VARS))

    input_files = sorted(glob.glob(args.input_glob))
    if not input_files:
        print(f"No input files found: {args.input_glob}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("Opening restart file and Y file group ...")
    ds_restart = nc.Dataset(file_path10)
    ds_special_p_restart = nc.Dataset(Config.SPECIAL_P_INPUT_NC)
    ds_r_list = [nc.Dataset(fp) for fp in [file_path17, file_path18, file_path19, file_path20, file_path21]]

    try:
        print("Building restart grid KDTree and column index map ...")
        restart_tree, restart_coords = build_restart_kdtree(ds_restart)
        col_index_map = build_column_index_map(ds_restart)
        pft_index_map = build_pft_index_map(ds_restart)

        print(f"Starting to process {len(input_files)} batch files ...")
        for fp in input_files:
            print(f"Processing: {fp}")
            df = pd.read_pickle(fp)

            # 1. Identify variables that need to be forcibly replaced (from Config)
            force_replace_vars = set(Config.SPECIAL_P_VARS)

            # 2. Find other missing variables besides those to be forcibly replaced
            missing_other_vars_1d = [
                v for v in configured_vars_1d
                if v not in force_replace_vars and not (v in df.columns and f"Y_{v}" in df.columns)
            ]
            missing_other_vars_2d = [
                v for v in configured_vars_2d
                if v not in force_replace_vars and not (v in df.columns and f"Y_{v}" in df.columns)
            ]

            # 3. Merge into the final list of variables to process
            #    - 1D variables = other missing 1D variables (since special variables are all 2D)
            #    - 2D variables = other missing 2D variables + all variables to be forcibly replaced
            vars_to_process_1d = missing_other_vars_1d
            # Use set.union and list() to merge and deduplicate
            vars_to_process_2d = list(set(missing_other_vars_2d).union(force_replace_vars))

            # 4. Check if there is nothing to do
            if not vars_to_process_1d and not vars_to_process_2d:
                print("   All variables already exist and no forced replacement is needed, skipping.")
                continue

            # 5. Print the list of variables to be processed this time (new print logic)
            print("   ----------------------------------------")
            if force_replace_vars.intersection(vars_to_process_2d):
                print(f"   Forcing replacement for variables: {list(force_replace_vars.intersection(vars_to_process_2d))}")
            if missing_other_vars_1d:
                print(f"   Variables to be added (1D): {missing_other_vars_1d}")
            if missing_other_vars_2d:
                print(f"   Variables to be added (2D): {missing_other_vars_2d}")
            print("   ----------------------------------------")

            df_aug = augment_dataframe_with_vars(
                df=df,
                ds_restart=ds_restart,
                ds_special_p_restart=ds_special_p_restart,
                ds_r_list=ds_r_list,
                restart_tree=restart_tree,
                restart_coords=restart_coords,
                col_index_map=col_index_map,
                pft_index_map=pft_index_map,
                vars_1d=vars_to_process_1d,
                vars_2d=vars_to_process_2d,
                special_p_vars=Config.SPECIAL_P_VARS
            )

            base_name = os.path.basename(fp)
            # Output naming follows the augmented prefix: enhanced_1_training_data_batch_XX.pkl
            out_name = f"{Config.ENHANCED_PREFIX}{base_name}"
            out_path = os.path.join(args.output_dir, out_name)
            df_aug.to_pickle(out_path)
            print(f"   Saved: {out_path}")
    finally:
        print("Closing all NetCDF files ...")
        ds_restart.close()
        ds_special_p_restart.close()
        for ds in ds_r_list:
            try:
                ds.close()
            except Exception:
                pass

    print("All done.")


if __name__ == "__main__":
    main()