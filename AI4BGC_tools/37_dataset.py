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
    gridcell_lat = ds_restart.variables["grid1d_lat"][:]
    gridcell_lon = ds_restart.variables["grid1d_lon"][:]
    coords = np.vstack((gridcell_lat, gridcell_lon)).T
    tree = cKDTree(coords)
    return tree, coords


def build_column_index_map(ds_restart: nc.Dataset) -> Dict[int, np.ndarray]:
    cols1d_gridcell_index = ds_restart.variables["cols1d_gridcell_index"][:]
    unique_ids = np.unique(cols1d_gridcell_index)
    mapping: Dict[int, np.ndarray] = {}
    for grid_id in unique_ids:
        mapping[int(grid_id)] = np.where(cols1d_gridcell_index == grid_id)[0]
    return mapping


def ensure_vars_exist(ds: nc.Dataset, var_names: List[str]) -> List[str]:
    existing = []
    for name in var_names:
        if name in ds.variables:
            existing.append(name)
    return existing


def build_pft_index_map(ds_restart: nc.Dataset) -> Dict[int, np.ndarray]:
    pfts1d_gridcell_index = ds_restart.variables["pfts1d_gridcell_index"][:]
    unique_ids = np.unique(pfts1d_gridcell_index)
    mapping: Dict[int, np.ndarray] = {}
    for grid_id in unique_ids:
        mapping[int(grid_id)] = np.where(pfts1d_gridcell_index == grid_id)[0]
    return mapping


def extract_col1d_x(ds_restart: nc.Dataset, var_name: str, col_indices: np.ndarray) -> List[float]:
    if col_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][col_indices]
    return values.astype(float).tolist()


def extract_col1d_y(ds_r_list: List[nc.Dataset], var_name: str, col_indices: np.ndarray) -> List[float]:
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
    """提取列尺度 2D 变量（形如 [n_cols, n_layers]）的 X 值列表。"""
    if col_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][col_indices, :]
    return np.asarray(values, dtype=float).tolist()


def extract_col2d_y(ds_r_list: List[nc.Dataset], var_name: str, col_indices: np.ndarray) -> List[List[float]]:
    """从多个 Y 文件中提取列尺度 2D 变量并按文件维度求平均，返回 [n_cols, n_layers] 列表。"""
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
    if pft_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][pft_indices]
    return np.asarray(values, dtype=float).tolist()


def extract_pft1d_y(ds_r_list: List[nc.Dataset], var_name: str, pft_indices: np.ndarray) -> List[float]:
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
    if pft_indices.size == 0:
        return []
    values = ds_restart.variables[var_name][pft_indices, :]
    return np.asarray(values, dtype=float).tolist()


def extract_pft2d_y(ds_r_list: List[nc.Dataset], var_name: str, pft_indices: np.ndarray) -> List[List[float]]:
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
    # 过滤掉缺少坐标的数据
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("DataFrame 缺少 Latitude/Longitude 列，无法映射到 restart 网格。")

    # 仅保留存在于数据集中的变量
    pool_vars_existing = ensure_vars_exist(ds_restart, pool_vars)
    if not pool_vars_existing:
        raise ValueError(f"在 restart 文件中未找到任何目标变量: {pool_vars}")

    # 同时要求这些变量在 Y 端文件中也存在
    pool_vars_final: List[str] = []
    for v in pool_vars_existing:
        if all(v in ds_r.variables for ds_r in ds_r_list):
            pool_vars_final.append(v)
        else:
            print(f"[警告] 变量 {v} 在某些 Y 文件中缺失，跳过该变量。")

    if not pool_vars_final:
        raise ValueError("目标变量在 Y 文件组中均不存在。")

    latitudes = df["Latitude"].to_numpy()
    longitudes = df["Longitude"].to_numpy()
    query_coords = np.vstack((latitudes, longitudes)).T

    # 最近邻 gridcell（restart 网格）索引
    _, nearest_restart_indices = restart_tree.query(query_coords, k=1)

    # 为每个观测行追加四个变量及其 Y_
    results_x: Dict[str, List[List[float]]] = {v: [] for v in pool_vars_final}
    results_y: Dict[str, List[List[float]]] = {f"Y_{v}": [] for v in pool_vars_final}

    for row_idx, restart_idx in enumerate(nearest_restart_indices):
        gridcell_id = int(restart_idx) + 1  # 文件中的索引通常从 1 开始
        col_indices = col_index_map.get(gridcell_id, np.array([], dtype=int))

        for v in pool_vars_final:
            x_vals = extract_col1d_x(ds_restart, v, col_indices)
            y_vals = extract_col1d_y(ds_r_list, v, col_indices)

            results_x[v].append(x_vals)
            results_y[f"Y_{v}"].append(y_vals)

        if (row_idx + 1) % 1000 == 0:
            print(f"  已处理 {row_idx + 1} / {len(df)} 行 ...")

    # 写入 DataFrame
    for v in pool_vars_final:
        df[v] = results_x[v]
        df[f"Y_{v}"] = results_y[f"Y_{v}"]

    return df


def augment_dataframe_with_vars(
    df: pd.DataFrame,
    ds_restart: nc.Dataset,
    ds_special_p_restart: nc.Dataset, # <--- ADDED
    ds_r_list: List[nc.Dataset],
    restart_tree: cKDTree,
    restart_coords: np.ndarray,
    col_index_map: Dict[int, np.ndarray],
    pft_index_map: Dict[int, np.ndarray],
    vars_1d: List[str],
    vars_2d: List[str],
    special_p_vars: List[str], # <--- ADDED
    
) -> pd.DataFrame:
    """
    将来自 restart 与 Y 文件组的列尺度 1D/2D 变量（名称来自配置或 txt 解析）追加到 DataFrame。
    - 对 1D 变量：提取形如 [n_cols] 的值；对 Y：在文件维度求平均后得到 [n_cols]
    - 对 2D 变量：提取形如 [n_cols, n_layers] 的值；对 Y：在文件维度求平均后得到 [n_cols, n_layers]
    返回增强后的 DataFrame。
    """
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("DataFrame 缺少 Latitude/Longitude 列，无法映射到 restart 网格。")

    # 变量存在性筛选（要求在 restart 与所有 Y 文件中均存在）
    vars_1d_existing = ensure_vars_exist(ds_restart, vars_1d)
    vars_2d_existing = ensure_vars_exist(ds_restart, vars_2d)
    
    final_1d: List[str] = [v for v in vars_1d_existing if all(v in ds_r.variables for ds_r in ds_r_list)]
    final_2d: List[str] = [v for v in vars_2d_existing if all(v in ds_r.variables for ds_r in ds_r_list)]

    if not final_1d and not final_2d:
        print("警告: 目标 1D/2D 变量在 restart 或 Y 文件组中均不存在。")
        return df
    # ========================== 新增的打印逻辑开始 ==========================
    print("\n" + "="*70)
    print("🔬 变量分类报告 (基于NetCDF维度自动识别)")
    print("="*70)

    classified_pft_1d, classified_col_1d = [], []
    classified_pft_2d, classified_col_2d = [], []

    # 对1D变量进行预扫描和分类
    for v in final_1d:
        var_obj = ds_restart.variables[v]
        if "pft" in tuple(var_obj.dimensions):
            classified_pft_1d.append(v)
        else:
            classified_col_1d.append(v)

    # 对2D变量进行预扫描和分类
    for v in final_2d:
        ds_for_x = ds_special_p_restart if v in special_p_vars else ds_restart
        var_obj = ds_for_x.variables[v]
        if "pft" in tuple(var_obj.dimensions):
            classified_pft_2d.append(v)
        else:
            classified_col_2d.append(v)

    # 打印格式化的报告
    print("▶️  识别为 [PFT 变量] (1D):")
    print(f"   {classified_pft_1d if classified_pft_1d else '(无)'}")

    print("\n▶️  识别为 [Column 变量] (1D):")
    print(f"   {classified_col_1d if classified_col_1d else '(无)'}")

    print("\n▶️  识别为 [PFT 变量] (2D):")
    print(f"   {classified_pft_2d if classified_pft_2d else '(无)'}")

    print("\n▶️  识别为 [Column 变量] (2D):")
    print(f"   {classified_col_2d if classified_col_2d else '(无)'}")

    print("="*70 + "\n")
    # ========================== 新增的打印逻辑结束 ==========================

    final_1d: List[str] = []
    for v in vars_1d_existing:
        if all(v in ds_r.variables for ds_r in ds_r_list):
            final_1d.append(v)
        else:
            print(f"[警告] 变量 {v} 在某些 Y 文件中缺失，跳过该变量。")

    final_2d: List[str] = []
    for v in vars_2d_existing:
        if all(v in ds_r.variables for ds_r in ds_r_list):
            final_2d.append(v)
        else:
            print(f"[警告] 变量 {v} 在某些 Y 文件中缺失，跳过该变量。")

    if not final_1d and not final_2d:
        print("警告: 目标 1D/2D 变量在 restart 或 Y 文件组中均不存在。")
        return df # <--- MODIFIED: 如果没有变量可处理，直接返回原始df
    # ========== 从这里粘贴新增的代码 ==========
    print("\n" + "="*50)
    print(f"[确认] 开始为批次数据增强变量，以下是X值的来源文件确认：")
    
    default_x_source_file = ds_restart.filepath()
    special_x_source_file = ds_special_p_restart.filepath()
    
    if final_2d:
        print("\n--- 2D变量来源 ---")
        for var_name in sorted(final_2d):
            if var_name in special_p_vars:
                print(f"  -> ⭐ 变量 '{var_name}':  将从特殊文件读取 \n\t    '{special_x_source_file}'")
            else:
                print(f"  ->   变量 '{var_name}':  将从默认文件读取 \n\t    '{default_x_source_file}'")
    
    if final_1d:
        print("\n--- 1D变量来源 ---")
        for var_name in sorted(final_1d):
            print(f"  ->   变量 '{var_name}':  将从默认文件读取 \n\t    '{default_x_source_file}'")

    print("="*50 + "\n")
    # ========== 新增代码粘贴结束 ==========
    latitudes = df["Latitude"].to_numpy()
    longitudes = df["Longitude"].to_numpy()
    query_coords = np.vstack((latitudes, longitudes)).T

    # 最近邻 gridcell（restart 网格）索引
    _, nearest_restart_indices = restart_tree.query(query_coords, k=1)

    # 结果容器
    results_x_1d: Dict[str, List[List[float]]] = {v: [] for v in final_1d}
    results_y_1d: Dict[str, List[List[float]]] = {f"Y_{v}": [] for v in final_1d}
    results_x_2d: Dict[str, List[List[List[float]]]] = {v: [] for v in final_2d}
    results_y_2d: Dict[str, List[List[List[float]]]] = {f"Y_{v}": [] for v in final_2d}

    for row_idx, restart_idx in enumerate(nearest_restart_indices):
        gridcell_id = int(restart_idx) + 1  # 文件中的索引通常从 1 开始
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
            # 判断当前变量是否为特殊P变量，以决定从哪个文件读取X值
            if v in special_p_vars:
                ds_for_x = ds_special_p_restart
            else:
                ds_for_x = ds_restart
            var_obj = ds_for_x.variables[v] # 使用正确的 ds 对象
            dims = tuple(var_obj.dimensions)
            
            if "pft" in dims:
                x_vals_2d = extract_pft2d_x(ds_for_x, v, pft_indices)
                y_vals_2d = extract_pft2d_y(ds_r_list, v, pft_indices) # Y值来源不变
            else:
                x_vals_2d = extract_col2d_x(ds_for_x, v, col_indices)
                y_vals_2d = extract_col2d_y(ds_r_list, v, col_indices) # Y值来源不变
            
            # var_obj = ds_restart.variables[v]
            # dims = tuple(var_obj.dimensions)
            # if "pfts1d" in dims:
            #     x_vals_2d = extract_pft2d_x(ds_restart, v, pft_indices)
            #     y_vals_2d = extract_pft2d_y(ds_r_list, v, pft_indices)
            # else:
            #     x_vals_2d = extract_col2d_x(ds_restart, v, col_indices)
            #     y_vals_2d = extract_col2d_y(ds_r_list, v, col_indices)
            results_x_2d[v].append(x_vals_2d)
            results_y_2d[f"Y_{v}"].append(y_vals_2d)

        if (row_idx + 1) % 1000 == 0:
            print(f"  已处理 {row_idx + 1} / {len(df)} 行 ...")

    # 写入 DataFrame
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
        "为现有 batched 训练数据追加按分类新增的 RESTART 变量及对应 Y_ 列（变量名来自 config/txt 配置）"
    )
    parser.add_argument(
        "--input_glob",
        default=Config.INPUT_GLOB,
        help="输入 pkl 批次文件的 glob 路径模式（默认原始数据路径与模式）"
    )
    parser.add_argument(
        "--output_dir",
        default=Config.OUTPUT_DIR,
        help="输出目录（默认增强数据目录）"
    )

    args = parser.parse_args()

    # 数据文件路径（写死为固定路径，独立运行无需依赖 config）
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

    # 从配置读取新增分类并映射到三个 RESTART 组：
    # - RESTART_PFT_VARS        <- dataset_new_1D_PFT_VARIABLES
    # - RESTART_COL_1D_VARS     <- dataset_new_RESTART_COL_1D_VARS
    # - RESTART_COL_2D_VARS     <- dataset_new_Water_variables + dataset_new_2D_VARIABLES
    restart_pft_vars = list(dict.fromkeys(Config.dataset_new_1D_PFT_VARIABLES))
    restart_col_1d_vars = list(dict.fromkeys(Config.dataset_new_RESTART_COL_1D_VARS))
    restart_col_2d_vars = list(dict.fromkeys(list(Config.dataset_new_Water_variables) + list(Config.dataset_new_2D_VARIABLES)))

    # 映射到增强函数需要的 1D/2D 输入
    # configured_vars_1d = list(dict.fromkeys(restart_pft_vars + restart_col_1d_vars))
    # configured_vars_2d = list(dict.fromkeys(restart_col_2d_vars))
    configured_vars_1d = list(dict.fromkeys(restart_pft_vars + restart_col_1d_vars))
    configured_vars_2d = list(dict.fromkeys(restart_col_2d_vars + Config.SPECIAL_P_VARS))

    input_files = sorted(glob.glob(args.input_glob))
    if not input_files:
        print(f"未找到任何输入文件: {args.input_glob}")
        sys.exit(1)

    # 确保目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    print("打开 restart 文件与 Y 文件组 ...")
    ds_restart = nc.Dataset(file_path10)
    ds_special_p_restart = nc.Dataset(Config.SPECIAL_P_INPUT_NC) 
    ds_r_list = [nc.Dataset(fp) for fp in [file_path17, file_path18, file_path19, file_path20, file_path21]]

    try:
        print("构建 restart 网格 KDTree 与列索引映射 ...")
        restart_tree, restart_coords = build_restart_kdtree(ds_restart)
        col_index_map = build_column_index_map(ds_restart)
        pft_index_map = build_pft_index_map(ds_restart)
        
        
        print(f"开始处理 {len(input_files)} 个批次文件 ...")
        for fp in input_files:
            print(f"处理: {fp}")
            df = pd.read_pickle(fp)

            # 仅对缺失的新增变量进行增强；若全部已存在则跳过该批次
            # 注意：只针对“新增分类变量”，不影响旧有列
            # missing_vars_1d = [v for v in configured_vars_1d if not (v in df.columns and f"Y_{v}" in df.columns)]
            # missing_vars_2d = [v for v in configured_vars_2d if not (v in df.columns and f"Y_{v}" in df.columns)]
            # existing_vars_1d = [v for v in configured_vars_1d if v not in missing_vars_1d]
            # existing_vars_2d = [v for v in configured_vars_2d if v not in missing_vars_2d]

            # if not missing_vars_1d and not missing_vars_2d:
            #     print("  新增变量在该批次均已存在，跳过。")
            #     print(f"  目标新增变量（1D）: {configured_vars_1d}")
            #     print(f"  目标新增变量（2D）: {configured_vars_2d}")
            #     print(f"  已存在（1D）: {existing_vars_1d}")
            #     print(f"  已存在（2D）: {existing_vars_2d}")
            #     continue
            # ========== 从这里开始复制，替换上面的旧代码块 ==========

            # 1. 识别需要强制替换的变量 (来自Config)
            force_replace_vars = set(Config.SPECIAL_P_VARS)

            # 2. 找出除了强制替换变量之外的其他缺失变量
            missing_other_vars_1d = [
                v for v in configured_vars_1d 
                if v not in force_replace_vars and not (v in df.columns and f"Y_{v}" in df.columns)
            ]
            missing_other_vars_2d = [
                v for v in configured_vars_2d 
                if v not in force_replace_vars and not (v in df.columns and f"Y_{v}" in df.columns)
            ]

            # 3. 合并成最终需要处理的变量列表
            #    - 1D变量 = 其他缺失的1D变量 (因为特殊变量都是2D的)
            #    - 2D变量 = 其他缺失的2D变量 + 所有需要强制替换的变量
            vars_to_process_1d = missing_other_vars_1d
            # 使用 set.union 和 list() 来合并并去重
            vars_to_process_2d = list(set(missing_other_vars_2d).union(force_replace_vars))

            # 4. 检查是否无事可做
            if not vars_to_process_1d and not vars_to_process_2d:
                print("  所有变量均已存在且无需强制替换，跳过。")
                continue

            # 5. 打印本次将要处理的变量清单 (新的打印逻辑)
            print("  ----------------------------------------")
            if force_replace_vars.intersection(vars_to_process_2d):
                print(f"  强制替换变量: {list(force_replace_vars.intersection(vars_to_process_2d))}")
            if missing_other_vars_1d:
                print(f"  待新增变量 (1D): {missing_other_vars_1d}")
            if missing_other_vars_2d:
                print(f"  待新增变量 (2D): {missing_other_vars_2d}")
            print("  ----------------------------------------")

            # ========== 替换到这里结束 ==========
            # # 打印本次将新增的变量清单
            # if missing_vars_1d:
            #     print(f"  待新增变量（1D）: {missing_vars_1d}")
            # if missing_vars_2d:
            #     print(f"  待新增变量（2D）: {missing_vars_2d}")

            df_aug = augment_dataframe_with_vars(
                df=df,
                ds_restart=ds_restart,
                ds_special_p_restart=ds_special_p_restart,
                ds_r_list=ds_r_list,
                restart_tree=restart_tree,
                restart_coords=restart_coords,
                col_index_map=col_index_map,
                pft_index_map=pft_index_map,
                # vars_1d=missing_vars_1d,
                # vars_2d=missing_vars_2d,
                vars_1d=vars_to_process_1d, # <--- 确认修改
                vars_2d=vars_to_process_2d, # <--- 确认修改
                special_p_vars=Config.SPECIAL_P_VARS
            )

            base_name = os.path.basename(fp)
            # 输出命名遵循增强前缀: enhanced_1_training_data_batch_XX.pkl
            out_name = f"{Config.ENHANCED_PREFIX}{base_name}"
            out_path = os.path.join(args.output_dir, out_name)
            df_aug.to_pickle(out_path)
            print(f"  已保存: {out_path}")
    finally:
        print("关闭所有 NetCDF 文件 ...")
        ds_restart.close()
        ds_special_p_restart.close()
        for ds in ds_r_list:
            try:
                ds.close()
            except Exception:
                pass

    print("全部完成。")


if __name__ == "__main__":
    main()

