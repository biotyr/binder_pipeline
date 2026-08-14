"""
阈值过滤逻辑 — 移植自 BindCraft

原始文件: /home/biotyr/esmbind/BindCraft-main/functions/generic_utils.py
"""

import json
import os
from typing import Any

import pandas as pd


def check_filters(
    data_dict: dict[str, Any],
    filters: dict[str, Any],
) -> list[str] | bool:
    """多维度阈值过滤 — 移植自 BindCraft generic_utils.py:371-411

    对设计数据逐项检查 filter 阈值,返回未满足条件列表或 True。

    Parameters
    ----------
    data_dict : dict
        {label: value} 设计指标字典,
        例如 {"Average_pLDDT": 0.85, "1_dG": -5.2, "Average_InterfaceAAs": {"K": 2, ...}}
    filters : dict
        filter 配置字典 (来自 settings_filters JSON),
        格式: {"<label>": {"threshold": <value_or_None>, "higher": <bool>}}

        label 举例:
          - "Average_pLDDT"        → 全部模型平均 pLDDT
          - "1_pLDDT"              → 模型 1 的 pLDDT
          - "Average_dG"           → 全部模型平均 dG
          - "Average_InterfaceAAs" → {"K": {"threshold": 3, "higher": False}}
          - "Average_Binder_Loop%" → binder loop 比例

    Returns
    -------
    list[str] | bool
        True = 全部通过; list[str] = 未满足条件的 label 列表
    """
    unmet_conditions: list[str] = []

    for label, conditions in filters.items():
        # AA 组成过滤 (例如 "Average_InterfaceAAs" → 子字典 {aa: {threshold, higher}})
        aa_prefixes = ('Average_InterfaceAAs', '1_InterfaceAAs', '2_InterfaceAAs',
                       '3_InterfaceAAs', '4_InterfaceAAs', '5_InterfaceAAs')
        if any(label == prefix or label.startswith(prefix + '_') for prefix in aa_prefixes):
            if label not in data_dict or data_dict[label] is None:
                continue
            for aa, aa_conditions in conditions.items():
                if not isinstance(aa_conditions, dict):
                    continue
                value = data_dict[label].get(aa)
                threshold = aa_conditions.get("threshold")
                if value is None or threshold is None:
                    continue
                if aa_conditions["higher"]:
                    if value < threshold:
                        unmet_conditions.append(f"{label}_{aa}")
                else:
                    if value > threshold:
                        unmet_conditions.append(f"{label}_{aa}")
        else:
            value = data_dict.get(label)
            threshold = conditions.get("threshold")
            if value is None or threshold is None:
                continue
            if conditions["higher"]:
                if value < threshold:
                    unmet_conditions.append(label)
            else:
                if value > threshold:
                    unmet_conditions.append(label)

    return True if len(unmet_conditions) == 0 else unmet_conditions


def calculate_averages(
    statistics: dict[int, dict[str, Any]],
    handle_aa: bool = False,
) -> dict[str, Any]:
    """多模型统计量取平均 — 移植自 BindCraft generic_utils.py:329-368

    对多个 AF2 模型 (1-5) 的预测结果逐项取平均。

    Parameters
    ----------
    statistics : dict
        {model_num: {stat_name: value}}, model_num ∈ {1..5}
    handle_aa : bool
        是否对 InterfaceAAs 字典做氨基酸维度平均

    Returns
    -------
    dict
        {stat_name: averaged_value}
    """
    sums: dict[str, float] = {}
    aa_sums: dict[str, float] = {}

    for model_num in range(1, 6):
        if model_num not in statistics:
            continue
        model_stats = statistics[model_num]
        for stat, value in model_stats.items():
            if stat not in sums:
                sums[stat] = 0.0

            v = value if value is not None else 0.0

            if handle_aa and stat == 'InterfaceAAs':
                for aa, count in v.items():
                    aa_sums[aa] = aa_sums.get(aa, 0.0) + count
            else:
                sums[stat] += float(v)

    n_models = len(statistics)
    averages = {
        stat: round(total / n_models, 2) for stat, total in sums.items()
    }

    if handle_aa:
        averages['InterfaceAAs'] = {
            aa: round(total / n_models, 2) for aa, total in aa_sums.items()
        }

    return averages


def generate_filter_pass_csv(
    failure_csv: str,
    filter_json: str,
) -> None:
    """初始化 failure 统计 CSV — 移植自 BindCraft generic_utils.py:56-94

    根据 filter JSON 的 key 生成初始化为 0 的 failure 计数 CSV。

    Parameters
    ----------
    failure_csv : str
        输出 CSV 路径
    filter_json : str
        filter 配置 JSON 文件路径
    """
    if os.path.exists(failure_csv):
        return

    with open(filter_json, 'r') as file:
        data = json.load(file)

    names = [
        'Trajectory_logits_pLDDT', 'Trajectory_softmax_pLDDT',
        'Trajectory_one-hot_pLDDT', 'Trajectory_final_pLDDT',
        'Trajectory_Contacts', 'Trajectory_Clashes', 'Trajectory_WrongHotspot',
    ]
    special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
    tracked_filters: set[str] = set()

    for key in data.keys():
        processed_name = key
        for prefix in special_prefixes:
            if key.startswith(prefix):
                processed_name = key.split('_', 1)[1]
                break

        if 'InterfaceAAs' in processed_name:
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            for aa in amino_acids:
                variant_name = f"InterfaceAAs_{aa}"
                if variant_name not in tracked_filters:
                    names.append(variant_name)
                    tracked_filters.add(variant_name)
        elif processed_name not in tracked_filters:
            names.append(processed_name)
            tracked_filters.add(processed_name)

    df = pd.DataFrame(columns=names)
    df.loc[0] = [0] * len(names)
    df.to_csv(failure_csv, index=False)


def update_failures(
    failure_csv: str,
    failure_column_or_dict: str | dict[str, int],
) -> None:
    """递增 failure 计数 — 移植自 BindCraft generic_utils.py:97-124

    Parameters
    ----------
    failure_csv : str
        failure 统计 CSV 路径
    failure_column_or_dict : str | dict
        单个列名 (递增 1) 或 {column: count} 字典
    """
    failure_df = pd.read_csv(failure_csv)

    def strip_model_prefix(name: str) -> str:
        parts = name.split('_')
        if parts[0].isdigit():
            return '_'.join(parts[1:])
        return name

    if isinstance(failure_column_or_dict, dict):
        for filter_name, count in failure_column_or_dict.items():
            stripped_name = strip_model_prefix(filter_name)
            if stripped_name in failure_df.columns:
                failure_df[stripped_name] += count
            else:
                failure_df[stripped_name] = count
    else:
        failure_column = strip_model_prefix(failure_column_or_dict)
        if failure_column in failure_df.columns:
            failure_df[failure_column] += 1
        else:
            failure_df[failure_column] = 1

    failure_df.to_csv(failure_csv, index=False)
