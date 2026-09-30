"""
效果评估模块
评估优化前后的各项指标并进行对比分析
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import importlib
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
config = importlib.import_module('config')
from utils import read_geojson_without_fiona, cdist_haversine, calculate_gini

DATA_DIR = config.DATA_DIR
OUTPUT_DIR = config.OUTPUT_DIR
GRID_SIZE = config.GRID_SIZE
COVER_THRESHOLD = config.COVER_THRESHOLD

def calculate_theil(values):
    """计算 Theil 指数(可分解的不平等度量)"""
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0
    mean_val = values.mean()
    if mean_val == 0:
        return 0
    return np.mean((values / mean_val) * np.log(values / mean_val))

def calculate_metrics(grid, charging_sites, demand_col='demand', coverage_radius=5000):
    """
    计算扩展评估指标(10+维度)

    参数:
        grid: 需求网格GeoDataFrame
        charging_sites: 充电站DataFrame
        demand_col: 需求列名
        coverage_radius: 覆盖半径(米)

    返回:
        指标字典
    """
    demand_coords = grid[['lon', 'lat']].values
    site_coords = charging_sites[['lon', 'lat']].values
    weights = grid[demand_col].values
    total_demand = np.sum(weights)

    if len(site_coords) == 0:
        return {'avg_distance': np.inf, 'coverage': 0.0, 'area_coverage': 0.0,
                'p95_distance': np.inf, 'gini': 1.0, 'dist_gini': 1.0, 'theil': 0.0,
                'overlap_rate': 0.0, 'demand_supply_ratio': 0.0,
                'ports_per_10k_demand': 0.0, 'total_sites': 0, 'total_ports': 0}

    dists = cdist_haversine(demand_coords, site_coords)
    min_dists = dists.min(axis=1)

    # --- 基础指标 ---
    avg_distance = np.average(min_dists, weights=weights)
    weighted_coverage = np.sum(weights * (min_dists <= coverage_radius)) / total_demand * 100
    area_coverage = (min_dists <= coverage_radius).mean() * 100
    p95_distance = np.percentile(min_dists, 95)

    # --- 公平性指标 ---
    # 距离基尼(可达距离的平等程度)
    dist_gini = calculate_gini(min_dists)
    # 可达性基尼(2SFCA得分 — 空间公平性的标准度量)
    if 'score_norm' in grid.columns:
        accessibility = grid['score_norm'].values
    elif 'score' in grid.columns:
        score = grid['score'].values
        accessibility = (score - score.min()) / (score.max() - score.min() + 1e-10)
    else:
        accessibility = 1.0 / (min_dists + 1.0)
    accessibility_gini = calculate_gini(accessibility)
    theil = calculate_theil(accessibility)

    # --- 供需指标 ---
    if 'supply' in grid.columns:
        supply_values = grid['supply'].values
    else:
        supply_values = np.zeros(len(grid))
        site_ports = charging_sites.get('ports', pd.Series([1] * len(charging_sites))).values
        for i, (lon, lat) in enumerate(site_coords):
            d = cdist_haversine([[lon, lat]], demand_coords)[0]
            supply_values[d < coverage_radius] += site_ports[i] if i < len(site_ports) else 1
    grid_demand = weights
    demand_supply_ratio = np.mean(grid_demand / (supply_values + 1e-6))

    # --- 服务重叠率 ---
    covered_by = (dists <= coverage_radius).sum(axis=1)
    overlap_rate = (covered_by > 1).mean() * 100

    # --- 站点效率 ---
    total_ports = charging_sites.get('ports', pd.Series([1] * len(charging_sites))).sum()
    ports_per_10k_demand = total_ports / (total_demand / 10000) if total_demand > 0 else 0

    return {
        'avg_distance': avg_distance,
        'coverage': weighted_coverage,
        'area_coverage': area_coverage,
        'p95_distance': p95_distance,
        'gini': accessibility_gini,
        'dist_gini': dist_gini,
        'theil': theil,
        'overlap_rate': overlap_rate,
        'demand_supply_ratio': demand_supply_ratio,
        'ports_per_10k_demand': ports_per_10k_demand,
        'total_sites': len(charging_sites),
        'total_ports': int(total_ports),
    }

def monte_carlo_simulation(grid, charging_gdf, new_sites=None, n_simulations=100, seed=42):
    """
    蒙特卡洛不确定性量化 — 对端口数等输入参数施加随机扰动,
    重复计算关键指标,输出均值和置信区间。

    参数:
        grid: 需求网格 GeoDataFrame (需包含 'score_norm' 或 'score' 列)
        charging_gdf: 现有充电站 GeoDataFrame (需包含 'ports' 列)
        new_sites: 新增站点 DataFrame (可选)
        n_simulations: 模拟次数
        seed: 随机种子

    返回:
        DataFrame: 各指标的 mean, std, ci_lower, ci_upper
    """
    rng = np.random.default_rng(seed)
    demand_coords = grid[['lon', 'lat']].values
    weights = grid['demand'].values

    if new_sites is not None and len(new_sites) > 0:
        all_sites_base = pd.concat([
            charging_gdf[['lon', 'lat', 'ports']],
            new_sites[['lon', 'lat', 'ports']]
        ], ignore_index=True)
    else:
        all_sites_base = charging_gdf[['lon', 'lat', 'ports']].copy()

    base_ports = all_sites_base['ports'].values.astype(float)
    site_coords = all_sites_base[['lon', 'lat']].values

    all_results = []
    for sim in range(n_simulations):
        # 对端口数施加对数正态扰动(σ=0.3, 即约30%相对误差)
        noise = rng.lognormal(mean=0.0, sigma=0.3, size=len(base_ports))
        ports_noisy = np.maximum(1, np.round(base_ports * noise)).astype(int)

        # 计算2SFCA得分
        dists = cdist_haversine(demand_coords, site_coords)
        gaussian_w = np.where(dists <= 5000,
                              np.exp(-0.5 * (dists / 5000) ** 2), 0)
        demand_weighted = gaussian_w * weights[:, np.newaxis]
        demand_sum = demand_weighted.sum(axis=0)
        Rj = np.where(demand_sum > 0, ports_noisy / demand_sum, 0)
        scores = (gaussian_w * Rj).sum(axis=1)

        # 计算关键指标
        min_dists = dists.min(axis=1)
        avg_dist = np.average(min_dists, weights=weights)
        coverage = np.sum(weights * (min_dists <= 5000)) / np.sum(weights) * 100
        p95 = np.percentile(min_dists, 95)

        v = np.sort(scores)
        n = len(v)
        gini = (2 * np.sum(np.arange(1, n + 1) * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)) if n > 1 and np.sum(v) > 0 else 0

        all_results.append({
            'simulation': sim,
            'avg_distance': avg_dist,
            'coverage': coverage,
            'p95_distance': p95,
            'gini': gini,
            'total_ports': int(np.sum(ports_noisy)),
        })

    df = pd.DataFrame(all_results)

    # 汇总统计
    metric_cols = ['avg_distance', 'coverage', 'p95_distance', 'gini', 'total_ports']
    summary = []
    for col in metric_cols:
        vals = df[col].values
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        ci_lower = np.percentile(vals, 2.5)
        ci_upper = np.percentile(vals, 97.5)
        summary.append({
            '指标': col,
            '均值': round(mean_v, 3),
            '标准差': round(std_v, 3),
            'CV(%)': round(std_v / mean_v * 100, 1) if mean_v != 0 else 0,
            '95%CI下界': round(ci_lower, 3),
            '95%CI上界': round(ci_upper, 3),
        })

    summary_df = pd.DataFrame(summary)
    return df, summary_df


def main():
    """
    主函数:效果评估
    """
    print("=" * 50)
    print("效果评估开始")
    print("=" * 50)
    
    print("读取需求网格...")
    grid = read_geojson_without_fiona(OUTPUT_DIR / "grid_supply_demand.geojson")
    grid['demand'] = grid.get('demand', 1)
    
    print("读取现有充电站...")
    charging_gdf = read_geojson_without_fiona(DATA_DIR / "charging_stations_xian.geojson")
    
    print("读取新增站点...")
    new_sites = pd.read_csv(OUTPUT_DIR / "new_sites.csv", encoding='utf-8-sig')
    new_sites['ports'] = new_sites.get('ports', 6)
    
    print("计算优化前指标...")
    metrics_before = calculate_metrics(grid, charging_gdf)
    print(f"优化前 - 平均距离: {metrics_before['avg_distance']:.2f}m, "
          f"覆盖率: {metrics_before['coverage']:.2f}%, Gini: {metrics_before['gini']:.4f}, "
          f"Theil: {metrics_before['theil']:.4f}")

    print("计算优化后指标...")
    all_sites = pd.concat([charging_gdf[['lon', 'lat', 'ports']], new_sites[['lon', 'lat', 'ports']]], ignore_index=True)
    metrics_after = calculate_metrics(grid, all_sites)
    print(f"优化后 - 平均距离: {metrics_after['avg_distance']:.2f}m, "
          f"覆盖率: {metrics_after['coverage']:.2f}%, Gini: {metrics_after['gini']:.4f}, "
          f"Theil: {metrics_after['theil']:.4f}")

    metric_names = ['平均可达距离(m)', '覆盖率(%)', 'P95可达距离(m)', '可达性Gini', '距离Gini',
                    'Theil指数', '服务重叠率(%)', '站点总数', '总端口数']
    metric_keys = ['avg_distance', 'coverage', 'p95_distance', 'gini', 'dist_gini',
                   'theil', 'overlap_rate', 'total_sites', 'total_ports']
    comparison = pd.DataFrame({
        '指标': metric_names,
        '优化前': [metrics_before[k] for k in metric_keys],
        '优化后': [metrics_after[k] for k in metric_keys],
        '变化量': [metrics_after[k] - metrics_before[k] for k in metric_keys],
        '变化率(%)': [
            (metrics_after[k] - metrics_before[k]) / metrics_before[k] * 100 if metrics_before[k] > 0 else 0
            for k in metric_keys
        ]
    })
    
    comparison.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False, encoding='utf-8-sig')
    print(f"输出对比表: {OUTPUT_DIR / 'comparison_table.csv'}")
    
    print("绘制对比柱状图...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    comparison_melt = comparison.melt(id_vars='指标', value_vars=['优化前', '优化后'])
    sns.barplot(data=comparison_melt, x='指标', y='value', hue='variable', palette=['blue', 'green'])
    plt.title('优化前后指标对比', fontsize=14)
    plt.xlabel('指标', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.legend(title='状态')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出对比图: {OUTPUT_DIR / 'comparison_chart.png'}")
    
    print("绘制雷达图...")
    labels = ['覆盖率', '可达性', '公平性(1-Gini)', '效率(1-重叠率)', '均衡性(1-Theil)']
    before = [
        metrics_before['coverage']/100,
        1 - min(metrics_before['avg_distance']/20000, 1),
        1 - metrics_before['gini'],
        1 - metrics_before['overlap_rate']/100,
        1 - min(metrics_before['theil'], 1)
    ]
    after = [
        metrics_after['coverage']/100,
        1 - min(metrics_after['avg_distance']/20000, 1),
        1 - metrics_after['gini'],
        1 - metrics_after['overlap_rate']/100,
        1 - min(metrics_after['theil'], 1)
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    before += before[:1]
    after += after[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
    ax.fill(angles, before, 'blue', alpha=0.25)
    ax.fill(angles, after, 'green', alpha=0.25)
    ax.plot(angles, before, 'blue', linewidth=2, label='优化前')
    ax.plot(angles, after, 'green', linewidth=2, label='优化后')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('优化前后综合指标对比', fontsize=14)
    plt.legend()
    plt.savefig(OUTPUT_DIR / "radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出雷达图: {OUTPUT_DIR / 'radar_chart.png'}")
    
    print("=" * 50)
    print("效果评估完成!")
    print("=" * 50)

    # 蒙特卡洛不确定性量化
    print("蒙特卡洛模拟(100次)...")
    mc_results, mc_summary = monte_carlo_simulation(grid, charging_gdf, new_sites, n_simulations=100)
    mc_summary.to_csv(OUTPUT_DIR / "monte_carlo_summary.csv", index=False, encoding='utf-8-sig')
    print(f"输出蒙特卡洛汇总: {OUTPUT_DIR / 'monte_carlo_summary.csv'}")
    for _, row in mc_summary.iterrows():
        print(f"  {row['指标']}: {row['均值']:.3f} ± {row['标准差']:.3f} (CV={row['CV(%)']}%)")

if __name__ == "__main__":
    main()
