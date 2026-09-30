"""
供需匹配模型模块
实现两步移动搜索法(2SFCA)进行供需匹配分析
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import sys
import importlib
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
config = importlib.import_module('config')
from utils import read_geojson_without_fiona, save_geojson, create_grid, cdist_haversine, simple_pca, extract_place_type

SEED = config.SEED
BOUNDARY_FILE = config.BOUNDARY_FILE
DISTRICT_GEOJSON = config.DISTRICT_GEOJSON
RAW_POI_FILE = config.RAW_POI_FILE
OUTPUT_DIR = config.OUTPUT_DIR
SEARCH_RADIUS = config.SEARCH_RADIUS
GRID_SIZE = config.GRID_SIZE
POPULATION_FILE = config.POPULATION_FILE
PLACE_TYPE_MAP = config.PLACE_TYPE_MAP
WEIGHTS = config.WEIGHTS
DATA_DIR = config.DATA_DIR

np.random.seed(SEED)


def two_stage_fca(grid_coords, demands, supplies, charging_coords, search_radius):
    """
    两步移动搜索法(2SFCA)— 正确实现

    参数:
        grid_coords: 网格中心点坐标 (n_demand, 2)
        demands: 各网格需求量 (n_demand,)
        supplies: 各充电站供给量(端口数)(n_supply,)
        charging_coords: 充电站坐标 (n_supply, 2)
        search_radius: 搜索半径(米)

    返回:
        各网格的供需匹配得分 (n_demand,)
    """
    dists = cdist_haversine(grid_coords, charging_coords)  # (n_demand, n_supply)

    # 高斯衰减权重矩阵
    gaussian_w = np.where(dists <= search_radius,
                          np.exp(-0.5 * (dists / search_radius) ** 2), 0)

    # Step 1: 计算每个充电站的供需比 Rj = Sj / SUM(Pk * Wkj)
    demand_weighted = gaussian_w * demands[:, np.newaxis]  # (n_demand, n_supply)
    demand_sum = demand_weighted.sum(axis=0)  # (n_supply,)
    Rj = np.where(demand_sum > 0, supplies / demand_sum, 0)  # (n_supply,)

    # Step 2: 计算每个需求点的可达性得分 Ai = SUM(Rj * Wij)
    scores = (gaussian_w * Rj).sum(axis=1)  # (n_demand,)

    return scores


def enhanced_2sfca(grid_coords, demands, supplies, charging_coords,
                   distance_breaks=None, zone_weights=None):
    """
    E2SFCA (Enhanced 2SFCA) — Luo & Qi (2009)
    将搜索半径分为多个子区段，每段赋予不同权重，
    更真实地刻画出行阻抗的非线性衰减。

    参数:
        grid_coords: 网格中心点 (n_demand, 2)
        demands: 需求量 (n_demand,)
        supplies: 供给量 (n_supply,)
        charging_coords: 充电站坐标 (n_supply, 2)
        distance_breaks: 距离分段，默认 [0, 1000, 3000, 5000] 米
        zone_weights: 各段权重，默认 [1.0, 0.68, 0.22]

    返回:
        各网格的可达性得分 (n_demand,)
    """
    if distance_breaks is None:
        distance_breaks = [0, 1000, 3000, 5000]
    if zone_weights is None:
        zone_weights = [1.0, 0.68, 0.22]

    dists = cdist_haversine(grid_coords, charging_coords)  # (n_demand, n_supply)

    # 逐区段计算衰减权重
    step_weights = np.zeros_like(dists)
    for k in range(len(distance_breaks) - 1):
        d_low = distance_breaks[k]
        d_high = distance_breaks[k + 1]
        mask = (dists >= d_low) & (dists < d_high)
        # 高斯衰减 + 区段权重
        gauss = np.exp(-0.5 * (dists / distance_breaks[-1]) ** 2)
        step_weights += mask * gauss * zone_weights[k]

    # Step 1: Rj = Sj / SUM(Pk * Wkj)
    demand_weighted = step_weights * demands[:, np.newaxis]
    demand_sum = demand_weighted.sum(axis=0)
    Rj = np.where(demand_sum > 0, supplies / demand_sum, 0)

    # Step 2: Ai = SUM(Rj * Wij)
    scores = (step_weights * Rj).sum(axis=1)

    return scores


def calculate_morans_i(values, coords, k=8):
    """
    计算全局Moran's I指数
    
    参数:
        values: 变量值数组
        coords: 坐标数组
        k: 最近邻数量
    
    返回:
        Moran's I值
    """
    n = len(values)
    dists = cdist_haversine(coords, coords)
    knn_indices = np.argsort(dists, axis=1)[:, 1:k+1]
    
    mean_val = np.mean(values)
    var_val = np.var(values)
    if var_val == 0:
        return 0
    
    w_sum = 0
    s0 = 0
    
    for i in range(n):
        neighbors = knn_indices[i]
        for j in neighbors:
            w = 1 / (dists[i, j] + 1e-10)
            w_sum += w * (values[i] - mean_val) * (values[j] - mean_val)
            s0 += w
    
    if s0 == 0:
        return 0
    
    moran_i = (n / s0) * (w_sum / (n * var_val))
    return moran_i

def calculate_gis(values, coords, k=8):
    """
    计算Getis-Ord Gi*统计量
    
    参数:
        values: 变量值数组
        coords: 坐标数组
        k: 最近邻数量
    
    返回:
        Gi*值数组
    """
    n = len(values)
    dists = cdist_haversine(coords, coords)
    knn_indices = np.argsort(dists, axis=1)[:, 1:k+1]
    
    mean_val = np.mean(values)
    
    gi_values = []
    for i in range(n):
        neighbors = knn_indices[i]
        weights = 1 / (dists[i, neighbors] + 1e-10)
        w_sum_w = np.sum(weights)
        if w_sum_w == 0:
            gi_values.append(0)
            continue
        
        weighted_sum = np.sum(weights * values[neighbors])
        numerator = weighted_sum - mean_val * w_sum_w
        denominator = np.std(values) * np.sqrt((n * np.sum(weights ** 2) - w_sum_w ** 2) / (n - 1))
        
        gi_values.append(numerator / denominator if denominator != 0 else 0)
    
    return np.array(gi_values)

def main():
    """
    主函数:供需匹配分析
    """
    print("=" * 50)
    print("供需匹配模型开始运行")
    print("=" * 50)
    
    try:
        print("读取边界数据...")
        boundary_file = DISTRICT_GEOJSON if DISTRICT_GEOJSON.exists() else BOUNDARY_FILE
        boundary = read_geojson_without_fiona(boundary_file)
        print(f"  边界数据读取成功: {len(boundary)} 个区")

        print("读取充电站数据...")
        charging_gdf = read_geojson_without_fiona(DATA_DIR / "charging_stations_xian.geojson")
        charging_gdf['ports'] = charging_gdf['ports'].fillna(3)
        print(f"  充电站数据读取成功: {len(charging_gdf)} 个站")

        print("读取POI数据...")
        df = pd.read_csv(RAW_POI_FILE, encoding='utf-8-sig')
        
        # 检测类型列
        type_cols = []
        for col in df.columns:
            if '类型' in str(col) or '分类' in str(col) or '大类' in str(col) or '中类' in str(col):
                type_cols.append(col)
        
        if type_cols:
            df['type'] = df[type_cols].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        else:
            df['type'] = ''

        # 检测并映射经纬度列名
        for col in df.columns:
            col_lower = str(col).lower()
            if '经度' in str(col) or 'lon' in col_lower or 'long' in col_lower:
                df['lon'] = df[col]
            elif '纬度' in str(col) or 'lat' in col_lower:
                df['lat'] = df[col]
        if 'lon' not in df.columns:
            df['lon'] = df.iloc[:, 3]  # fallback
        if 'lat' not in df.columns:
            df['lat'] = df.iloc[:, 4]  # fallback
        print(f"  POI数据读取成功: {len(df)} 条记录")

        print("创建网格...")
        grid = create_grid(boundary, GRID_SIZE, max_cells=3000)
        print(f"  网格创建成功: {len(grid)} 个网格单元")

        print("计算需求指数...")
        poi_types = df[['lon', 'lat', 'type']].copy()
        poi_types['place_type'] = poi_types['type'].apply(lambda x: extract_place_type(x))
        
        demand_features = pd.DataFrame()
        grid_coords = grid[['lon', 'lat']].values

        # 将经纬度转换为局部平面坐标(米)，供 cKDTree 做半径查询，避免全量网格×POI距离矩阵
        lon0, lat0 = grid_coords[:, 0].mean(), grid_coords[:, 1].mean()
        m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
        m_per_deg_lat = 110574.0
        grid_xy = np.column_stack([
            (grid_coords[:, 0] - lon0) * m_per_deg_lon,
            (grid_coords[:, 1] - lat0) * m_per_deg_lat,
        ])
        KERNEL_RADIUS = 5000.0  # 高斯核在5km外近似为0

        for place_type in PLACE_TYPE_MAP.keys():
            mask = poi_types['place_type'] == place_type
            subset = poi_types[mask]
            if len(subset) == 0:
                demand_features[place_type] = 0
                continue
            poi_coords = subset[['lon', 'lat']].values
            poi_xy = np.column_stack([
                (poi_coords[:, 0] - lon0) * m_per_deg_lon,
                (poi_coords[:, 1] - lat0) * m_per_deg_lat,
            ])
            tree = cKDTree(poi_xy)
            demand_sum = np.zeros(len(grid))
            for i in range(len(grid)):
                nbrs = tree.query_ball_point(grid_xy[i], KERNEL_RADIUS)
                if nbrs:
                    d = np.linalg.norm(poi_xy[nbrs] - grid_xy[i], axis=1)
                    demand_sum[i] = np.exp(-0.5 * (d / 1000.0) ** 2).sum()
            demand_features[place_type] = demand_sum

        if demand_features.empty:
            grid['demand'] = 1
        else:
            demand_features_norm = (demand_features - demand_features.mean()) / (demand_features.std() + 1e-10)
            grid['demand'] = simple_pca(demand_features_norm.values, n_components=1).flatten()
            grid['demand'] = (grid['demand'] - grid['demand'].min()) / (grid['demand'].max() - grid['demand'].min() + 1e-10)

        # 保存交通POI密度(归一化到0-1),供选址优化"交通导向"情景做第三目标
        if '交通' in demand_features.columns:
            transit = np.asarray(demand_features['交通'], dtype=np.float64).ravel()
            t_min, t_max = transit.min(), transit.max()
            grid['transit_density'] = (transit - t_min) / (t_max - t_min + 1e-10) if t_max > t_min else np.zeros(len(grid))
        else:
            grid['transit_density'] = 0
        print("  需求指数计算成功")

        print("计算供给量...")
        grid['supply'] = 0
        charging_coords = charging_gdf[['lon', 'lat']].values
        supplies = charging_gdf['ports'].values
        
        for i, row in enumerate(charging_gdf.itertuples()):
            dists = cdist_haversine([[row.lon, row.lat]], grid_coords)
            mask = dists[0] < GRID_SIZE / 2
            grid.loc[mask, 'supply'] += row.ports
        print("  供给量计算成功")

        print("计算2SFCA得分...")
        sensitivity_results = []
        demands = grid['demand'].values
        for radius in SEARCH_RADIUS:
            scores = two_stage_fca(grid_coords, demands, supplies, charging_coords, radius)
            grid[f'score_{radius}'] = scores
            sensitivity_results.append({
                'parameter': f'radius={radius}m',
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'min_score': float(scores.min()),
                'max_score': float(scores.max())
            })
            print(f"  半径 {radius}m: 平均得分={scores.mean():.4f}")

        # E2SFCA (子区段权重)
        e2sfca_scores = enhanced_2sfca(grid_coords, demands, supplies, charging_coords)
        grid['score_e2sfca'] = e2sfca_scores
        sensitivity_results.append({
            'parameter': 'E2SFCA (1.0/0.68/0.22)',
            'mean_score': float(e2sfca_scores.mean()),
            'std_score': float(e2sfca_scores.std()),
            'min_score': float(e2sfca_scores.min()),
            'max_score': float(e2sfca_scores.max())
        })
        print(f"  E2SFCA: 平均得分={e2sfca_scores.mean():.4f}")

        # 衰减函数对比(使用中间搜索半径)
        mid_radius = SEARCH_RADIUS[len(SEARCH_RADIUS)//2]
        for decay_name, decay_fn in [
            ('exponential', lambda d, d0: np.where(d <= d0, np.exp(-d / d0), 0)),
            ('binary', lambda d, d0: np.where(d <= d0, 1.0, 0)),
        ]:
            dists = cdist_haversine(grid_coords, charging_coords)
            gaussian_w = decay_fn(dists, mid_radius)
            demand_weighted = gaussian_w * demands[:, np.newaxis]
            demand_sum = demand_weighted.sum(axis=0)
            Rj = np.where(demand_sum > 0, supplies / demand_sum, 0)
            scores = (gaussian_w * Rj).sum(axis=1)
            grid[f'score_{decay_name}'] = scores
            sensitivity_results.append({
                'parameter': f'decay={decay_name}',
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'min_score': float(scores.min()),
                'max_score': float(scores.max())
            })
            print(f"  衰减函数 {decay_name}: 平均得分={scores.mean():.4f}")

        grid['score'] = grid['score_e2sfca']
        grid['score_norm'] = (grid['score'] - grid['score'].min()) / (grid['score'].max() - grid['score'].min() + 1e-10)

        pd.DataFrame(sensitivity_results).to_csv(OUTPUT_DIR / "sensitivity_analysis.csv", index=False, encoding='utf-8-sig')
        print(f"  输出敏感性分析结果")

        print("计算Moran's I...")
        moran_i = calculate_morans_i(grid['score_norm'].values, grid_coords)
        print(f"  Moran's I: {moran_i:.4f}")

        print("冷热点分析...")
        gi_values = calculate_gis(grid['score_norm'].values, grid_coords)
        grid['gi'] = gi_values
        grid['hotspot'] = np.digitize(gi_values, np.percentile(gi_values, [25, 50, 75])) + 1
        print("  冷热点分析成功")

        save_geojson(grid, OUTPUT_DIR / "grid_supply_demand.geojson")
        print(f"  输出网格供需得分: grid_supply_demand.geojson")

        print("绘制供需分布图...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        boundary.plot(ax=ax, color='lightgray', edgecolor='black')
        grid.plot(ax=ax, column='score_norm', cmap='RdYlBu_r', legend=True,
                  legend_kwds={'label': '供需匹配得分', 'orientation': 'horizontal'})
        charging_gdf.plot(ax=ax, marker='o', color='black', markersize=5)
        plt.title('西安市充电站供需匹配得分空间分布图', fontsize=14)
        plt.axis('off')
        plt.savefig(OUTPUT_DIR / "supply_demand_map.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  输出供需分布图: supply_demand_map.png")

        print("绘制冷热点图...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        boundary.plot(ax=ax, color='lightgray', edgecolor='black')
        hotspot_colors = {1: 'blue', 2: 'lightblue', 3: 'orange', 4: 'red'}
        for q in sorted(grid['hotspot'].unique()):
            subset = grid[grid['hotspot'] == q]
            subset.plot(ax=ax, color=hotspot_colors.get(q, 'gray'), alpha=0.6)
        plt.title('西安市充电站供需冷热点分析图(Getis-Ord Gi*)', fontsize=14)
        plt.legend(['冷点', '次冷点', '次热点', '热点'], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        plt.savefig(OUTPUT_DIR / "hotspot_map.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  输出冷热点图: hotspot_map.png")

        print("识别服务盲区...")
        threshold = grid['score_norm'].quantile(0.2)
        blind_spots = grid[grid['score_norm'] <= threshold]
        save_geojson(blind_spots, OUTPUT_DIR / "service_blind_spots.geojson")
        print(f"  输出服务盲区: service_blind_spots.geojson (共{len(blind_spots)}个)")

        print("=" * 50)
        print("供需匹配模型完成!")
        print("=" * 50)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
