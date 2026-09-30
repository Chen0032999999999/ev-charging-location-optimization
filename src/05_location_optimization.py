"""
选址优化模块
使用贪婪算法进行充电站选址优化
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib
from pathlib import Path
from shapely.geometry import Point
from shapely.ops import unary_union

sys.path.append(str(Path(__file__).parent))
config = importlib.import_module('config')
from utils import read_geojson_without_fiona, cdist_haversine

SEED = config.SEED
BOUNDARY_FILE = config.BOUNDARY_FILE
DISTRICT_GEOJSON = config.DISTRICT_GEOJSON
DATA_DIR = config.DATA_DIR
OUTPUT_DIR = config.OUTPUT_DIR
ROAD_FILE = config.ROAD_FILE
NEW_SITES_COUNT = config.NEW_SITES_COUNT
MIN_DISTANCE_EXISTING = config.MIN_DISTANCE_EXISTING
COVER_THRESHOLD = config.COVER_THRESHOLD

np.random.seed(SEED)

def _build_road_buffer(roads, buffer_m=150):
    """
    构建路网缓冲区 — 距道路 buffer_m 米内的区域视为"可通车"

    参数:
        roads: 路网GeoDataFrame (WGS84)
        buffer_m: 缓冲距离(米)

    返回:
        road_buffer_utm: UTM投影下的缓冲多边形
        roads_utm: UTM投影的路网
    """
    roads_utm = roads.to_crs("EPSG:32649")
    road_buffer_utm = roads_utm.buffer(buffer_m).union_all()
    return road_buffer_utm, roads_utm


def _snap_to_nearest_road(lon, lat, roads_wgs84, road_points_cache=None, top_k=5):
    """
    将点吸附到最近的道路上(从前top_k个最近路网点中随机选择,避免扎堆)

    参数:
        lon, lat: 待检查的经纬度
        roads_wgs84: 路网(WGS84)
        road_points_cache: 预计算的路网点列表(用于加速)
        top_k: 从最近的K个路网点中随机选择

    返回:
        (snapped_lon, snapped_lat, was_snapped: bool)
    """
    if road_points_cache is not None and len(road_points_cache) > 0:
        pts = np.array(road_points_cache)
        dists = cdist_haversine([[lon, lat]], pts)[0]
        # 取前top_k个最近点,随机选一个避免扎堆
        top_indices = np.argpartition(dists, min(top_k, len(dists)-1))[:top_k]
        chosen_idx = np.random.choice(top_indices)
        base_lon, base_lat = pts[chosen_idx][0], pts[chosen_idx][1]
        # 加微小随机偏移(±20m)
        jitter = 0.0002  # ~20m at this latitude
        return base_lon + (np.random.random() - 0.5) * jitter * 2, \
               base_lat + (np.random.random() - 0.5) * jitter * 2, True

    # fallback: 遍历路网几何
    min_dist = float('inf')
    nearest_pt = (lon, lat)
    for geom in roads_wgs84.geometry:
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            for cx, cy in coords:
                d = cdist_haversine([[lon, lat]], [[cx, cy]])[0][0]
                if d < min_dist:
                    min_dist = d
                    nearest_pt = (cx, cy)
    jitter = 0.0002
    return nearest_pt[0] + (np.random.random() - 0.5) * jitter * 2, \
           nearest_pt[1] + (np.random.random() - 0.5) * jitter * 2, True


def generate_candidates(blind_spots, charging_gdf, boundary=None, num_candidates=50, min_dist_existing=500):
    """
    生成候选点 — 优先盲区质心、路网沿线,过滤无效位置

    参数:
        blind_spots: 服务盲区GeoDataFrame
        charging_gdf: 现有充电站GeoDataFrame
        boundary: 边界GeoDataFrame(用于约束随机点范围)
        num_candidates: 候选点数量
        min_dist_existing: 与现有站点最小距离(米)

    返回:
        候选点DataFrame
    """
    candidates = []
    existing_coords = charging_gdf[['lon', 'lat']].values

    def _far_enough(lon, lat):
        """检查点是否与所有现有站点保持最小距离"""
        if len(existing_coords) == 0:
            return True
        dists = cdist_haversine([[lon, lat]], existing_coords)
        return dists.min() >= min_dist_existing

    # === 构建路网约束 (核心:确保候选点可通车) ===
    road_points_cache = None      # 路网点缓存
    road_buffer_utm = None        # 路网缓冲面(UTM)
    roads_wgs84 = None

    if ROAD_FILE and ROAD_FILE.exists():
        try:
            roads = gpd.read_file(ROAD_FILE)
            roads_wgs84 = roads.to_crs(epsg=4326)

            # 提取路网采样点缓存(用于快速吸附)
            road_points_cache = []
            for geom in roads_wgs84.geometry:
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    # 每100m采样一个点
                    step = max(1, len(coords) // max(1, int(geom.length * 111000) // 100))
                    for i in range(0, len(coords), step):
                        road_points_cache.append(coords[i])
            # 去重
            road_points_cache = list(set(road_points_cache))

            # 构建路网缓冲区 (UTM投影)
            road_buffer_utm, _ = _build_road_buffer(roads, buffer_m=150)
            print(f"  路网约束构建完成: {len(road_points_cache)}个采样点, 缓冲150m")
        except Exception as e:
            print(f"  路网加载失败,将不使用道路约束: {e}")

    def _is_car_accessible(lon, lat):
        """检查点是否在道路可达范围内"""
        if road_buffer_utm is None:
            return True, lon, lat  # 无路网数据时不做约束
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
        x, y = transformer.transform(lon, lat)
        pt_utm = Point(x, y)
        accessible = road_buffer_utm.contains(pt_utm)
        if accessible:
            return True, lon, lat
        # 不在缓冲区内,吸附到最近道路
        snap_lon, snap_lat, _ = _snap_to_nearest_road(lon, lat, roads_wgs84, road_points_cache)
        return False, snap_lon, snap_lat

    # 1. 盲区网格质心(最高优先级) — 需通过道路约束
    snapped_count = 0
    if len(blind_spots) > 0:
        for _, row in blind_spots.iterrows():
            if len(candidates) >= num_candidates:
                break
            lon, lat = row['lon'], row['lat']
            accessible, use_lon, use_lat = _is_car_accessible(lon, lat)
            if not accessible:
                snapped_count += 1
            if _far_enough(use_lon, use_lat):
                candidates.append((use_lon, use_lat))

    if snapped_count > 0:
        print(f"  盲区吸附: {snapped_count}个点不在路网范围内,已吸附至最近道路")

    # 2. 路网采样(中优先级) — 天然可通车
    if len(candidates) < num_candidates and road_points_cache:
        np.random.shuffle(road_points_cache)
        for lon, lat in road_points_cache:
            if len(candidates) >= num_candidates:
                break
            if _far_enough(lon, lat):
                candidates.append((lon, lat))

    # 3. 边界内随机采样 — 必须落在路网缓冲区内
    if len(candidates) < num_candidates:
        if boundary is not None and len(boundary) > 0:
            bounds = boundary.total_bounds
            union_geom = unary_union(boundary.geometry.tolist())
            attempts = 0
            max_attempts = num_candidates * 20  # 增加尝试次数,因为路网约束过滤更严
            while len(candidates) < num_candidates and attempts < max_attempts:
                lon = np.random.uniform(bounds[0], bounds[2])
                lat = np.random.uniform(bounds[1], bounds[3])
                pt = Point(lon, lat)
                if not union_geom.contains(pt):
                    attempts += 1
                    continue
                accessible, use_lon, use_lat = _is_car_accessible(lon, lat)
                if _far_enough(use_lon, use_lat):
                    candidates.append((use_lon, use_lat))
                attempts += 1
        else:
            if len(blind_spots) > 0:
                min_lon, min_lat, max_lon, max_lat = blind_spots.total_bounds
            else:
                min_lon, min_lat, max_lon, max_lat = 107.4, 33.3, 109.6, 34.7
            max_attempts = num_candidates * 20
            attempts = 0
            while len(candidates) < num_candidates and attempts < max_attempts:
                lon = np.random.uniform(min_lon, max_lon)
                lat = np.random.uniform(min_lat, max_lat)
                accessible, use_lon, use_lat = _is_car_accessible(lon, lat)
                if _far_enough(use_lon, use_lat):
                    candidates.append((use_lon, use_lat))
                attempts += 1

    candidates = pd.DataFrame(candidates[:num_candidates], columns=['lon', 'lat'])
    candidates['candidate_id'] = range(len(candidates))
    road_status = "路网约束(150m缓冲)" if road_buffer_utm is not None else "无路网约束"
    print(f"  候选点: {len(candidates)}个, 来源=盲区吸附+路网采样+边界约束, {road_status}")

    return candidates

def _transit_proximity(demand_points, coords, radius_m=2000):
    """
    计算坐标点的交通枢纽邻近性 — 公交/地铁/车站等交通POI密度的空间邻近程度

    参数:
        demand_points: 需求网格GeoDataFrame(需含 transit_density 列, 0-1)
        coords: 待评估坐标 (n, 2) 经纬度
        radius_m: 邻近搜索半径(米)

    返回:
        (n,) 数组, 每个点在半径内网格交通密度均值(0-1); 无 transit_density 时返回全0
    """
    if 'transit_density' not in demand_points.columns:
        return np.zeros(len(coords))
    from scipy.spatial import cKDTree
    transit_vals = demand_points['transit_density'].values.astype(float)
    grid_coords = demand_points[['lon', 'lat']].values
    lon0, lat0 = grid_coords[:, 0].mean(), grid_coords[:, 1].mean()
    m_lon = 111320.0 * np.cos(np.radians(lat0))
    m_lat = 110574.0
    grid_xy = np.column_stack([
        (grid_coords[:, 0] - lon0) * m_lon,
        (grid_coords[:, 1] - lat0) * m_lat,
    ])
    tree = cKDTree(grid_xy)
    out = np.zeros(len(coords))
    for i in range(len(coords)):
        x = (coords[i, 0] - lon0) * m_lon
        y = (coords[i, 1] - lat0) * m_lat
        nbrs = tree.query_ball_point([x, y], radius_m)
        if nbrs:
            out[i] = float(np.mean(transit_vals[nbrs]))
    return out


def greedy_p_median(demand_points, candidates, existing_sites, k=5, min_distance=500, coverage_weight=0.7, transit_weight=0.0):
    """
    多目标贪婪P-Median算法 (覆盖率 + 公平性 + 可选交通枢纽覆盖)

    参数:
        demand_points: 需求点DataFrame(需包含 demand 列;交通导向时还需 transit_density 列)
        candidates: 候选点DataFrame
        existing_sites: 现有站点DataFrame
        k: 新增站点数量
        min_distance: 与现有站点最小距离
        coverage_weight: 覆盖率权重 (0-1),剩余权重给公平性(1-Gini)
        transit_weight: 交通枢纽覆盖权重 (0-1),>0 时加入第三目标

    返回:
        选中的站点DataFrame
    """
    demand_coords = demand_points[['lon', 'lat']].values
    candidate_coords = candidates[['lon', 'lat']].values
    existing_coords = existing_sites[['lon', 'lat']].values

    # 计算候选点到现有站点的距离
    dist_candidate_existing = cdist_haversine(candidate_coords, existing_coords)
    min_dist_to_existing = dist_candidate_existing.min(axis=1)

    # 筛选有效候选点
    valid_mask = np.where(min_dist_to_existing >= min_distance)[0]

    if len(valid_mask) < k:
        print(f"警告:有效候选点不足{k}个,仅使用{len(valid_mask)}个有效候选点")

    valid_candidates = candidate_coords[valid_mask]
    valid_indices = valid_mask

    # 计算需求点到候选点的距离
    dist_demand_candidate = cdist_haversine(demand_coords, valid_candidates)
    weights = demand_points['demand'].values

    # 候选点交通邻近性 — 交通导向第三目标: 距公交/地铁/车站等枢纽越近得分越高
    use_transit = transit_weight > 0 and 'transit_density' in demand_points.columns
    candidate_transit = _transit_proximity(demand_points, candidate_coords) if use_transit else None

    def _minmax(a):
        """min-max归一化到[0,1]; 量纲全等时返回0,避免除零"""
        a = np.asarray(a, dtype=float)
        rng = a.max() - a.min()
        if rng < 1e-12:
            return np.zeros_like(a)
        return (a - a.min()) / rng

    # 贪婪选择(多目标) — 各目标先归一化到[0,1]再加权,避免量纲差异(覆盖率~1e-3 vs 公平性~0.7)相互淹没
    selected_indices = []
    selected_coords = []

    for _ in range(min(k, len(valid_candidates))):
        best_idx = -1
        best_score = -np.inf

        # 先算本轮每个剩余候选的三个目标值
        cand_pos = [i for i in range(len(valid_candidates)) if i not in selected_indices]
        cov_raw, eq_raw, tr_raw = [], [], []
        for i in cand_pos:
            temp_selected = selected_coords + [valid_candidates[i]]
            temp_coords = np.array(temp_selected)
            min_dists = cdist_haversine(demand_coords, temp_coords).min(axis=1)
            # 目标1:加权覆盖率
            cov_raw.append(np.sum(weights * (min_dists <= COVER_THRESHOLD)) / (np.sum(weights) + 1e-10))
            # 目标2:距离公平性(1 - 距离Gini)
            eq_raw.append(1.0 - _quick_gini(min_dists))
            # 目标3:交通枢纽邻近性(候选点自身, 与已选集合无关)
            tr_raw.append(candidate_transit[valid_indices[i]] if candidate_transit is not None else 0.0)

        cov_n = _minmax(np.array(cov_raw))
        eq_n = _minmax(np.array(eq_raw))
        tr_n = _minmax(np.array(tr_raw)) if use_transit else np.zeros(len(cov_raw))

        for j, i in enumerate(cand_pos):
            if use_transit:
                composite = (1 - transit_weight) * (coverage_weight * cov_n[j] + (1 - coverage_weight) * eq_n[j]) + transit_weight * tr_n[j]
            else:
                composite = coverage_weight * cov_n[j] + (1 - coverage_weight) * eq_n[j]
            if composite > best_score:
                best_score = composite
                best_idx = i

        if best_idx != -1:
            selected_indices.append(best_idx)
            selected_coords.append(valid_candidates[best_idx])

    selected_candidates = candidates.iloc[valid_indices[selected_indices]].copy()

    return selected_candidates


def _quick_gini(values):
    """快速Gini系数计算"""
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0
    return (2 * np.sum(np.arange(1, n + 1) * v) - (n + 1) * np.sum(v)) / (n * np.sum(v))


def calculate_marginal_benefit(grid, candidates, charging_gdf, max_k=20, min_distance=500):
    """
    边际效益曲线 — 计算k=1→max_k每新增一站点的覆盖增益

    返回:
        DataFrame: k, coverage, marginal_gain, avg_distance, gini 逐站变化
    """
    demand_coords = grid[['lon', 'lat']].values
    weights = grid['demand'].values
    existing_coords = charging_gdf[['lon', 'lat']].values
    candidate_coords = candidates[['lon', 'lat']].values

    # 计算现有站点基线
    if len(existing_coords) > 0:
        dists_existing = cdist_haversine(demand_coords, existing_coords)
        covered_before = (dists_existing.min(axis=1) <= COVER_THRESHOLD).astype(float)
    else:
        covered_before = np.zeros(len(demand_coords))

    baseline_coverage = np.sum(weights * covered_before) / (np.sum(weights) + 1e-10)

    # 逐个添加站点,记录边际增益
    selected_mask = np.zeros(len(candidates), dtype=bool)
    results = []
    all_covered = covered_before.copy()

    for k in range(1, max_k + 1):
        best_idx = -1
        best_marginal = -1
        best_covered = None

        for i in range(len(candidates)):
            if selected_mask[i]:
                continue
            new_dists = cdist_haversine(demand_coords, [candidate_coords[i]])
            new_covered = np.maximum(all_covered, (new_dists[:, 0] <= COVER_THRESHOLD).astype(float))
            marginal = np.sum(weights * (new_covered - all_covered))
            if marginal > best_marginal:
                best_marginal = marginal
                best_idx = i
                best_covered = new_covered

        if best_idx == -1:
            break

        selected_mask[best_idx] = True
        all_covered = best_covered

        total_coverage = np.sum(weights * all_covered) / (np.sum(weights) + 1e-10)

        # 计算当前距离和Gini
        sel_coords = candidate_coords[selected_mask]
        all_coords = np.vstack([existing_coords, sel_coords]) if len(existing_coords) > 0 else sel_coords
        min_dists = cdist_haversine(demand_coords, all_coords).min(axis=1)
        avg_dist = np.average(min_dists, weights=weights)
        gini = _quick_gini(min_dists)

        results.append({
            'k': k,
            '覆盖率': round(total_coverage * 100, 1),
            '边际增益(%)': round(best_marginal / (np.sum(weights) + 1e-10) * 100, 2),
            '累计增益(%)': round((total_coverage - baseline_coverage) * 100, 1),
            '平均距离(m)': round(avg_dist, 0),
            '距离Gini': round(gini, 3),
        })

    return pd.DataFrame(results)


def run_scenario_comparison(grid, candidates, charging_gdf, k=5, min_distance=500):
    """
    多情景政策对比 — 4种政策取向下选址方案的效果比较

    情景定义:
        1. 公平优先 (Equity-first):  coverage_weight=0.1, 优先缩小区域差距
        2. 效率优先 (Efficiency-first): coverage_weight=0.9, 优先最大化服务覆盖
        3. 交通导向 (Transit-oriented): 增加交通枢纽覆盖目标,优先覆盖公交/地铁密集区
        4. 均衡发展 (Balanced):         coverage_weight=0.5, 覆盖与公平同等权重

    返回:
        DataFrame: 各情景的评估指标对比表
        dict: {scenario_name: selected_sites_df}
    """
    SCENARIOS = {
        '公平优先': {'coverage_weight': 0.1, 'desc': '优先缩小区域充电服务差距'},
        '效率优先': {'coverage_weight': 0.9, 'desc': '优先最大化总体充电覆盖'},
        '交通导向': {'coverage_weight': 0.5, 'transit_weight': 0.3, 'desc': '兼顾公交枢纽与高需求区'},
        '均衡发展': {'coverage_weight': 0.5, 'desc': '覆盖与公平同等权重'},
    }

    demand_coords = grid[['lon', 'lat']].values
    weights = grid['demand'].values
    existing_coords = charging_gdf[['lon', 'lat']].values

    results = []
    all_selections = {}

    for name, params in SCENARIOS.items():
        coverage_weight = params['coverage_weight']
        transit_weight = params.get('transit_weight', 0.0)
        selected = greedy_p_median(
            grid, candidates, charging_gdf,
            k=k, min_distance=min_distance,
            coverage_weight=coverage_weight,
            transit_weight=transit_weight
        )

        all_selections[name] = selected

        # 计算评估指标
        if len(selected) > 0:
            sel_coords = selected[['lon', 'lat']].values
            all_coords = np.vstack([existing_coords, sel_coords])
            dists = cdist_haversine(demand_coords, all_coords)
            min_dists = dists.min(axis=1)

            coverage = np.sum(weights * (min_dists <= COVER_THRESHOLD)) / (np.sum(weights) + 1e-10)
            avg_dist = np.average(min_dists, weights=weights)
            p95_dist = np.percentile(min_dists, 95)
            gini = _quick_gini(min_dists)
            equity = 1.0 - gini
            # 交通枢纽邻近性: 新增站点周边交通POI密度均值(0-1)
            transit_mean = float(np.mean(_transit_proximity(grid, sel_coords)))
        else:
            coverage = avg_dist = p95_dist = gini = equity = transit_mean = 0.0

        results.append({
            'name': name,
            '政策说明': params['desc'],
            'coverage_weight': coverage_weight,
            'transit_weight': transit_weight,
            'coverage': coverage,
            'avg_dist': avg_dist,
            'p95_dist': p95_dist,
            'gini': gini,
            'equity': equity,
            'transit_mean': transit_mean,
            'n_sites': len(selected),
        })

    # 交通邻近性跨情景归一化到[0,1],供交通导向情景的综合得分作为第三目标
    tr_vals = np.array([r['transit_mean'] for r in results])
    tr_norm = (tr_vals - tr_vals.min()) / (tr_vals.max() - tr_vals.min() + 1e-12) if tr_vals.max() > tr_vals.min() else np.zeros_like(tr_vals)

    comparison_rows = []
    for r, trn in zip(results, tr_norm):
        cw, tw = r['coverage_weight'], r['transit_weight']
        if tw > 0:
            composite = (1 - tw) * (cw * r['coverage'] + (1 - cw) * r['equity']) + tw * trn
        else:
            composite = cw * r['coverage'] + (1 - cw) * r['equity']
        comparison_rows.append({
            '情景': r['name'],
            '政策说明': r['政策说明'],
            '覆盖率权重': cw,
            '公平性权重': round(1 - cw, 1),
            '交通权重': tw,
            '加权覆盖率': round(r['coverage'] * 100, 1),
            '平均距离(m)': round(r['avg_dist'], 0),
            'P95距离(m)': round(r['p95_dist'], 0),
            '距离Gini': round(r['gini'], 3),
            '公平性得分': round(r['equity'], 3),
            '交通邻近性(x1000)': round(r['transit_mean'] * 1000, 2),
            '综合得分': round(composite, 3),
            '新增站点数': r['n_sites'],
        })

    comparison_df = pd.DataFrame(comparison_rows)
    return comparison_df, all_selections


def main():
    """
    主函数:选址优化
    """
    print("=" * 50)
    print("选址优化开始")
    print("=" * 50)
    
    print("读取边界数据...")
    boundary_file = DISTRICT_GEOJSON if DISTRICT_GEOJSON.exists() else BOUNDARY_FILE
    boundary = read_geojson_without_fiona(boundary_file)
    
    print("读取充电站数据...")
    charging_gdf = read_geojson_without_fiona(DATA_DIR / "charging_stations_xian.geojson")
    
    print("读取服务盲区...")
    blind_spots = read_geojson_without_fiona(OUTPUT_DIR / "service_blind_spots.geojson")
    
    print("读取需求网格...")
    grid = read_geojson_without_fiona(OUTPUT_DIR / "grid_supply_demand.geojson")
    
    print("生成候选点...")
    candidates = generate_candidates(blind_spots, charging_gdf, boundary=boundary)
    print(f"候选点数量: {len(candidates)}")
    
    print("贪婪P-Median选址优化...")
    selected_sites = greedy_p_median(grid, candidates, charging_gdf, k=NEW_SITES_COUNT, min_distance=MIN_DISTANCE_EXISTING)
    print(f"选中站点数量: {len(selected_sites)}")
    
    selected_sites['ports'] = 6
    selected_sites.to_csv(OUTPUT_DIR / "new_sites.csv", index=False, encoding='utf-8-sig')
    print(f"输出新增站点: {OUTPUT_DIR / 'new_sites.csv'}")

    # 多情景政策对比
    print("多情景政策对比...")
    comparison_df, all_selections = run_scenario_comparison(
        grid, candidates, charging_gdf, k=NEW_SITES_COUNT, min_distance=MIN_DISTANCE_EXISTING
    )
    comparison_df.to_csv(OUTPUT_DIR / "scenario_comparison.csv", index=False, encoding='utf-8-sig')
    print(f"输出情景对比: {OUTPUT_DIR / 'scenario_comparison.csv'}")
    print(comparison_df.to_string(index=False))

    # 边际效益曲线
    print("边际效益分析...")
    marginal_df = calculate_marginal_benefit(grid, candidates, charging_gdf, max_k=20)
    marginal_df.to_csv(OUTPUT_DIR / "marginal_benefit.csv", index=False, encoding='utf-8-sig')
    print(f"输出边际效益: {OUTPUT_DIR / 'marginal_benefit.csv'}")
    if len(marginal_df) > 0:
        print(f"  从k=1到{len(marginal_df)}: 覆盖率 {marginal_df['覆盖率'].iloc[0]:.1f}% → {marginal_df['覆盖率'].iloc[-1]:.1f}%")
    else:
        print("  候选点为空，无法计算边际效益")
    
    print("绘制优化结果图...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    boundary.plot(ax=ax, color='lightgray', edgecolor='black')
    grid.plot(ax=ax, column='score_norm', cmap='RdYlBu_r', alpha=0.5, legend=False)
    charging_gdf.plot(ax=ax, marker='o', color='blue', markersize=10, label='现有站点')
    
    if len(selected_sites) > 0:
        selected_sites.plot(ax=ax, x='lon', y='lat', marker='*', color='red', markersize=150, label='新增站点')
    
    plt.title('西安市充电站选址优化结果图', fontsize=14)
    plt.legend()
    plt.axis('off')
    plt.savefig(OUTPUT_DIR / "optimization_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出优化图: {OUTPUT_DIR / 'optimization_map.png'}")
    
    print("=" * 50)
    print("选址优化完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
