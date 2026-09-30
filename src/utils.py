"""
工具函数模块
提供空间分析、数据处理等通用功能
"""
import pandas as pd
import numpy as np
import json
from shapely.geometry import Point, Polygon, box, shape, mapping
from shapely.ops import unary_union
import geopandas as gpd
from pathlib import Path

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两点之间的Haversine距离(米)
    
    参数:
        lon1, lat1: 点1的经纬度
        lon2, lat2: 点2的经纬度
    
    返回:
        距离(米)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    r = 6371000  # 地球半径(米)
    return c * r

def cdist_haversine(coords1, coords2):
    """
    计算两组坐标之间的距离矩阵(米),纯NumPy向量化实现

    参数:
        coords1: 第一组坐标,shape (n1, 2)
        coords2: 第二组坐标,shape (n2, 2)

    返回:
        距离矩阵,shape (n1, n2)
    """
    coords1 = np.asarray(coords1, dtype=np.float64)
    coords2 = np.asarray(coords2, dtype=np.float64)

    lon1 = np.radians(coords1[:, 0])   # (n1,)
    lat1 = np.radians(coords1[:, 1])   # (n1,)
    lon2 = np.radians(coords2[:, 0])   # (n2,)
    lat2 = np.radians(coords2[:, 1])   # (n2,)

    dlon = lon1[:, np.newaxis] - lon2[np.newaxis, :]   # (n1, n2)
    dlat = lat1[:, np.newaxis] - lat2[np.newaxis, :]   # (n1, n2)

    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1[:, np.newaxis])
         * np.cos(lat2[np.newaxis, :])
         * np.sin(dlon / 2) ** 2)

    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371000 * c

def read_geojson_without_fiona(filepath):
    """
    不使用fiona读取GeoJSON文件
    
    参数:
        filepath: 文件路径
    
    返回:
        GeoDataFrame
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data['features']
    geometries = []
    properties = []
    
    for feature in features:
        geom = shape(feature['geometry'])
        geometries.append(geom)
        properties.append(feature['properties'])
    
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
    return gdf

def save_geojson(gdf, filepath):
    """
    保存GeoDataFrame为GeoJSON文件
    
    参数:
        gdf: GeoDataFrame
        filepath: 输出文件路径
    """
    features = []
    for _, row in gdf.iterrows():
        feature = {
            "type": "Feature",
            "properties": row.drop('geometry').to_dict(),
            "geometry": mapping(row['geometry'])
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

def create_grid(boundary, grid_size_m, max_cells=5000):
    """
    创建渔网网格
    
    参数:
        boundary: 边界GeoDataFrame
        grid_size_m: 网格大小(米)
        max_cells: 最大网格数量
    
    返回:
        网格GeoDataFrame
    """
    bounds = boundary.total_bounds
    xmin, ymin, xmax, ymax = bounds

    # 将米转换为经纬度单位（经度方向按平均纬度 cos 修正，保持网格近似正方形）
    mid_lat = (ymin + ymax) / 2.0
    cell_size_lon = grid_size_m / (111000 * np.cos(np.radians(mid_lat)))
    cell_size_lat = grid_size_m / 111000

    x_coords = np.arange(xmin, xmax, cell_size_lon)
    y_coords = np.arange(ymin, ymax, cell_size_lat)

    boundary_geom = unary_union(boundary.geometry.tolist())

    polygons = []
    centroids = []

    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            cell = box(x, y, x + cell_size_lon, y + cell_size_lat)
            if cell.intersects(boundary_geom):
                polygons.append(cell)
                centroids.append((x + cell_size_lon / 2, y + cell_size_lat / 2))

    # 若超过最大数量，均匀抽样以覆盖整个区域（而非截断在西南角）
    if len(polygons) > max_cells:
        idx = np.linspace(0, len(polygons) - 1, max_cells).astype(int)
        polygons = [polygons[i] for i in idx]
        centroids = [centroids[i] for i in idx]

    grid = gpd.GeoDataFrame({
        'geometry': polygons,
        'lon': [c[0] for c in centroids],
        'lat': [c[1] for c in centroids]
    }, crs="EPSG:4326")
    grid['grid_id'] = range(len(grid))

    return grid

def simple_pca(data, n_components=1):
    """
    简单PCA实现(不依赖sklearn)
    
    参数:
        data: 数据矩阵,shape (n_samples, n_features)
        n_components: 主成分数量
    
    返回:
        主成分得分
    """
    data_centered = data - data.mean(axis=0)
    cov_matrix = np.cov(data_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    principal_component = eigenvectors[:, 0]
    scores = data_centered @ principal_component
    return scores.reshape(-1, 1)

def calculate_gini(values):
    """
    计算基尼系数
    
    参数:
        values: 数值数组
    
    返回:
        基尼系数
    """
    values = np.sort(values)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
    return gini

def df_to_markdown(df, floatfmt='.2f'):
    """
    将DataFrame转换为markdown表格
    
    参数:
        df: DataFrame
        floatfmt: 浮点数格式
    
    返回:
        markdown表格字符串
    """
    if len(df) == 0:
        return "无数据"
    
    lines = []
    lines.append("| " + " | ".join(str(col) for col in df.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in df.columns) + " |")
    
    for _, row in df.iterrows():
        row_values = []
        for val in row.values:
            if isinstance(val, float):
                row_values.append(f"{val:{floatfmt}}")
            else:
                row_values.append(str(val))
        lines.append("| " + " | ".join(row_values) + " |")
    
    return "\n".join(lines)


# ── 共享数据处理函数 ──────────────────────────────────────

# 运营商关键词（与 config.py 保持同步的副本，避免循环导入）
_OPERATOR_KEYWORDS = {
    "特来电": ["特来电"],
    "国家电网": ["国家电网", "国网", "SGCC"],
    "南方电网": ["南方电网", "南网"],
    "星星充电": ["星星充电", "万帮"],
    "云快充": ["云快充"],
    "蔚来": ["蔚来", "NIO"],
    "特斯拉": ["特斯拉", "Tesla"],
    "小鹏": ["小鹏", "XPeng"],
    "理想": ["理想", "Li Auto"],
}

_PLACE_TYPE_MAP = {
    "住宅": ["住宅", "小区", "居住区", "公寓", "居民点"],
    "商业": ["商场", "购物中心", "超市", "商业", "便利店"],
    "办公": ["办公", "写字楼", "大厦", "办公楼"],
    "交通": ["公交", "地铁", "车站", "机场", "停车场"],
    "工业": ["工业", "工厂", "产业园"],
    "教育": ["学校", "大学", "学院", "中学", "小学"],
    "医疗": ["医院", "诊所", "医疗", "卫生院"],
}


def extract_operator(name, type_str=""):
    """从名称和类型中提取运营商信息"""
    text = str(name) + str(type_str)
    for operator, keywords in _OPERATOR_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return operator
    return "其他"


def extract_place_type(type_str):
    """从类型字符串中提取场所类型"""
    type_str = str(type_str)
    for place_type, keywords in _PLACE_TYPE_MAP.items():
        for kw in keywords:
            if kw in type_str:
                return place_type
    return "其他"


def detect_columns(df):
    """自动检测DataFrame中的经纬度、名称、类型列"""
    mapping = {}
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        if '经度' in col_str or 'lon' in col_lower or 'long' in col_lower:
            mapping['lon'] = col
        elif '纬度' in col_str or 'lat' in col_lower:
            mapping['lat'] = col
        elif '名称' in col_str or 'name' in col_lower:
            mapping['name'] = col
        elif '类型' in col_str or '分类' in col_str or '大类' in col_str or '中类' in col_str:
            if 'type_cols' not in mapping:
                mapping['type_cols'] = []
            mapping['type_cols'].append(col)
    return mapping


def get_district_column(gdf):
    """从GeoDataFrame中自动检测区县列名"""
    if '区县' in gdf.columns:
        return '区县'
    elif '名称' in gdf.columns:
        return '名称'
    else:
        return gdf.columns[0]


def compute_road_network_distances(origins, destinations, place="西安市, China",
                                    network_type='drive', cache_path=None):
    """
    使用 OSMnx 下载路网并计算最短路径距离 (替代直线距离)

    参数:
        origins: 起点坐标数组 (n_origins, 2) [lon, lat]
        destinations: 终点坐标数组 (n_dests, 2) [lon, lat]
        place: 地名,用于下载路网
        network_type: 'drive' | 'walk' | 'bike'
        cache_path: 路网缓存 GeoJSON 路径 (避免重复下载)

    返回:
        distance_matrix: (n_origins, n_dests) 路网距离矩阵(米)
        None if osmnx 不可用或下载失败
    """
    try:
        import osmnx as ox
    except ImportError:
        print("警告: osmnx 未安装,回退到直线距离")
        return None

    import time
    t0 = time.time()

    # 读取缓存或下载路网
    G = None
    if cache_path and Path(cache_path).exists():
        try:
            G = ox.io.load_graphml(cache_path)
            print(f"  从缓存加载路网: {cache_path}")
        except Exception:
            pass

    if G is None:
        try:
            print(f"  正在下载 {place} 路网 (type={network_type})...")
            G = ox.graph_from_place(place, network_type=network_type, simplify=True)
            if cache_path:
                ox.io.save_graphml(G, cache_path)
                print(f"  路网已缓存: {cache_path}")
        except Exception as e:
            print(f"  路网下载失败: {e}")
            return None

    # 投影到UTM
    G_proj = ox.project_graph(G)
    print(f"  路网: {len(G_proj.nodes)} 节点, {len(G_proj.edges)} 边 ({(time.time()-t0):.1f}s)")

    # 投影坐标
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", G_proj.graph['crs'], always_xy=True)

    # 找最近节点
    def _nearest_nodes(coords):
        xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
        pts = list(zip(xs, ys))
        return ox.nearest_nodes(G_proj, [p[0] for p in pts], [p[1] for p in pts])

    orig_nodes = _nearest_nodes(origins)
    dest_nodes = _nearest_nodes(destinations)
    print(f"  最近节点映射完成 ({(time.time()-t0):.1f}s)")

    # 计算距离矩阵 — 只计算必要的OD对以避免全矩阵计算成本过高
    # 使用networkx的single_source_dijkstra批量计算
    import networkx as nx
    n_orig = len(orig_nodes)
    n_dest = len(dest_nodes)

    # 限制规模: 超过500个起点时抽样
    if n_orig > 500:
        print(f"  起点过多({n_orig}), 将抽样500个用于对比评估")
        # 不在这里抽样, 而是标记返回部分结果

    dist_matrix = np.full((n_orig, n_dest), np.nan)

    # 以目的地点为源, 反向计算到所有起点的距离
    unique_dest_nodes = list(set(dest_nodes))
    dest_to_idx = {n: i for i, n in enumerate(dest_nodes)}

    for src_node in unique_dest_nodes:
        try:
            lengths = nx.single_source_dijkstra_path_length(G_proj, src_node, weight='length')
            for oi, orig_node in enumerate(orig_nodes):
                if orig_node in lengths:
                    d = lengths[orig_node]
                    # 填充所有使用此节点的目的地
                    for di in [j for j, n in enumerate(dest_nodes) if n == src_node]:
                        if np.isnan(dist_matrix[oi, di]) or d < dist_matrix[oi, di]:
                            dist_matrix[oi, di] = d
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    nan_count = np.isnan(dist_matrix).sum()
    if nan_count > 0:
        print(f"  路网距离矩阵有 {nan_count} 个NaN, 将用直线距离填充")
        straight = cdist_haversine(origins, destinations)
        dist_matrix = np.where(np.isnan(dist_matrix), straight, dist_matrix)

    print(f"  路网距离计算完成 ({(time.time()-t0):.1f}s)")
    return dist_matrix


def temporal_demand_weights(poi_types, time_period, config_module=None):
    """
    时空需求权重 — 根据时段调整各类POI的需求权重

    参数:
        poi_types: POI类型Series
        time_period: 'morning_peak' | 'noon' | 'evening_peak' | 'night'
        config_module: 配置模块 (含 PLACE_TYPE_MAP, WEIGHTS)

    返回:
        各POI的需求权重数组
    """
    if config_module is None:
        import importlib
        config_module = importlib.import_module('config')

    # 时段特征权重 — 基于交通调查典型出行目的分布
    # morning_peak: 通勤(办公/住宅)权重高
    # noon: 餐饮/商业权重高
    # evening_peak: 商业/娱乐权重高
    # night: 住宅权重高, 商业降低
    TEMPORAL_MODIFIERS = {
        'morning_peak': {
            '住宅': 1.5, '办公': 2.0, '交通': 1.8,
            '商业': 0.5, '餐饮': 0.4, '娱乐': 0.2,
            '教育': 1.8, '医疗': 1.0, '公园': 0.3
        },
        'noon': {
            '住宅': 0.5, '办公': 0.8, '交通': 1.2,
            '商业': 1.8, '餐饮': 2.0, '娱乐': 0.6,
            '教育': 1.0, '医疗': 1.5, '公园': 1.0
        },
        'evening_peak': {
            '住宅': 0.6, '办公': 0.3, '交通': 1.5,
            '商业': 2.0, '餐饮': 1.8, '娱乐': 1.5,
            '教育': 0.5, '医疗': 0.8, '公园': 1.2
        },
        'night': {
            '住宅': 2.0, '办公': 0.1, '交通': 0.4,
            '商业': 0.3, '餐饮': 0.5, '娱乐': 0.8,
            '教育': 0.1, '医疗': 0.3, '公园': 0.2
        },
    }

    modifiers = TEMPORAL_MODIFIERS.get(time_period, {})

    # 获取PLACE_TYPE_MAP的反向映射

    PLACE_TYPE_MAP = getattr(config_module, 'PLACE_TYPE_MAP', {})
    WEIGHTS = getattr(config_module, 'WEIGHTS', {})

    weights = np.ones(len(poi_types))
    for i, pt in enumerate(poi_types):
        for place_type, keywords in PLACE_TYPE_MAP.items():
            if any(kw in str(pt) for kw in keywords):
                base_w = WEIGHTS.get(place_type, 1.0)
                mod = modifiers.get(place_type, 1.0)
                weights[i] = base_w * mod
                break

    return weights
