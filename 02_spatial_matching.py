"""
空间匹配模块
将充电站与西安市边界进行空间匹配,输出GeoJSON文件并绘制分布图
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from shapely.geometry import Point, mapping
import sys
import importlib
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
config = importlib.import_module('config')
from utils import read_geojson_without_fiona, save_geojson

BOUNDARY_FILE = config.BOUNDARY_FILE
DATA_DIR = config.DATA_DIR
OUTPUT_DIR = config.OUTPUT_DIR
COLORS = config.COLORS
DISTRICT_FILE = config.DISTRICT_FILE
DISTRICT_GEOJSON = config.DISTRICT_GEOJSON

def load_xian_boundary():
    """
    加载西安市边界
    优先使用西安市区县.geojson(包含详细区县信息)

    返回:
        西安市各区县边界GeoDataFrame
    """
    print("加载西安市边界...")

    # 首先尝试加载西安市区县.geojson(包含详细区县信息)
    district_geojson = DISTRICT_GEOJSON
    if district_geojson.exists():
        try:
            print(f"加载西安市区县.geojson...")
            districts = gpd.read_file(district_geojson, encoding='utf-8')
            print(f"成功加载 {len(districts)} 个区县")
            print(f"区县名称: {districts['名称'].tolist()}")
            
            # 确保有正确的坐标系
            if districts.crs is None:
                districts.crs = "EPSG:4326"
            
            return districts
        except Exception as e:
            print(f"读取西安市区县.geojson失败: {e}")
    
    if DISTRICT_FILE.exists():
        print(f"尝试从区县.shp筛选西安市...")
        try:
            districts = gpd.read_file(DISTRICT_FILE)
            print(f"区县数据读取成功,共{len(districts)}条记录")
            print(f"区县数据列名: {districts.columns.tolist()}")
            
            xian_mask = None
            if 'NAME' in districts.columns:
                xian_mask = districts['NAME'].str.contains('雁塔|长安|新城|碑林|莲湖|灞桥|未央|阎良|临潼|高陵|鄠邑|蓝田')
            elif '市' in districts.columns:
                xian_mask = districts['市'] == '西安市'
            elif 'adcode' in districts.columns:
                xian_mask = districts['adcode'].astype(str).str.startswith('6101')
            elif 'city' in districts.columns:
                xian_mask = districts['city'].str.contains('西安')
            
            if xian_mask is not None and xian_mask.any():
                xian_districts = districts[xian_mask]
                print(f"成功筛选出西安市 {len(xian_districts)} 个区县")
                return xian_districts
            else:
                print("未能从区县.shp筛选出西安市,使用备用边界文件")
        except Exception as e:
            print(f"读取区县.shp失败: {e}")
    
    if BOUNDARY_FILE.exists():
        boundary = read_geojson_without_fiona(BOUNDARY_FILE)
        return boundary
    else:
        raise FileNotFoundError(f"边界文件不存在: {BOUNDARY_FILE}")

def main():
    """
    主函数:空间匹配和可视化
    """
    print("=" * 50)
    print("空间匹配开始")
    print("=" * 50)
    
    print("读取充电站数据...")
    charging_df = pd.read_csv(DATA_DIR / "charging_stations_clean.csv", encoding='utf-8-sig')
    
    print("加载西安市边界...")
    boundary = load_xian_boundary()
    print(f"边界数据投影: {boundary.crs}")
    
    print("创建充电站GeoDataFrame...")
    geometry = [Point(xy) for xy in zip(charging_df['lon'], charging_df['lat'])]
    charging_gdf = gpd.GeoDataFrame(charging_df, geometry=geometry, crs="EPSG:4326")
    
    print("空间连接筛选西安市内站点并匹配区县...")
    
    if '区县' in boundary.columns:
        district_col_in_boundary = '区县'
    elif '名称' in boundary.columns:
        district_col_in_boundary = '名称'
    else:
        district_col_in_boundary = boundary.columns[0]
    
    print(f"边界数据中的区县字段: {district_col_in_boundary}")
    
    charging_xian = gpd.sjoin(charging_gdf, boundary, how='left', predicate='within')

    joined_district_col = f"{district_col_in_boundary}_right"
    if joined_district_col in charging_xian.columns:
        charging_xian['区县'] = charging_xian[joined_district_col]
    elif district_col_in_boundary in charging_xian.columns:
        charging_xian['区县'] = charging_xian[district_col_in_boundary]

    # 去重: 区县多边形存在重叠(约14%), sjoin 'within' 会把边界附近的站点重复匹配到多个区,
    # 导致站点数量翻倍(3719→6071)。这里每个站点只保留首次匹配的区县。
    before_dedup = len(charging_xian)
    charging_xian = charging_xian[~charging_xian.index.duplicated(keep='first')]
    if len(charging_xian) < before_dedup:
        print(f"  去重: 移除 {before_dedup - len(charging_xian)} 条因区县重叠导致的重复记录")

    print(f"西安市内充电站数量: {len(charging_xian)}")
    
    unmatched_count = len(charging_gdf) - len(charging_xian[charging_xian['区县'].notna()])
    if unmatched_count > 0:
        print(f"警告: 有 {unmatched_count} 个充电站未能匹配到区县")
        print("尝试使用 'intersects' 谓词重新匹配...")
        charging_xian_intersects = gpd.sjoin(charging_gdf, boundary, how='left', predicate='intersects')
        
        intersects_district_col = f"{district_col_in_boundary}_right"
        if intersects_district_col in charging_xian_intersects.columns:
            charging_xian_intersects['区县'] = charging_xian_intersects[intersects_district_col]
        elif district_col_in_boundary in charging_xian_intersects.columns:
            charging_xian_intersects['区县'] = charging_xian_intersects[district_col_in_boundary]
        # 同样去重,避免重叠区县导致重复匹配
        charging_xian_intersects = charging_xian_intersects[~charging_xian_intersects.index.duplicated(keep='first')]
        
        matched_with_intersects = len(charging_xian_intersects[charging_xian_intersects['区县'].notna()])
        print(f"使用 intersects 匹配到的充电站数量: {matched_with_intersects}")
        
        missing_mask = charging_xian['区县'].isna()
        if missing_mask.any():
            for idx in charging_xian[missing_mask].index:
                if idx in charging_xian_intersects.index:
                    district_val = charging_xian_intersects.loc[idx, '区县']
                    if pd.notna(district_val):
                        charging_xian.loc[idx, '区县'] = district_val
            
            newly_matched = (~charging_xian['区县'].isna()).sum() - (len(charging_xian) - unmatched_count)
            if newly_matched > 0:
                print(f"额外匹配到 {newly_matched} 个充电站")
    
    missing_mask = charging_xian['区县'].isna()
    if missing_mask.any():
        print(f"对 {missing_mask.sum()} 个缺失区县信息的充电站进行最近邻匹配...")
        from scipy.spatial import cKDTree
        
        boundary_copy = boundary.copy()
        if district_col_in_boundary != '区县':
            boundary_copy['区县'] = boundary_copy[district_col_in_boundary]
        
        boundary_centroids = boundary_copy.copy()
        boundary_centroids['centroid_lon'] = boundary_centroids.geometry.centroid.x
        boundary_centroids['centroid_lat'] = boundary_centroids.geometry.centroid.y
        
        tree_data = boundary_centroids[['centroid_lon', 'centroid_lat']].values
        tree = cKDTree(tree_data)
        
        missing_stations = charging_xian[missing_mask]
        query_points = missing_stations[['lon', 'lat']].values
        distances, indices = tree.query(query_points)
        
        for idx, station_idx in enumerate(missing_stations.index):
            nearest_district = boundary_centroids.iloc[indices[idx]]['区县']
            charging_xian.loc[station_idx, '区县'] = nearest_district
        
        print(f"已为 {missing_mask.sum()} 个充电站分配区县信息")
    
    save_geojson(charging_xian, DATA_DIR / "charging_stations_xian.geojson")
    print(f"输出GeoJSON: {DATA_DIR / 'charging_stations_xian.geojson'}")
    
    if '区县' in charging_xian.columns:
        district_summary = charging_xian['区县'].value_counts()
        print("\n各区县充电站数量统计:")
        for district, count in district_summary.items():
            print(f"  {district}: {count}")
        
        all_districts_in_boundary = set(boundary[district_col_in_boundary].unique())
        districts_with_stations = set(charging_xian['区县'].dropna().unique())
        districts_without_stations = all_districts_in_boundary - districts_with_stations
        
        if districts_without_stations:
            print(f"\n以下区县没有充电站数据 ({len(districts_without_stations)}个):")
            for d in sorted(districts_without_stations):
                print(f"  - {d}")
        else:
            print("\n所有区县都有充电站数据")
    
    print("绘制静态分布图...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    boundary.plot(ax=ax, color='lightgray', edgecolor='black')
    
    for operator in charging_xian['operator'].unique():
        subset = charging_xian[charging_xian['operator'] == operator]
        subset.plot(ax=ax, 
                   marker='o', 
                   color=COLORS.get(operator, '#95A5A6'), 
                   markersize=subset['ports'] * 10,
                   alpha=0.7,
                   label=operator)
    
    plt.title('西安市充电站空间分布图(按运营商分类)', fontsize=14)
    plt.legend(title='运营商', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.savefig(OUTPUT_DIR / "distribution_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出分布图: {OUTPUT_DIR / 'distribution_map.png'}")
    
    print("绘制区县级密度图...")
    
    if '区县' in boundary.columns:
        district_col = '区县'
    elif '名称' in boundary.columns:
        district_col = '名称'
    else:
        district_col = boundary.columns[0]
    
    district_counts = charging_xian.groupby('区县').size().reset_index(name='count')
    
    all_districts_df = boundary[[district_col]].copy()
    all_districts_df = all_districts_df.rename(columns={district_col: '区县'})
    
    boundary_with_stats = all_districts_df.merge(district_counts, on='区县', how='left')
    boundary_with_stats['count'] = boundary_with_stats['count'].fillna(0).astype(int)
    
    boundary_with_stats = boundary.merge(boundary_with_stats, left_on=district_col, right_on='区县', how='left')
    boundary_with_stats['count'] = boundary_with_stats['count'].fillna(0).astype(int)
    boundary_with_stats['area_sqkm'] = boundary_with_stats.to_crs(epsg=32649).geometry.area / 1e6
    boundary_with_stats['density'] = boundary_with_stats['count'] / boundary_with_stats['area_sqkm']
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    boundary_with_stats.plot(ax=ax, column='density', cmap='YlOrRd', legend=True,
                          legend_kwds={'label': '充电站密度(个/km²)', 'orientation': 'horizontal', 'shrink': 0.7})
    boundary_with_stats.boundary.plot(ax=ax, color='black', linewidth=1.5)
    
    for idx, row in boundary_with_stats.iterrows():
        centroid = row.geometry.centroid
        district_name = row.get('区县', row.get(district_col, ''))
        count = int(row.get('count', 0))
        
        area = row['area_sqkm']
        fontsize = 8 if area > 500 else 7 if area > 200 else 6
        
        ax.text(centroid.x, centroid.y, f'{district_name}\n({count})', 
               ha='center', va='center', fontsize=fontsize, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.title('西安市各区县充电站密度分布图', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.savefig(OUTPUT_DIR / "district_choropleth.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出密度图: {OUTPUT_DIR / 'district_choropleth.png'}")
    
    print("生成交互地图...")
    center_lat = charging_xian['lat'].mean() if len(charging_xian) > 0 else 34.26
    center_lon = charging_xian['lon'].mean() if len(charging_xian) > 0 else 108.94
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    style_function = lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1}
    
    boundary_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    for _, row in boundary.iterrows():
        feature = {
            "type": "Feature",
            "properties": row.drop('geometry').to_dict(),
            "geometry": mapping(row['geometry'])
        }
        boundary_geojson["features"].append(feature)
    
    folium.GeoJson(boundary_geojson, style_function=style_function).add_to(m)
    
    for _, row in charging_xian.iterrows():
        name_val = row.get('name', row.get('名称', '未知'))
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=row['ports'] * 1.5,
            color=COLORS.get(row['operator'], '#95A5A6'),
            fill=True,
            fill_color=COLORS.get(row['operator'], '#95A5A6'),
            popup=f"<strong>{name_val}</strong><br>运营商: {row['operator']}<br>端口数: {row['ports']}"
        ).add_to(m)
    
    m.save(OUTPUT_DIR / "xian_charging_interactive.html")
    print(f"输出交互地图: {OUTPUT_DIR / 'xian_charging_interactive.html'}")
    
    print("=" * 50)
    print("空间匹配完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
