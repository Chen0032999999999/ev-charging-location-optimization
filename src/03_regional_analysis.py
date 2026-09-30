"""
区域分析模块
按区县和运营商进行统计分析,绘制相关图表
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import importlib
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
config = importlib.import_module('config')
from utils import read_geojson_without_fiona

BOUNDARY_FILE = config.BOUNDARY_FILE
DATA_DIR = config.DATA_DIR
OUTPUT_DIR = config.OUTPUT_DIR
COLORS = config.COLORS

def main():
    """
    主函数:区域分析和运营商分析
    """
    print("=" * 50)
    print("区域分析开始")
    print("=" * 50)
    
    print("读取充电站数据...")
    charging_gdf = read_geojson_without_fiona(DATA_DIR / "charging_stations_xian.geojson")
    
    print("读取边界数据...")
    district_geojson_path = config.DISTRICT_GEOJSON
    if district_geojson_path.exists():
        print(f"加载 {district_geojson_path.name}...")
        boundary = read_geojson_without_fiona(district_geojson_path)
    else:
        boundary = read_geojson_without_fiona(BOUNDARY_FILE)
        
        if '区县' not in boundary.columns and '名称' not in boundary.columns:
            print("错误: 无法找到区县信息")
            return
    
    print("按区县统计...")
    boundary['area_sqkm'] = boundary.to_crs(epsg=32649).geometry.area / 1e6
    
    if '区县' in boundary.columns:
        district_col = '区县'
    elif '名称' in boundary.columns:
        district_col = '名称'
    else:
        district_col = boundary.columns[0]
    
    print(f"使用区县字段: {district_col}")
    
    if '区县' not in charging_gdf.columns:
        print("警告: 充电站数据中没有'区县'字段,尝试从其他字段推断...")
        if '区县_left' in charging_gdf.columns:
            charging_gdf['区县'] = charging_gdf['区县_left']
        elif '区县_right' in charging_gdf.columns:
            charging_gdf['区县'] = charging_gdf['区县_right']
        else:
            print("错误: 无法找到区县字段")
            return
    
    district_counts = charging_gdf.groupby('区县').size().reset_index(name='count')
    port_stats = charging_gdf.groupby('区县').agg(
        avg_ports=('ports', 'mean'),
        total_ports=('ports', 'sum')
    ).reset_index()
    district_counts = district_counts.merge(port_stats, on='区县', how='left')
    
    all_districts = boundary[[district_col, 'area_sqkm']].copy()
    all_districts = all_districts.rename(columns={district_col: '区县'})
    
    district_stats = all_districts.merge(district_counts, on='区县', how='left')
    district_stats['count'] = district_stats['count'].fillna(0).astype(int)
    district_stats['avg_ports'] = district_stats['avg_ports'].fillna(0)
    district_stats['total_ports'] = district_stats['total_ports'].fillna(0).astype(int)
    district_stats['density'] = district_stats['count'] / district_stats['area_sqkm']
    district_stats['port_density'] = district_stats['total_ports'] / district_stats['area_sqkm']
    
    district_stats.to_csv(OUTPUT_DIR / "district_stats.csv", index=False, encoding='utf-8-sig')
    print(f"输出区县统计: {OUTPUT_DIR / 'district_stats.csv'}")
    print(f"\n区县统计摘要:")
    print(f"  总区县数: {len(district_stats)}")
    print(f"  有充电站的区县: {(district_stats['count'] > 0).sum()}")
    print(f"  无充电站的区县: {(district_stats['count'] == 0).sum()}")
    
    print(f"\n详细统计:")
    for _, row in district_stats.iterrows():
        print(f"  {row['区县']}: {row['count']}个充电站, 密度={row['density']:.4f}个/km2")
    
    print("按运营商统计...")
    op_counts = charging_gdf.groupby('operator').size().reset_index(name='count')
    op_ports = charging_gdf.groupby('operator').agg(
        total_ports=('ports', 'sum')
    ).reset_index()
    operator_stats = op_counts.merge(op_ports, on='operator', how='left')
    operator_stats['percentage'] = (operator_stats['count'] / operator_stats['count'].sum() * 100).round(2)
    operator_stats = operator_stats.sort_values('count', ascending=False)
    
    operator_stats.to_csv(OUTPUT_DIR / "operator_stats.csv", index=False, encoding='utf-8-sig')
    print(f"输出运营商统计: {OUTPUT_DIR / 'operator_stats.csv'}")
    
    print("绘制运营商站点数条形图...")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=operator_stats, x='operator', y='count', 
                palette=[COLORS.get(op, '#95A5A6') for op in operator_stats['operator']])
    plt.title('各运营商充电站数量分布', fontsize=14)
    plt.xlabel('运营商', fontsize=12)
    plt.ylabel('站点数量', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "operator_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出条形图: {OUTPUT_DIR / 'operator_bar.png'}")
    
    print("绘制端口数分布图...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=charging_gdf, x='ports', bins=range(1, 25), kde=False)
    plt.title('充电站端口数分布', fontsize=14)
    plt.xlabel('端口数', fontsize=12)
    plt.ylabel('站点数量', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ports_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出端口分布图: {OUTPUT_DIR / 'ports_distribution.png'}")
    
    print("=" * 50)
    print("区域分析完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
