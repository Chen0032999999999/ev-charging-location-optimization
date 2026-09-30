"""
POI数据预处理模块
从西安市.csv中筛选充电站并提取相关信息
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import importlib

sys.path.append(str(Path(__file__).parent))
config = importlib.import_module('config')
from utils import extract_operator, extract_place_type  # noqa: E402

SEED = config.SEED
RAW_POI_FILE = config.RAW_POI_FILE
DATA_DIR = config.DATA_DIR
CHARGING_KEYWORDS = config.CHARGING_KEYWORDS

np.random.seed(SEED)

def detect_columns(df):
    """
    自动检测数据列名
    
    参数:
        df: DataFrame
    
    返回:
        列名映射字典
    """
    col_mapping = {}
    type_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if 'lon' in col_lower or '经度' in str(col) or 'long' in col_lower:
            col_mapping['lon'] = col
        elif 'lat' in col_lower or '纬度' in str(col):
            col_mapping['lat'] = col
        elif 'type' in col_lower or '类型' in str(col) or '分类' in str(col) or '大类' in str(col) or '中类' in str(col):
            type_cols.append(col)
        elif 'name' in col_lower or '名称' in str(col):
            col_mapping['name'] = col
        elif 'district' in col_lower or '区县' in str(col) or '地级市' in str(col):
            col_mapping['district'] = col
        elif 'adcode' in col_lower or 'code' in col_lower:
            col_mapping['adcode'] = col
    if type_cols:
        col_mapping['type'] = type_cols
    return col_mapping

def estimate_ports(name, type_str, operator, np_random):
    """
    基于站点名称、类型和运营商估算充电端口数

    参数:
        name: 站点名称
        type_str: 类型字符串
        operator: 运营商
        np_random: 带种子的随机状态

    返回:
        估算端口数 (int)
    """
    text = str(name) + str(type_str)

    # 运营商基准端口数(基于行业公开数据)
    OPERATOR_BASE = {
        "特来电": (8, 2),    # 行业龙头,站点规模偏大
        "国家电网": (6, 2),
        "星星充电": (7, 2),
        "云快充": (5, 2),
        "蔚来": (4, 1),      # 换电+超充组合
        "特斯拉": (6, 2),    # 超充站为主
        "小鹏": (4, 2),
        "理想": (3, 1),
        "南方电网": (5, 2),
        "其他": (4, 2),
    }

    # 类型关键词修正系数
    if "超充" in text:
        multiplier, extra = 1.8, 4
    elif "快充" in text:
        multiplier, extra = 1.3, 2
    elif "慢充" in text:
        multiplier, extra = 0.6, -1
    elif "换电" in text or "换电站" in text:
        multiplier, extra = 0.4, -1
    elif "充电桩" in text and "充电站" not in text:
        multiplier, extra = 0.3, -2
    else:
        multiplier, extra = 1.0, 0

    base, std = OPERATOR_BASE.get(operator, (4, 2))
    raw = base * multiplier + extra + np_random.normal(0, std * 0.5)
    ports = int(round(max(1, min(24, raw))))

    return ports

def main():
    """
    主函数:读取POI数据,筛选充电站,清洗数据并输出
    """
    print("=" * 50)
    print("POI数据预处理开始")
    print("=" * 50)
    
    if not RAW_POI_FILE.exists():
        print(f"错误:POI文件不存在 {RAW_POI_FILE}")
        return
    
    print("读取POI数据...")
    df = pd.read_csv(RAW_POI_FILE, encoding='utf-8-sig')
    print(f"原始数据行数: {len(df)}")
    
    col_mapping = detect_columns(df)
    print(f"检测到的列映射: {col_mapping}")
    
    lon_col = col_mapping.get('lon', '经度')
    lat_col = col_mapping.get('lat', '纬度')
    type_cols = col_mapping.get('type', ['大类', '中类'])
    name_col = col_mapping.get('name', '名称')
    district_col = col_mapping.get('district', '区县')
    
    df['lon'] = df[lon_col]
    df['lat'] = df[lat_col]
    df['name'] = df[name_col]
    df['区县'] = df[district_col]
    
    if isinstance(type_cols, list):
        df['type'] = df[type_cols].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    else:
        df['type'] = df[type_cols]
    
    print("筛选充电站...")
    charging_mask = df['type'].fillna('').str.contains('|'.join(CHARGING_KEYWORDS)) | \
                    df['name'].fillna('').str.contains('|'.join(CHARGING_KEYWORDS))
    charging_df = df[charging_mask].copy()
    
    print(f"筛选后充电站数量: {len(charging_df)}")
    
    print("清洗数据...")
    charging_df = charging_df.dropna(subset=['lon', 'lat'])
    charging_df = charging_df[(charging_df['lon'] >= 107.4) & (charging_df['lon'] <= 109.6) &
                              (charging_df['lat'] >= 33.3) & (charging_df['lat'] <= 34.7)]
    
    print(f"清洗后充电站数量: {len(charging_df)}")
    
    print("提取运营商信息...")
    charging_df['operator'] = charging_df.apply(lambda row: extract_operator(row['name'], row['type']), axis=1)
    
    print("估算充电端口数...")
    rng = np.random.RandomState(SEED)
    charging_df['ports'] = charging_df.apply(
        lambda row: estimate_ports(row['name'], row['type'], row['operator'], rng), axis=1
    )
    print(f"  端口分布: min={charging_df['ports'].min()}, max={charging_df['ports'].max()}, "
          f"mean={charging_df['ports'].mean():.1f}, median={charging_df['ports'].median():.0f}")
    
    print("提取场所类型...")
    charging_df['place_type'] = charging_df['type'].apply(extract_place_type)
    
    charging_df.to_csv(DATA_DIR / "charging_stations_clean.csv", index=False, encoding='utf-8-sig')

    print(f"输出文件: {DATA_DIR / 'charging_stations_clean.csv'}")
    print("=" * 50)
    print("POI预处理完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
