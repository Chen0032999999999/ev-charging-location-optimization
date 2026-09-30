import os
from pathlib import Path

# ── 中文字体配置(matplotlib) ──────────────────────────────────
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_POI_FILE = DATA_DIR / "西安市.csv"
BOUNDARY_FILE = DATA_DIR / "西安市_wgs84.geojson"
DISTRICT_GEOJSON = DATA_DIR / "西安市区县.geojson"
ROAD_FILE = DATA_DIR / "西安市_wgs84.shp"
DISTRICT_FILE = DATA_DIR / "区县.shp"
POPULATION_FILE = DATA_DIR / "陕西省_西安市_2024.shp"

SEARCH_RADIUS = [3000, 5000, 8000]
GRID_SIZE = 1000
NEW_SITES_COUNT = 5
MIN_DISTANCE_EXISTING = 500
COVER_THRESHOLD = 5000

SEED = 42

# ── 应用运行时参数（app.py 引用）──────────────────────────────
AMAP_KEY = os.environ.get("AMAP_KEY", "")   # 高德地图API密钥（环境变量读取）
UTM_EPSG = "EPSG:32649"                     # UTM Zone 49N 投影坐标系
MAX_GRID_COUNT = 15000                      # 交互式应用最大网格数
IP_TIME_LIMIT = 60                          # 整数规划求解超时（秒）
SPREAD_DISTANCE = 1500                      # 站点最小间距约束（米）
BLIND_THRESHOLD = 0.2                       # 默认盲区百分位数阈值

CHARGING_KEYWORDS = ["充电站", "充电桩", "充换电", "超充", "快充"]
OPERATOR_KEYWORDS = {
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

PLACE_TYPE_MAP = {
    "住宅": ["住宅", "小区", "居住区", "公寓", "居民点"],
    "商业": ["商场", "购物中心", "超市", "商业", "便利店"],
    "办公": ["办公", "写字楼", "大厦", "办公楼"],
    "交通": ["公交", "地铁", "车站", "机场", "停车场"],
    "工业": ["工业", "工厂", "产业园"],
    "教育": ["学校", "大学", "学院", "中学", "小学"],
    "医疗": ["医院", "诊所", "医疗", "卫生院"],
}

WEIGHTS = {
    "住宅": 0.5,
    "商业": 0.3,
    "办公": 0.2,
    "交通": 0.1,
}

COLORS = {
    "特来电": "#FF6B6B",
    "国家电网": "#4ECDC4",
    "星星充电": "#45B7D1",
    "云快充": "#96CEB4",
    "蔚来": "#FFEAA7",
    "特斯拉": "#DDA0DD",
    "小鹏": "#98D8C8",
    "理想": "#F7DC6F",
    "其他": "#95A5A6",
}

if __name__ == "__main__":
    print(f"配置加载完成:")
    print(f"  工作目录: {BASE_DIR}")
    print(f"  POI文件: {RAW_POI_FILE.name} (存在: {RAW_POI_FILE.exists()})")
    print(f"  路网文件: {ROAD_FILE.name} (存在: {ROAD_FILE.exists()})")
    print(f"  区县文件: {DISTRICT_FILE.name} (存在: {DISTRICT_FILE.exists()})")
    print(f"  人口文件: {POPULATION_FILE.name} (存在: {POPULATION_FILE.exists()})")
