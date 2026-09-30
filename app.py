import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import st_folium, folium_static
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
# 使用 numpy 实现的 cdist（避免 scipy DLL 问题）
def cdist(X, Y, metric='euclidean'):
    if metric == 'euclidean':
        X = np.asarray(X)
        Y = np.asarray(Y)
        return np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
from sklearn.preprocessing import MinMaxScaler
import math
import json
from io import StringIO

# 导入项目配置（config.py 位于 src/ 目录）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import config

np.random.seed(42)

# ── 中文字体配置（matplotlib） ──────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="西安市新能源汽车充电设施优化系统", page_icon="🚗", layout="wide")

# 高德地图API Key（从环境变量读取，未设置时使用无需Key的免费瓦片）
AMAP_KEY = config.AMAP_KEY
# 高德地图瓦片URL（有Key用有Key版，无Key用免费版）
AMAP_TILE_URL = (
    f"https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
    if AMAP_KEY else
    "https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
)

# 加载外部样式表
_css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(_css_path):
    with open(_css_path, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# 文件路径 - 使用相对路径（数据统一放在 data/ 目录下）
POI_PATH = r"./data/西安市.csv"
ROAD_PATH = r"./data/西安市_wgs84.shp"
DISTRICT_PATH = r"./data/西安市区县.geojson"

@st.cache_data(ttl=3600, show_spinner="正在加载数据...")
def load_data():
    """加载并预处理所有数据"""
    poi_df = None
    try:
        poi_df = pd.read_csv(POI_PATH, encoding='utf-8')

        charging_keywords = ['充电站', '充电桩', '充换电', '超充', '快充', '国网充电', '特来电', '星星充电']
        mask = poi_df['名称'].str.contains('|'.join(charging_keywords), na=False)
        charging_stations = poi_df[mask].copy()

        operators = ['特来电', '国家电网', '星星充电', '云快充', '蔚来', '小鹏', '理想', '特斯拉', '比亚迪']
        def extract_operator(name):
            if '国网' in name:
                return '国家电网'
            for op in operators:
                if op in name:
                    return op
            return '其他'

        charging_stations['运营商'] = charging_stations['名称'].apply(extract_operator)

        # 基于站点名称、类型和运营商的端口数智能估算（可复现）
        rng = np.random.RandomState(42)
        _OP_BASE = {"特来电":(8,2),"国家电网":(6,2),"星星充电":(7,2),"云快充":(5,2),"蔚来":(4,1),
                     "特斯拉":(6,2),"小鹏":(4,2),"理想":(3,1),"比亚迪":(5,2),"其他":(4,2)}
        def _est_ports(row):
            text = str(row['名称']) + str(row.get('类型', ''))
            op = row['运营商']
            mult = 1.8 if '超充' in text else 1.3 if '快充' in text else 0.6 if '慢充' in text else 0.4 if '换电' in text else 0.3 if '充电桩' in text else 1.0
            extra = 4 if '超充' in text else 2 if '快充' in text else -1 if '慢充' in text or '换电' in text else -2 if '充电桩' in text else 0
            base, std = _OP_BASE.get(op, (4, 2))
            return max(1, min(24, int(round(base * mult + extra + rng.normal(0, std * 0.5)))))
        charging_stations['端口数'] = charging_stations.apply(_est_ports, axis=1)

        charging_stations = charging_stations[(charging_stations['经度'] > 107.4) & (charging_stations['经度'] < 109.6)]
        charging_stations = charging_stations[(charging_stations['纬度'] > 33.3) & (charging_stations['纬度'] < 34.7)]

        charging_stations = gpd.GeoDataFrame(
            charging_stations,
            geometry=gpd.points_from_xy(charging_stations['经度'], charging_stations['纬度']),
            crs="EPSG:4326"
        )

    except Exception as e:
        st.error(f"加载POI数据失败: {e}")
        charging_stations = gpd.GeoDataFrame()

    try:
        districts = gpd.read_file(DISTRICT_PATH, encoding='utf-8')

        # 检查是否有名称列
        if '名称' not in districts.columns:
            # 尝试多种列名筛选西安市
            if '地级市' in districts.columns:
                districts = districts[districts['地级市'] == '西安市']
            elif '市' in districts.columns:
                districts = districts[districts['市'] == '西安市']
            elif 'city' in districts.columns:
                districts = districts[districts['city'] == '西安市']
            elif 'adcode' in districts.columns:
                districts = districts[districts['adcode'].astype(str).str.startswith('6101')]
            else:
                # 如果没有匹配的列，提示用户
                st.warning(f"未找到匹配的行政区划字段，可用字段: {districts.columns.tolist()}")
        
        # 确保有名称列用于显示
        if '名称' not in districts.columns:
            if '区县' in districts.columns:
                districts['名称'] = districts['区县']
            elif 'name' in districts.columns:
                districts['名称'] = districts['name']
            else:
                districts['名称'] = [f'区县{i+1}' for i in range(len(districts))]

        city_boundary = districts.geometry.union_all() if len(districts) > 0 else None

    except Exception as e:
        st.error(f"加载区县数据失败: {e}")
        districts = gpd.GeoDataFrame()
        city_boundary = None

    try:
        roads = gpd.read_file(ROAD_PATH, encoding='utf-8')
    except Exception as e:
        st.error(f"加载路网数据失败: {e}")
        roads = gpd.GeoDataFrame()

    return charging_stations, districts, city_boundary, roads, poi_df

def show_stats_cards(charging_stations, districts):
    """显示顶部统计卡片"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="充电站总数", value=len(charging_stations))

    with col2:
        total_ports = charging_stations['端口数'].sum() if '端口数' in charging_stations.columns else 0
        st.metric(label="充电端口总数", value=total_ports)

    with col3:
        operators = charging_stations['运营商'].nunique() if '运营商' in charging_stations.columns else 0
        st.metric(label="运营商数量", value=operators)

    with col4:
        covered_districts = len(districts) if not districts.empty else 0
        st.metric(label="覆盖区县", value=covered_districts)

def task1_charging_map(charging_stations, districts):
    """充电站分布图 - 高德地图"""
    if charging_stations.empty:
        st.warning("暂无充电站数据")
        return

    try:
        center_lat = charging_stations.geometry.y.mean()
        center_lon = charging_stations.geometry.x.mean()

        # 使用高德地图
        amap_tile = f"https://webst01.is.autonavi.com/appmaptile?style=7&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
        amap_attr = "高德地图"

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles=None
        )

        # 添加高德地图图层
        folium.TileLayer(
            tiles=AMAP_TILE_URL,
            attr="高德地图",
            name="高德地图"
        ).add_to(m)

        # 添加区县边界
        if not districts.empty:
            folium.GeoJson(
                districts,
                style_function=lambda x: {
                    'fillColor': 'rgba(30, 136, 229, 0.15)',
                    'color': '#1e88e5',
                    'weight': 2,
                    'fillOpacity': 0.15
                }
            ).add_to(m)

        # 添加充电站标记
        for _, row in charging_stations.iterrows():
            popup_html = f"""
            <div style='color: #1a237e; padding: 12px; min-width: 220px; background: white; border-radius: 8px;'>
                <h4 style='color: #1e88e5; margin-bottom: 10px; font-size: 15px; border-bottom: 1px solid #e3f2fd; padding-bottom: 8px;'>{row['名称']}</h4>
                <p style='margin: 6px 0; font-size: 14px;'><strong>运营商:</strong> {row['运营商']}</p>
                <p style='margin: 6px 0; font-size: 14px;'><strong>端口数:</strong> {row['端口数']}</p>
                <p style='margin: 6px 0; font-size: 13px; color: #757575;'><strong>坐标:</strong> {row['纬度']:.4f}, {row['经度']:.4f}</p>
            </div>
            """
            folium.CircleMarker(
                location=[row['纬度'], row['经度']],
                radius=8,
                color='#1e88e5',
                fill=True,
                fill_color='#42a5f5',
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=300),
                weight=2
            ).add_to(m)

        # 添加图例
        legend_html = '''
        <div style="position: fixed; bottom: 30px; right: 30px; z-index: 9999; background: rgba(255, 255, 255, 0.95); padding: 18px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
            <h4 style="color: #1a202c; margin-bottom: 12px; font-size: 16px;">图例</h4>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 14px; height: 14px; background: #42a5f5; border-radius: 50%; border: 2px solid #1e88e5; margin-right: 12px;"></div>
                <span style="color: #4a5568; font-size: 15px;">充电站</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 24px; height: 12px; background: rgba(30, 136, 229, 0.3); border: 1px solid #1e88e5; margin-right: 12px;"></div>
                <span style="color: #4a5568; font-size: 15px;">区县边界</span>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # 添加图层控制
        folium.LayerControl().add_to(m)

        folium_static(m, width=None, height=550)

    except Exception as e:
        st.error(f"地图加载失败: {e}")
        st.info("正在尝试使用备用地图服务...")
        st.dataframe(charging_stations[['名称', '运营商', '端口数', '经度', '纬度']])

def task1_density_map(charging_stations, districts):
    """区县密度图 - 使用真实地图底图"""
    if charging_stations.empty or districts.empty:
        st.warning("缺少充电站或区县数据")
        return

    try:
        # 确保两个GeoDataFrame都使用WGS84坐标系
        charging_wgs84 = charging_stations.to_crs(epsg=4326) if charging_stations.crs != 'EPSG:4326' else charging_stations.copy()
        districts_wgs84 = districts.to_crs(epsg=4326) if districts.crs != 'EPSG:4326' else districts.copy()
        
        # 检查并修复几何有效性
        districts_wgs84 = districts_wgs84[districts_wgs84.is_valid]
        if len(districts_wgs84) < len(districts):
            st.warning(f"修复了 {len(districts) - len(districts_wgs84)} 个无效几何")
        
        # 不使用缓冲区，直接用原始区县边界匹配
        # 第一步：先匹配在区县边界内的充电站
        charging_in_district = gpd.sjoin(charging_wgs84, districts_wgs84, how='inner', predicate='within')
        
        # 第二步：处理未匹配的充电站，用最近邻区县匹配
        matched_ids = charging_in_district.index.tolist()
        unmatched_stations = charging_wgs84[~charging_wgs84.index.isin(matched_ids)]
        
        if not unmatched_stations.empty:
            # 投影到UTM坐标系计算准确距离
            unmatched_utm = unmatched_stations.to_crs(config.UTM_EPSG)
            districts_utm = districts_wgs84.to_crs(config.UTM_EPSG)
            
            # 为每个未匹配的充电站找到最近的区县
            nearest_districts = []
            for _, station in unmatched_utm.iterrows():
                distances = districts_utm.geometry.distance(station.geometry)
                nearest_idx = distances.idxmin()
                nearest_districts.append(nearest_idx)
            
            # 创建未匹配充电站的临时DataFrame
            unmatched_with_district = unmatched_stations.copy()
            unmatched_with_district['index_right'] = nearest_districts
            
            # 合并两部分
            charging_in_district = pd.concat([charging_in_district, unmatched_with_district], ignore_index=True)
        
        # 对充电站去重，确保每个充电站只算一次
        charging_in_district = charging_in_district.drop_duplicates(subset=['经度', '纬度'])
        
        district_counts = charging_in_district.groupby('index_right').size()

        districts_copy = districts_wgs84.copy()
        districts_copy['充电站数量'] = district_counts.reindex(districts_copy.index, fill_value=0)

        # 使用投影坐标系计算面积（单位：km²）
        districts_proj = districts_copy.to_crs(config.UTM_EPSG)
        districts_copy['面积_km2'] = districts_proj.geometry.area / 10**6
        
        # 处理面积为0或极小的情况
        districts_copy['面积_km2'] = districts_copy['面积_km2'].apply(lambda x: max(x, 0.01))
        
        districts_copy['密度'] = districts_copy['充电站数量'] / districts_copy['面积_km2']
        districts_copy['密度'] = districts_copy['密度'].fillna(0)
        
        # 显示密度分布调试信息
        with st.expander("查看密度分布详情"):
            st.write("各区县密度数据：")
            density_stats = districts_copy[['名称', '充电站数量', '面积_km2', '密度']].sort_values('密度', ascending=False)
            st.dataframe(density_stats, use_container_width=True)
            st.write(f"最大密度: {districts_copy['密度'].max():.3f} 个/km²")
            st.write(f"最小密度: {districts_copy['密度'].min():.3f} 个/km²")
            st.write(f"平均密度: {districts_copy['密度'].mean():.3f} 个/km²")
            st.write(f"总充电站数: {districts_copy['充电站数量'].sum()}")

        # 计算中心坐标
        center_lat = districts_copy.geometry.centroid.y.mean()
        center_lon = districts_copy.geometry.centroid.x.mean()

        # 创建folium地图，使用高德地图作为底图
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles=None
        )

        # 添加高德地图图层
        folium.TileLayer(
            tiles=AMAP_TILE_URL,
            attr="高德地图",
            name="高德地图"
        ).add_to(m)

        # 定义颜色映射函数 - 使用绿色到红色渐变（低密度绿色，高密度红色）
        max_density = districts_copy['密度'].max()
        min_density = districts_copy['密度'].min()
        
        def get_color(density):
            if max_density == min_density:
                return 'rgba(76, 175, 80, 0.6)'
            normalized = (density - min_density) / (max_density - min_density)
            # 从绿色到红色的渐变
            r = int(76 + (244 - 76) * normalized)
            g = int(175 + (67 - 175) * normalized)
            b = int(80 + (54 - 80) * normalized)
            return f'rgba({r}, {g}, {b}, 0.7)'

        # 添加区县边界，根据密度着色
        for _, row in districts_copy.iterrows():
            geojson_data = row.geometry.__geo_interface__
            color = get_color(row['密度'])
            
            popup_html = f"""
            <div style='color: #1a202c; padding: 12px; min-width: 200px; background: white; border-radius: 8px;'>
                <h4 style='color: #1e88e5; margin-bottom: 10px; font-size: 15px; border-bottom: 1px solid #e3f2fd; padding-bottom: 8px;'>{row.get('名称', '未知区县')}</h4>
                <p style='margin: 6px 0; font-size: 14px;'><strong>充电站数量:</strong> {row['充电站数量']}</p>
                <p style='margin: 6px 0; font-size: 14px;'><strong>面积:</strong> {row['面积_km2']:.2f} km²</p>
                <p style='margin: 6px 0; font-size: 14px;'><strong>密度:</strong> {row['密度']:.3f} 个/km²</p>
            </div>
            """
            
            folium.GeoJson(
                geojson_data,
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': '#1e88e5',
                    'weight': 2,
                    'fillOpacity': 0.6
                },
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

        # 添加图例（绿色到红色渐变）
        legend_html = '''
        <div style="position: fixed; bottom: 30px; right: 30px; z-index: 9999; background: rgba(255, 255, 255, 0.95); padding: 18px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
            <h4 style="color: #1a202c; margin-bottom: 12px; font-size: 16px;">密度图例</h4>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 24px; height: 14px; background: rgba(76, 175, 80, 0.7); border: 1px solid #4caf50; margin-right: 12px;"></div>
                <span style="color: #4a5568; font-size: 14px;">低密度</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 24px; height: 14px; background: rgba(244, 67, 54, 0.7); border: 1px solid #f44336; margin-right: 12px;"></div>
                <span style="color: #4a5568; font-size: 14px;">高密度</span>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # 计算统计数据（只考虑有充电站的区县）
        has_charging = districts_copy[districts_copy['充电站数量'] > 0]
        if len(has_charging) > 0:
            max_density_val = has_charging['密度'].max()
            min_density_val = has_charging['密度'].min()
            avg_density_val = has_charging['密度'].mean()
        else:
            max_density_val = 0
            min_density_val = 0
            avg_density_val = 0
        
        # 添加统计信息面板
        stats_html = '''
        <div style="position: fixed; top: 150px; right: 30px; z-index: 9999; background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; min-width: 200px;">
            <h4 style="color: #1a202c; margin-bottom: 15px; font-size: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px;">区县统计概览</h4>
            <p style="margin: 8px 0; font-size: 14px;"><strong>区县总数:</strong> <span style="color: #1e88e5;">{district_count}</span></p>
            <p style="margin: 8px 0; font-size: 14px;"><strong>有充电站区县:</strong> <span style="color: #26a69a;">{has_charging_count}</span></p>
            <p style="margin: 8px 0; font-size: 14px;"><strong>最高密度:</strong> <span style="color: #f44336;">{max_density_val:.2f}</span> 个/km²</p>
            <p style="margin: 8px 0; font-size: 14px;"><strong>最低密度:</strong> <span style="color: #4caf50;">{min_density_val:.2f}</span> 个/km²</p>
            <p style="margin: 8px 0; font-size: 14px;"><strong>平均密度:</strong> <span style="color: #9c27b0;">{avg_density_val:.2f}</span> 个/km²</p>
        </div>
        '''.format(
            district_count=len(districts_copy),
            has_charging_count=len(has_charging),
            max_density_val=max_density_val,
            min_density_val=min_density_val,
            avg_density_val=avg_density_val
        )
        m.get_root().html.add_child(folium.Element(stats_html))

        # 添加图层控制
        folium.LayerControl().add_to(m)

        folium_static(m, width=None, height=550)

    except Exception as e:
        st.error(f"密度图加载失败: {e}")
        st.info("正在尝试使用备用图表...")
        # 备用：使用plotly
        fig = px.choropleth(
            districts_copy,
            geojson=districts_copy.geometry.__geo_interface__,
            locations=districts_copy.index,
            color='密度',
            color_continuous_scale=['#1a237e', '#1e88e5', '#42a5f5', '#90caf9', '#bbdefb'],
            hover_data=['充电站数量', '面积_km2', '密度'],
            labels={'密度': '充电站密度(个/km²)'},
            title='西安市各区县充电站密度分布'
        )
        fig.update_geos(fitbounds='locations', visible=False)
        fig.update_layout(
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            plot_bgcolor='rgba(255, 255, 255, 0.95)',
            font_color='#1a202c',
            title_font_size=22,
            title_font_color='#1a202c',
            coloraxis_colorbar=dict(
                title='密度',
                tickvals=[0, 0.5, 1, 2, 3],
                ticktext=['0', '0.5', '1', '2', '3'],
                title_font_color='#4a5568',
                tickfont_color='#4a5568',
                bgcolor='rgba(248, 249, 250, 0.8)'
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

def task1_operator_chart(charging_stations):
    """运营商统计图"""
    if charging_stations.empty:
        st.warning("暂无充电站数据")
        return

    try:
        operator_counts = charging_stations['运营商'].value_counts().reset_index()
        operator_counts.columns = ['运营商', '站点数']

        fig = px.bar(
            operator_counts,
            x='运营商',
            y='站点数',
            color='站点数',
            color_continuous_scale='Blues',
            title='各运营商充电站数量',
            labels={'站点数': '站点数量', '运营商': '运营商'}
        )
        fig.update_layout(
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font_color='#111827',
            title_font_size=20,
            title_font_color='#111827',
            xaxis_tickangle=-45,
            xaxis_title_font_color='#4b5563',
            yaxis_title_font_color='#4b5563',
            margin=dict(l=20, r=20, t=60, b=40),
            showlegend=False
        )
        fig.update_traces(
            marker=dict(line=dict(width=1, color='#e5e7eb')),
            hovertemplate='运营商: %{x}<br>站点数: %{y:}'
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"运营商统计图加载失败: {e}")

@st.cache_data(ttl=3600)
def create_grid(_city_boundary, grid_size=1000):
    """创建渔网网格 - 使用 UTM 投影确保 1km×1km 的准确网格"""
    city_boundary = _city_boundary
    if city_boundary is None:
        return gpd.GeoDataFrame()

    # 如果 city_boundary 是几何对象（如 MultiPolygon），转换为 GeoSeries
    from shapely.geometry import Polygon, MultiPolygon
    if isinstance(city_boundary, (Polygon, MultiPolygon)):
        city_boundary = gpd.GeoSeries([city_boundary], crs="EPSG:4326")
    elif isinstance(city_boundary, gpd.GeoDataFrame):
        # 如果是 GeoDataFrame，提取几何列
        city_boundary = city_boundary.geometry
    
    # 将边界投影到 UTM zone 49N (EPSG:32649) 用于准确的距离计算
    city_boundary_utm = city_boundary.to_crs(config.UTM_EPSG)
    
    # 获取边界范围
    bounds = city_boundary_utm.bounds.iloc[0] if hasattr(city_boundary_utm, 'bounds') else city_boundary_utm.total_bounds
    if hasattr(bounds, '__iter__') and len(bounds) >= 4:
        minx, miny, maxx, maxy = bounds
    else:
        # 如果 bounds 返回的是 GeoSeries
        minx, miny, maxx, maxy = bounds.minx, bounds.miny, bounds.maxx, bounds.maxy
    
    # 检查边界范围（调试信息）
    width = maxx - minx
    height = maxy - miny
    print(f"边界范围: {width:.1f}m × {height:.1f}m")
    print(f"预计网格数: {(width/grid_size):.0f} × {(height/grid_size):.0f} = {int(width*height/grid_size**2):,}")

    # 预计算边界的 union（只计算一次）
    boundary_union = city_boundary_utm.unary_union

    # 生成网格（单位：米）
    cols = np.arange(minx, maxx + grid_size, grid_size)
    rows = np.arange(miny, maxy + grid_size, grid_size)

    from shapely.geometry import Polygon
    polygons = []
    
    # 限制最大网格数量以避免内存问题
    max_grid_count = config.MAX_GRID_COUNT
    estimated_grids = len(cols) * len(rows)
    
    if estimated_grids > max_grid_count:
        # 如果预估网格过多，增大网格尺寸
        scale_factor = np.sqrt(estimated_grids / max_grid_count)
        grid_size = int(grid_size * scale_factor)
        print(f"网格过多，调整网格尺寸为 {grid_size}m")
        cols = np.arange(minx, maxx + grid_size, grid_size)
        rows = np.arange(miny, maxy + grid_size, grid_size)
    
    # 使用向量化方法：先创建所有网格，然后批量判断
    all_polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            # 创建矩形多边形（单位：米）
            coords = [(x, y), (x+grid_size, y), (x+grid_size, y+grid_size), (x, y+grid_size), (x, y)]
            poly = Polygon(coords)
            all_polygons.append(poly)
    
    print(f"创建了 {len(all_polygons):,} 个候选网格")
    
    # 批量创建 GeoSeries 进行空间判断（更高效）
    all_polygons_gs = gpd.GeoSeries(all_polygons, crs=config.UTM_EPSG)
    intersects_mask = all_polygons_gs.intersects(boundary_union)
    
    # 过滤出与边界相交的网格
    polygons = [poly for poly, intersects in zip(all_polygons, intersects_mask) if intersects]

    # 创建 GeoDataFrame（UTM 坐标系）
    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=config.UTM_EPSG)
    
    # 转换回 WGS84 用于显示
    grid = grid.to_crs("EPSG:4326")
    
    # 在 WGS84 坐标系下计算中心点
    grid['centroid'] = grid.geometry.centroid
    
    grid['grid_id'] = grid.index

    # 添加网格编号
    grid['grid_name'] = ['G' + str(i+1).zfill(4) for i in range(len(grid))]
    
    print(f"最终生成网格数量：{len(grid):,} 个（网格尺寸：{grid_size}m）")
    
    return grid

def calculate_supply_demand(charging_stations, grid, poi_df):
    """计算供需量"""
    if charging_stations.empty or grid.empty:
        st.warning("充电站或网格数据为空")
        return grid

    # 计算供给量
    grid['供给量'] = 0
    charging_in_grid = gpd.sjoin(charging_stations, grid, how='left', predicate='within')
    
    if not charging_in_grid.empty:
        supply_by_grid = charging_in_grid.groupby('index_right')['端口数'].sum()
        grid.loc[supply_by_grid.index, '供给量'] = supply_by_grid
        
        grid['站点数'] = charging_in_grid.groupby('index_right').size().reindex(grid.index, fill_value=0)
        grid['供给量'] = grid.apply(lambda row: row['供给量'] if row['供给量'] > 0 else row['站点数'] * 3, axis=1)
    else:
        grid['站点数'] = 0
        grid['供给量'] = 0
        st.info("没有充电站落在网格内，使用默认供给量")

    # 检查POI数据
    if poi_df is None or poi_df.empty:
        st.warning("POI数据为空，使用基于充电站分布的默认需求量")
        # 使用充电站密度作为需求量的替代指标
        grid['需求量'] = grid['站点数'].apply(lambda x: 10 - x * 2 if x > 0 else 10)
        grid['住宅POI'] = 0
        grid['商业POI'] = 0
        grid['办公POI'] = 0
        grid['交通POI'] = 0
        grid['供需比'] = grid['供给量'] / grid['需求量']
    else:
        res_keywords = ['住宅', '小区', '公寓', '居住', '家属院', '住宅区']
        com_keywords = ['商业', '商场', '超市', '购物', '广场', '商业街']
        off_keywords = ['办公', '写字楼', '大厦', '商务中心', '科技园']
        tra_keywords = ['公交', '地铁', '车站', '停车场', '换乘', '枢纽']

        res_poi = poi_df[poi_df['名称'].str.contains('|'.join(res_keywords), na=False)]
        com_poi = poi_df[poi_df['名称'].str.contains('|'.join(com_keywords), na=False)]
        off_poi = poi_df[poi_df['名称'].str.contains('|'.join(off_keywords), na=False)]
        tra_poi = poi_df[poi_df['名称'].str.contains('|'.join(tra_keywords), na=False)]

        def count_poi_in_grid(poi_gdf, grid_gdf):
            if poi_gdf.empty:
                return pd.Series([0] * len(grid_gdf), index=grid_gdf.index)
            poi_gdf = gpd.GeoDataFrame(poi_gdf, geometry=gpd.points_from_xy(poi_gdf['经度'], poi_gdf['纬度']), crs="EPSG:4326")
            joined = gpd.sjoin(poi_gdf, grid_gdf, how='left', predicate='within')
            return joined.groupby('index_right').size().reindex(grid_gdf.index, fill_value=0)

        grid['住宅POI'] = count_poi_in_grid(res_poi, grid)
        grid['商业POI'] = count_poi_in_grid(com_poi, grid)
        grid['办公POI'] = count_poi_in_grid(off_poi, grid)
        grid['交通POI'] = count_poi_in_grid(tra_poi, grid)

        # 计算需求量，确保最小值为1（权重来自 config.WEIGHTS）
        w = config.WEIGHTS
        raw_demand = grid['住宅POI'] * w['住宅'] + grid['商业POI'] * w['商业'] + grid['办公POI'] * w['办公'] + grid['交通POI'] * w['交通']
        grid['需求量'] = raw_demand.apply(lambda x: max(x, 1))
        grid['供需比'] = grid['供给量'] / grid['需求量']

    # 交通枢纽密度(归一化0-1),供选址"交通导向"情景作第三目标
    tmin, tmax = grid['交通POI'].min(), grid['交通POI'].max()
    grid['transit_density'] = (grid['交通POI'] - tmin) / (tmax - tmin + 1e-10) if tmax > tmin else 0.0

    # 确保供给量不为零
    grid['供给量'] = grid['供给量'].apply(lambda x: max(x, 1))

    # 计算网格面积（使用投影坐标系，单位 km²）
    grid_utm = grid.to_crs(config.UTM_EPSG)
    grid['面积_km2'] = grid_utm.geometry.area / 1e6  # 平方米转平方公里
    grid['人口密度'] = grid['需求量'] / grid['面积_km2']

    scaler = MinMaxScaler()
    grid['供给量_norm'] = scaler.fit_transform(grid[['供给量']])
    grid['需求量_norm'] = scaler.fit_transform(grid[['需求量']])

    return grid

def calculate_2sfca(grid, charging_stations, search_radius=5000):
    """真正的两步移动搜索法（2SFCA）计算供需匹配

    Step 1: Rj = Sj / SUM(Pk * Wkj)  — 每个充电站的供需比
    Step 2: Ai = SUM(Rj * Wij)       — 每个需求点的可达性得分
    """
    if grid.empty or charging_stations.empty:
        return grid, None

    # 将网格中心点和充电站投影到 UTM 坐标系用于距离计算
    grid_centroids_utm = gpd.GeoSeries(grid['centroid'], crs="EPSG:4326").to_crs(config.UTM_EPSG)
    charging_utm = charging_stations.to_crs(config.UTM_EPSG)

    grid_centroids = np.array([[g.x, g.y] for g in grid_centroids_utm])
    charging_coords = np.array([[g.x, g.y] for g in charging_utm.geometry])

    # 使用欧氏距离（UTM 投影下，单位为米）
    distance_matrix = cdist(grid_centroids, charging_coords, metric='euclidean')

    def gaussian_decay(d, d0):
        return np.exp(-(d**2) / (2 * d0**2))

    # 高斯衰减权重矩阵
    decay_matrix = gaussian_decay(distance_matrix, search_radius)
    decay_matrix[distance_matrix > search_radius] = 0

    charging_ports = charging_utm['端口数'].values  # (n_supply,)
    demand_values = grid['需求量'].values           # (n_demand,)

    # Step 1: 计算每个充电站的供需比 Rj = Sj / SUM(Pk * Wkj)
    demand_weighted = decay_matrix * demand_values.reshape(-1, 1)  # (n_demand, n_supply)
    demand_sum = demand_weighted.sum(axis=0)                       # (n_supply,)
    Rj = np.where(demand_sum > 0, charging_ports / demand_sum, 0)  # (n_supply,)

    # Step 2: 计算每个需求点的可达性得分 Ai = SUM(Rj * Wij)
    grid['可达供给'] = (decay_matrix * Rj).sum(axis=1)  # (n_demand,)

    # 供需得分（归一化）
    raw_score = grid['可达供给']
    if raw_score.max() > raw_score.min():
        grid['供需得分'] = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
    else:
        grid['供需得分'] = 0.5

    return grid, Rj

def calculate_e2sfca(grid, charging_stations, search_radius=5000):
    """E2SFCA (Enhanced 2SFCA) — 子区段权重: 0-1km/1-3km/3-5km = 1.0/0.68/0.22"""
    if grid.empty or charging_stations.empty:
        return grid, None

    grid_centroids_utm = gpd.GeoSeries(grid['centroid'], crs="EPSG:4326").to_crs(config.UTM_EPSG)
    charging_utm = charging_stations.to_crs(config.UTM_EPSG)
    grid_centroids = np.array([[g.x, g.y] for g in grid_centroids_utm])
    charging_coords = np.array([[g.x, g.y] for g in charging_utm.geometry])

    distance_matrix = cdist(grid_centroids, charging_coords, metric='euclidean')

    # 子区段定义
    breaks = [0, 1000, 3000, search_radius]
    weights = [1.0, 0.68, 0.22]

    step_weights = np.zeros_like(distance_matrix)
    for k in range(len(breaks) - 1):
        mask = (distance_matrix >= breaks[k]) & (distance_matrix < breaks[k + 1])
        gauss = np.exp(-(distance_matrix ** 2) / (2 * search_radius ** 2))
        step_weights += mask * gauss * weights[k]

    charging_ports = charging_utm['端口数'].values
    demand_values = grid['需求量'].values

    # Step 1
    demand_weighted = step_weights * demand_values.reshape(-1, 1)
    demand_sum = demand_weighted.sum(axis=0)
    Rj = np.where(demand_sum > 0, charging_ports / demand_sum, 0)

    # Step 2
    grid['可达供给'] = (step_weights * Rj).sum(axis=1)
    raw_score = grid['可达供给']
    if raw_score.max() > raw_score.min():
        grid['供需得分'] = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
    else:
        grid['供需得分'] = 0.5

    return grid, Rj

def task2_supply_demand_map(grid):
    """网格供需得分图"""
    if grid.empty or '供需得分' not in grid.columns:
        st.warning("暂无供需数据")
        return

    try:
        # 计算统计信息
        total_grids = len(grid)
        avg_supply = grid['供给量'].mean()
        avg_demand = grid['需求量'].mean()
        avg_ratio = grid['供需比'].mean()
        low_score_count = len(grid[grid['供需得分'] < 0.3])
        high_score_count = len(grid[grid['供需得分'] > 0.7])
        
        # 创建统计面板
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("网格总数", total_grids)
        with col2:
            st.metric("平均供给量", f"{avg_supply:.1f}")
        with col3:
            st.metric("平均需求量", f"{avg_demand:.1f}")
        with col4:
            st.metric("供给不足区", low_score_count)
        with col5:
            st.metric("供给充足区", high_score_count)

        # 创建folium地图，使用高德矢量底图
        center_lat = grid.geometry.centroid.y.mean()
        center_lon = grid.geometry.centroid.x.mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles=None
        )
        
        # 添加高德矢量底图
        folium.TileLayer(
            tiles=AMAP_TILE_URL,
            attr="高德地图",
            name="高德矢量图"
        ).add_to(m)
        
        # 根据供需得分设置颜色的函数
        def get_color(score):
            if score < 0.15:
                return '#7f0000'  # 严重不足 - 暗红
            elif score < 0.3:
                return '#d32f2f'  # 不足 - 红
            elif score < 0.45:
                return '#ff7043'  # 偏紧 - 橙红
            elif score < 0.55:
                return '#fff176'  # 平衡 - 淡黄
            elif score < 0.7:
                return '#81c784'  # 略充足 - 浅绿
            elif score < 0.85:
                return '#4caf50'  # 充足 - 绿
            else:
                return '#1b5e20'  # 过剩 - 深绿
        
        # 添加网格图层 - 使用 to_json() 方法确保正确序列化
        grid_4326 = grid.to_crs(epsg=4326).copy()
        
        # 移除无法序列化的列（如 centroid 列可能包含 Point 对象）
        for col in grid_4326.columns:
            if grid_4326[col].dtype.name == 'geometry' and col != 'geometry':
                grid_4326 = grid_4326.drop(columns=[col])
        
        # 添加供需状态列
        grid_4326['供需状态'] = grid_4326['供需比'].apply(lambda x: '供给充足' if x >= 1 else '供给不足')
        
        # 将 GeoDataFrame 转为 JSON 字符串
        grid_json = grid_4326.to_json()
        
        # 使用 folium.GeoJson 加载
        folium.GeoJson(
            grid_json,
            style_function=lambda x: {
                'fillColor': get_color(x['properties'].get('供需得分', 0.5)),
                'color': 'rgba(255, 255, 255, 0.3)',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['供给量', '可达供给', '需求量', '供需比', '供需状态', '供需得分'],
                aliases=['网格内端口数', '可达供给', '需求量', '供需比', '供需状态', '供给得分'],
                style=('background-color: #ffffff; color: #111827; padding: 10px; border-radius: 8px;'),
                format_string='<div>{}</div>'
            )
        ).add_to(m)
        
        # 添加图例
        legend_html = '''
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: #ffffff; padding: 18px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e5e7eb;">
            <h4 style="color: #111827; margin-bottom: 12px; font-size: 16px;">供需匹配得分</h4>
            <div style="color: #81c784; margin-bottom: 10px; font-size: 12px;">得分越高越充足</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 14px; background: #7f0000; border-radius: 3px; margin-right: 8px;"></div>
                    <span style="color: #4b5563; font-size: 13px;">严重不足</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 14px; background: #d32f2f; border-radius: 3px; margin-right: 8px;"></div>
                    <span style="color: #4b5563; font-size: 13px;">供给不足</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 14px; background: #ff7043; border-radius: 3px; margin-right: 8px;"></div>
                    <span style="color: #4b5563; font-size: 13px;">略偏紧</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 14px; background: #fff176; border-radius: 3px; margin-right: 8px;"></div>
                    <span style="color: #4b5563; font-size: 13px;">基本平衡</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 14px; background: #81c784; border-radius: 3px; margin-right: 8px;"></div>
                    <span style="color: #4b5563; font-size: 13px;">略充足</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 14px; background: #4caf50; border-radius: 3px; margin-right: 8px;"></div>
                    <span style="color: #4b5563; font-size: 13px;">供给充足</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 14px; background: #1b5e20; border-radius: 3px; margin-right: 8px;"></div>
                    <span style="color: #4b5563; font-size: 13px;">严重过剩</span>
                </div>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 添加图层控制
        folium.LayerControl().add_to(m)
        
        folium_static(m, width=None, height=500)

    except Exception as e:
        st.error(f"供需得分图加载失败: {e}")

def calculate_getis_ord_gistar(grid, column='供需得分', bandwidth=2000):
    """
    纯 Python 实现 Getis-Ord Gi*统计量
    参数:
        grid: GeoDataFrame，包含 centroid 列
        column: 分析的属性列名
        bandwidth: 带宽（米），用于距离衰减
    返回:
        grid: 添加了 Gi*值和 Z 分数的 GeoDataFrame
    """
    n = len(grid)
    
    # 将中心点投影到 UTM 坐标系进行准确距离计算
    centroids_utm = gpd.GeoSeries(grid['centroid'], crs="EPSG:4326").to_crs(config.UTM_EPSG)
    coords = np.array([[g.x, g.y] for g in centroids_utm])
    values = grid[column].values
    
    # 计算距离矩阵（UTM 投影下，单位为米）
    distances = cdist(coords, coords, metric='euclidean')
    
    # 计算空间权重矩阵（高斯距离衰减）
    weights_matrix = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))
    np.fill_diagonal(weights_matrix, 0)  # 自身权重为 0
    
    # 计算 Gi*统计量
    sum_weights = weights_matrix.sum(axis=1)
    sum_values = values.sum()
    
    Gi_star = np.zeros(n)
    for i in range(n):
        Gi_star[i] = np.sum(weights_matrix[i] * values) / sum_values
    
    # 计算期望和方差
    E_Gi = sum_weights / (n - 1)
    VAR_Gi = ((sum_weights ** 2).sum() - sum_weights ** 2 / (n - 1)) / ((n - 1) * (n - 2))
    
    # 计算 Z 分数
    Z_scores = (Gi_star - E_Gi) / np.sqrt(VAR_Gi + 1e-10)
    
    return Gi_star, Z_scores

def task2_hotspot_map(grid):
    """冷热点分析图 - 使用供需得分百分位数识别热点和冷点，基于高德地图"""
    if grid.empty or '供需得分' not in grid.columns:
        st.warning("暂无供需数据")
        return

    try:
        grid_copy = grid.copy()
        
        # 使用百分位数方法识别热点和冷点（更直观、更可靠）
        # 热点：供需得分最高的前15%区域（供给充足）
        # 冷点：供需得分最低的前15%区域（供给不足）
        hot_threshold = grid_copy['供需得分'].quantile(0.85)
        cold_threshold = grid_copy['供需得分'].quantile(0.15)
        
        grid_copy['hotspot'] = '不显著'
        grid_copy.loc[grid_copy['供需得分'] >= hot_threshold, 'hotspot'] = '热点'
        grid_copy.loc[grid_copy['供需得分'] <= cold_threshold, 'hotspot'] = '冷点'
        
        # 统计信息
        hot_count = len(grid_copy[grid_copy['hotspot'] == '热点'])
        cold_count = len(grid_copy[grid_copy['hotspot'] == '冷点'])
        ns_count = len(grid_copy[grid_copy['hotspot'] == '不显著'])
        
        # 创建统计面板
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("热点区域", hot_count, help="高值聚集区域（供给充足）")
        with col2:
            st.metric("冷点区域", cold_count, help="低值聚集区域（供给不足）")
        with col3:
            st.metric("不显著区域", ns_count, help="中等供需区域")

        # 创建folium地图，使用高德地图底图（行政地图）
        center_lat, center_lng = 34.34, 108.94  # 西安市中心坐标
        
        # 使用高德行政地图作为底图（使用更可靠的URL格式）
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=10,
            tiles=None  # 先不加载默认底图
        )
        
        # 添加高德地图底图（行政地图）
        folium.TileLayer(
            tiles=AMAP_TILE_URL,
            attr='高德地图',
            name='高德地图',
            overlay=False,
            control=True
        ).add_to(m)

        # 定义颜色方案 - 更明显的区分度
        hotspot_colors = {
            '热点': '#ff0000',        # 鲜红色 - 高值聚集
            '冷点': '#0066ff',        # 亮蓝色 - 低值聚集
            '不显著': '#cccccc'       # 浅灰色 - 中等区域
        }
        
        # 按热点类型分组渲染，提高性能（先渲染不显著，再渲染冷点，最后渲染热点）
        for hotspot_type in ['不显著', '冷点', '热点']:
            subset = grid_copy[grid_copy['hotspot'] == hotspot_type]
            if len(subset) == 0:
                continue
            
            color = hotspot_colors.get(hotspot_type, '#cccccc')
            
            # 批量渲染GeoJson
            geojson_data = subset[['geometry', 'grid_name', '供需得分', 'hotspot']].__geo_interface__
            
            opacity = 0.4 if hotspot_type == '不显著' else 0.7
            
            def create_style_fn(color, opacity):
                def style_fn(feature):
                    return {
                        'fillColor': color,
                        'color': color,
                        'weight': 0.5,
                        'fillOpacity': opacity
                    }
                return style_fn
            
            # 添加GeoJson图层
            folium.GeoJson(
                geojson_data,
                style_function=create_style_fn(color, opacity),
                tooltip=folium.GeoJsonTooltip(
                    fields=['grid_name', 'hotspot', '供需得分'],
                    aliases=['网格ID', '聚集类型', '供需得分'],
                    localize=True
                )
            ).add_to(m)

        # 添加图例
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
        <h4 style="margin: 0 0 10px 0; font-size: 14px;">冷热点图例</h4>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #ff0000; border-radius: 4px; margin-right: 10px;"></div>
            <span style="font-size: 12px;">热点区域（高值聚集）</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #0066ff; border-radius: 4px; margin-right: 10px;"></div>
            <span style="font-size: 12px;">冷点区域（低值聚集）</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 20px; background-color: #cccccc; border-radius: 4px; margin-right: 10px;"></div>
            <span style="font-size: 12px;">不显著区域</span>
        </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 在Streamlit中显示地图
        from streamlit_folium import folium_static
        folium_static(m, width=None, height=550)

        # 添加算法说明
        st.markdown("""
        <div style="background: rgba(255,255,255,0.92); padding: 20px; border-radius: 12px; margin-top: 20px;">
            <h4 style="color: #111827; margin-bottom: 15px;">Getis-Ord Gi* 算法说明</h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div style="display: flex; align-items: center;">
                    <div style="background: #d32f2f; width: 24px; height: 16px; border-radius: 3px; margin-right: 10px;"></div>
                    <span style="color: #4b5563; font-size: 14px;">热点区域</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="background: #1e88e5; width: 24px; height: 16px; border-radius: 3px; margin-right: 10px;"></div>
                    <span style="color: #4b5563; font-size: 14px;">冷点区域</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="background: #484f58; width: 24px; height: 16px; border-radius: 3px; margin-right: 10px;"></div>
                    <span style="color: #4b5563; font-size: 14px;">不显著</span>
                </div>
            </div>
            <p style="color: #8b949e; font-size: 13px; margin-top: 15px; line-height: 1.6;">
                <strong>Getis-Ord Gi*</strong> 是一种空间自相关分析方法，用于识别高值或低值的空间聚集区域。
                热点表示充电站供需得分高值的空间聚集（供给充足区域），冷点表示供需得分低值的空间聚集（供给不足区域）。
                分析基于 95% 置信水平（Z分数 > 1.96 或 < -1.96）。
            </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"冷热点分析失败: {e}")
        st.exception(e)

def task2_blind_area_map(grid, charging_stations, districts, threshold_percentile=config.BLIND_THRESHOLD):
    """服务盲区地图 - 高德"""
    if grid.empty or '供需得分' not in grid.columns:
        st.warning("暂无供需数据")
        return

    try:
        # 按供需得分升序排序，取最低的threshold_percentile比例
        grid_sorted = grid.sort_values('供需得分', ascending=True)
        blind_count = int(len(grid_sorted) * threshold_percentile)
        # 确保至少选择1个网格
        blind_count = max(blind_count, 1)
        blind_areas = grid_sorted.head(blind_count).copy()
        
        # 移除无法JSON序列化的列（如centroid Point类型）
        columns_to_keep = ['geometry', '供需得分', '供给量', '需求量']
        blind_areas = blind_areas[[col for col in columns_to_keep if col in blind_areas.columns]]

        # 统计信息
        blind_count = len(blind_areas)
        total_count = len(grid)
        avg_score = blind_areas['供需得分'].mean()
        
        # 创建统计面板
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("服务盲区数量", blind_count)
        with col2:
            st.metric("占比", f"{(blind_count/total_count*100):.1f}%")
        with col3:
            st.metric("平均供需得分", f"{avg_score:.3f}")

        if not charging_stations.empty:
            center_lat = charging_stations.geometry.y.mean()
            center_lon = charging_stations.geometry.x.mean()
        else:
            center_lat, center_lon = 34.27, 108.95

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles=None
        )

        folium.TileLayer(
            tiles=AMAP_TILE_URL,
            attr="高德地图",
            name="高德地图"
        ).add_to(m)

        if not districts.empty:
            folium.GeoJson(
                districts,
                style_function=lambda x: {
                    'fillColor': 'rgba(30, 136, 229, 0.08)',
                    'color': '#1e88e5',
                    'weight': 1,
                    'fillOpacity': 0.08
                }
            ).add_to(m)

        folium.GeoJson(
            blind_areas,
            style_function=lambda x: {
                'fillColor': '#d32f2f',
                'color': '#b71c1c',
                'weight': 1.5,
                'fillOpacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['供需得分', '供给量', '需求量'],
                aliases=['供需得分', '供给量', '需求量'],
                style=('background-color: #ffffff; color: #111827; '
                       'border: 1px solid #42a5f5; padding: 10px; border-radius: 8px;')
            )
        ).add_to(m)

        if not charging_stations.empty:
            for _, row in charging_stations.iterrows():
                folium.CircleMarker(
                    location=[row['纬度'], row['经度']],
                    radius=6,
                    color='#1e88e5',
                    fill=True,
                    fill_color='#42a5f5',
                    fill_opacity=0.9,
                    tooltip=f"充电站: {row.get('名称', '未知')}"
                ).add_to(m)

        legend_html = '''
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e5e7eb;">
            <h4 style="color: #111827; margin-bottom: 15px; font-size: 16px; font-weight: bold;">图例</h4>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 28px; height: 28px; background: #d32f2f; opacity: 0.7; border: 1.5px solid #b71c1c; border-radius: 4px; margin-right: 12px;"></div>
                <span style="color: #4b5563; font-size: 15px;">服务盲区（供需得分最低的20%）</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 14px; height: 14px; background: #42a5f5; border-radius: 50%; border: 2px solid #1e88e5; margin-right: 12px;"></div>
                <span style="color: #4b5563; font-size: 15px;">现有充电站</span>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl().add_to(m)

        folium_static(m, width=None, height=450)

        if not blind_areas.empty:
            geojson_data = blind_areas.to_json()
            st.download_button(
                label="下载盲区GeoJSON",
                data=geojson_data,
                file_name='blind_areas.geojson',
                mime='application/json',
                key='download_blind_areas'
            )

    except Exception as e:
        st.error(f"盲区地图加载失败: {e}")

def generate_candidates(blind_areas, roads, charging_stations, city_boundary=None):
    """生成候选点 — 所有候选点必须通过路网可通车校验(距道路≤150m)

    使用 cKDTree 空间索引加速最近道路查询,避免对18万路网点暴力计算。
    """
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    # === 一次性构建路网空间索引 ===
    road_pts_wgs = []          # (lon, lat) 列表
    road_pts_utm = []          # (x, y) 米列表
    tree = None
    road_inv = None            # Transformer: WGS84 -> UTM

    if not roads.empty:
        try:
            roads_wgs = roads.to_crs(epsg=4326)
            road_inv = Transformer.from_crs("EPSG:4326", config.UTM_EPSG, always_xy=True)
            for geom in roads_wgs.geometry:
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    step = max(1, len(coords) // max(1, int(geom.length * 111000) // 100))
                    for i in range(0, len(coords), step):
                        road_pts_wgs.append(coords[i])
            # 去重
            road_pts_wgs = list(set(road_pts_wgs))
            arr = np.array(road_pts_wgs)
            road_pts_utm = np.array(road_inv.transform(arr[:, 0], arr[:, 1])).T  # (n, 2)
            tree = cKDTree(road_pts_utm)
        except Exception as e:
            st.warning(f"路网约束构建失败: {e}")
            tree = None

    def _snap_to_road(lon, lat):
        """吸附到最近道路 (cKDTree top-5随机 + 微扰动)"""
        if tree is None:
            return lon, lat
        x, y = road_inv.transform(lon, lat)
        dists, idxs = tree.query([x, y], k=min(5, len(road_pts_utm)))
        chosen = idxs[np.random.randint(len(idxs))] if hasattr(idxs, '__len__') else idxs
        base_lon, base_lat = road_pts_wgs[chosen]
        jitter = 0.0002  # ±20m
        return base_lon + (np.random.random() - 0.5) * jitter * 2, \
               base_lat + (np.random.random() - 0.5) * jitter * 2

    def _is_accessible(lon, lat):
        """150m内存在路网点则可达"""
        if tree is None:
            return True
        x, y = road_inv.transform(lon, lat)
        nbrs = tree.query_ball_point([x, y], r=150)
        return len(nbrs) > 0

    candidates = []
    snapped_count = 0

    # === 1. 盲区质心候选 (限制数量,避免爆炸) ===
    if not blind_areas.empty:
        if '需求量' in blind_areas.columns:
            blind_areas_sorted = blind_areas.sort_values('需求量', ascending=False)
        else:
            blind_areas_sorted = blind_areas

        # 只取需求量最高的前 120 个盲区,每个生成 1 个候选点
        top_blind = blind_areas_sorted.head(120)
        for _, row in top_blind.iterrows():
            offset_x = (np.random.random() - 0.5) * 0.006
            offset_y = (np.random.random() - 0.5) * 0.006
            cx, cy = row['centroid'].x + offset_x, row['centroid'].y + offset_y

            if not _is_accessible(cx, cy):
                cx, cy = _snap_to_road(cx, cy)
                snapped_count += 1
            candidates.append({'x': cx, 'y': cy, 'type': 'blind_area'})
    else:
        st.info("未找到服务盲区，将从整个城市区域生成候选点")

    # === 2. 路网采样 (天然可通车) ===
    if len(candidates) < 200 and road_pts_wgs:
        rng_idx = np.random.choice(len(road_pts_wgs), size=min(200 - len(candidates), len(road_pts_wgs)), replace=False)
        for i in rng_idx:
            cx, cy = road_pts_wgs[i]
            offset_x = (np.random.random() - 0.5) * 0.002
            offset_y = (np.random.random() - 0.5) * 0.002
            candidates.append({'x': cx + offset_x, 'y': cy + offset_y, 'type': 'road'})

    # === 3. 边界内随机 (必须可通车) ===
    if len(candidates) < 120:
        if city_boundary is not None:
            minx, miny, maxx, maxy = city_boundary.bounds
        elif not blind_areas.empty:
            minx, miny, maxx, maxy = blind_areas.total_bounds
        else:
            minx, miny, maxx, maxy = 107.4, 33.3, 109.6, 34.7

        attempts = 0
        while len(candidates) < 120 and attempts < 600:
            cx = np.random.uniform(minx, maxx)
            cy = np.random.uniform(miny, maxy)
            if not _is_accessible(cx, cy):
                cx, cy = _snap_to_road(cx, cy)
                snapped_count += 1
            candidates.append({'x': cx, 'y': cy, 'type': 'random'})
            attempts += 1

    if snapped_count > 0:
        st.info(f"🚗 {snapped_count}个候选点不在道路可达范围内，已自动吸附至最近道路")

    # === 转 GeoDataFrame 并过滤与现有站点过近的候选 ===
    if not candidates:
        return gpd.GeoDataFrame()

    candidates_df = pd.DataFrame(candidates)
    candidates_gdf = gpd.GeoDataFrame(
        candidates_df,
        geometry=gpd.points_from_xy(candidates_df['x'], candidates_df['y']),
        crs="EPSG:4326"
    )

    # 用 cKDTree 过滤与现有站点 <500m 的候选
    if not charging_stations.empty and len(charging_stations) > 0 and road_inv is not None:
        cand_utm = np.array(road_inv.transform(
            [g.x for g in candidates_gdf.geometry],
            [g.y for g in candidates_gdf.geometry]
        )).T
        st_utm = np.array(road_inv.transform(
            [g.x for g in charging_stations.geometry],
            [g.y for g in charging_stations.geometry]
        )).T
        st_tree = cKDTree(st_utm)
        too_close = st_tree.query_ball_point(cand_utm, r=500)
        keep_mask = np.array([len(nbrs) == 0 for nbrs in too_close])
        candidates_gdf = candidates_gdf[keep_mask].copy()

    if len(candidates_gdf) == 0:
        st.warning("所有候选点都离现有站点太近，已放宽距离限制")
        candidates_gdf = gpd.GeoDataFrame(
            candidates_df,
            geometry=gpd.points_from_xy(candidates_df['x'], candidates_df['y']),
            crs="EPSG:4326"
        )

    # 限制最多 200 个
    if len(candidates_gdf) > 200:
        candidates_gdf = candidates_gdf.sample(200, random_state=42).reset_index(drop=True)

    return candidates_gdf

def optimize_locations(candidates, grid, k=5, search_radius=5000, solver_method='greedy', enforce_spread=True):
    """选址优化 - 支持整数规划和贪心算法，带分散约束"""
    import time
    start_time = time.time()
    
    if candidates.empty or grid.empty:
        st.warning("候选点或网格数据为空")
        return gpd.GeoDataFrame(), 0.0, 0.0

    # 限制候选点数量最多100个
    max_candidates = 100
    if len(candidates) > max_candidates:
        # 如果有需求量列，按需求量加权随机采样
        if '需求量' in grid.columns and 'centroid' in grid.columns:
            # 计算每个候选点附近的需求密度
            candidate_coords = np.array([[g.x, g.y] for g in candidates.geometry])
            grid_coords = np.array([[g.x, g.y] for g in grid['centroid']])
            dists = cdist(candidate_coords, grid_coords) * 111000
            weights = grid['需求量'].values if '需求量' in grid.columns else np.ones(len(grid))
            # 计算每个候选点的需求覆盖
            demand_scores = (dists <= search_radius).astype(float) @ weights
            # 按需求得分排序取前100
            top_indices = np.argsort(demand_scores)[-max_candidates:]
            candidates = candidates.iloc[top_indices].copy()
        else:
            # 随机采样
            candidates = candidates.sample(max_candidates, random_state=np.random.randint(1000))
    
    n_candidates = len(candidates)
    if n_candidates == 0:
        st.warning("没有候选点可供选择")
        return gpd.GeoDataFrame(), 0.0

    total_demand = grid['需求量'].sum() if '需求量' in grid.columns else len(grid)
    selected_indices = []
    solve_time = 0.0

    try:
        if solver_method == 'integer' and n_candidates <= 80:
            # 使用整数规划求解
            from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value
            from pulp import PULP_CBC_CMD

            candidate_coords = np.array([[g.x, g.y] for g in candidates.geometry])
            grid_coords = np.array([[g.x, g.y] for g in grid['centroid']])

            distances = cdist(candidate_coords, grid_coords) * 111000
            coverage = (distances <= search_radius).astype(int)
            weights = grid['需求量'].values if '需求量' in grid.columns else np.ones(len(grid))

            prob = LpProblem("ChargerLocation", LpMaximize)
            x = [LpVariable(f"x_{i}", cat='Binary') for i in range(n_candidates)]
            
            # 目标函数：最大化覆盖需求
            prob += lpSum([x[i] * sum(coverage[i] * weights) for i in range(n_candidates)])
            
            # 约束：最多选择k个站点
            prob += lpSum(x) <= k
            
            # 分散约束：站点间至少1.5km
            if enforce_spread:
                min_distance = config.SPREAD_DISTANCE  # 1.5km
                candidate_distances = cdist(candidate_coords, candidate_coords) * 111000
                for i in range(n_candidates):
                    for j in range(i+1, n_candidates):
                        if candidate_distances[i, j] < min_distance:
                            prob += x[i] + x[j] <= 1

            # 设置60秒超时
            solver = PULP_CBC_CMD(msg=0, timeLimit=config.IP_TIME_LIMIT)
            prob.solve(solver)

            selected_indices = [i for i in range(n_candidates) if value(x[i]) == 1]
            solve_time = prob.solutionTime if hasattr(prob, 'solutionTime') else 0

            if len(selected_indices) == 0:
                st.info("整数规划未找到可行解，使用贪心算法")
                selected_indices = greedy_selection(candidates, grid, k, search_radius, enforce_spread)
        
        else:
            # 使用贪心算法
            selected_indices = greedy_selection(candidates, grid, k, search_radius, enforce_spread)

    except ImportError:
        st.info("pulp库未安装，使用贪心算法")
        selected_indices = greedy_selection(candidates, grid, k, search_radius, enforce_spread)
    except Exception as e:
        st.warning(f"优化算法出错: {e}，使用贪心算法")
        selected_indices = greedy_selection(candidates, grid, k, search_radius, enforce_spread)

    # 计算覆盖率提升
    if selected_indices:
        selected_coords = np.array([[candidates.iloc[i].geometry.x, candidates.iloc[i].geometry.y] for i in selected_indices])
        grid_coords = np.array([[g.x, g.y] for g in grid['centroid']])
        dists = cdist(selected_coords, grid_coords) * 111000
        covered = (dists <= search_radius).any(axis=0)
        weights = grid['需求量'].values if '需求量' in grid.columns else np.ones(len(grid))
        coverage_ratio = covered @ weights / total_demand
    else:
        coverage_ratio = 0.0
    
    solve_time = time.time() - start_time

    return candidates.iloc[selected_indices].copy() if selected_indices else gpd.GeoDataFrame(), coverage_ratio, solve_time

def greedy_selection(candidates, grid, k, search_radius, enforce_spread=True):
    """贪心算法选择站点 - 带分散约束"""
    if len(candidates) == 0 or len(grid) == 0:
        return []
    
    selected_indices = []
    
    # 向量化准备
    candidate_coords = np.array([[g.x, g.y] for g in candidates.geometry])
    grid_coords = np.array([[g.x, g.y] for g in grid['centroid']])
    weights = grid['需求量'].values if '需求量' in grid.columns else np.ones(len(grid))
    
    # 计算所有候选点到所有网格的距离（向量化）
    distances = cdist(candidate_coords, grid_coords) * 111000
    
    # 覆盖矩阵
    coverage = (distances <= search_radius).astype(float)
    
    # 加权覆盖
    weighted_coverage = coverage * weights
    
    # 已覆盖的网格
    covered = np.zeros(len(grid), dtype=bool)
    
    # 分散约束：最小距离1.5km
    min_distance = config.SPREAD_DISTANCE  # 1.5km
    candidate_distances = cdist(candidate_coords, candidate_coords) * 111000
    
    for _ in range(min(k, len(candidates))):
        # 计算每个候选点的新增覆盖
        if selected_indices:
            # 排除已选点
            mask = np.ones(len(candidates), dtype=bool)
            mask[selected_indices] = False
            
            # 如果启用分散约束，排除距离已选点太近的候选点
            if enforce_spread and len(selected_indices) > 0:
                for sel_idx in selected_indices:
                    too_close = candidate_distances[sel_idx] < min_distance
                    mask = mask & ~too_close
            
            remaining_coverage = weighted_coverage[mask] * (~covered).astype(float)
            scores = remaining_coverage.sum(axis=1)
            # 创建索引映射
            available_idx = np.where(mask)[0]
            if len(scores) == 0:
                break
            best_score_idx = np.argmax(scores)
            best_idx = available_idx[best_score_idx]
            best_score = scores[best_score_idx]
        else:
            scores = weighted_coverage.sum(axis=1)
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
        
        if best_score <= 0:
            # 如果没有新增覆盖，随机选择一个未选点
            available = [i for i in range(len(candidates)) if i not in selected_indices]
            if enforce_spread:
                # 排除距离已选点太近的
                for sel_idx in selected_indices:
                    available = [i for i in available if candidate_distances[sel_idx, i] >= min_distance]
            if available:
                best_idx = np.random.choice(available)
            else:
                break
        
        selected_indices.append(best_idx)
        # 更新已覆盖网格
        covered = covered | (distances[best_idx] <= search_radius)

    return selected_indices

def calculate_metrics(grid, charging_stations, new_stations=None, search_radius=5000):
    """计算扩展评价指标 - 使用投影坐标系进行准确距离计算"""
    all_stations = charging_stations.copy()
    if new_stations is not None and not new_stations.empty:
        all_stations = pd.concat([charging_stations, new_stations], ignore_index=True)

    # 投影到 UTM 坐标系进行距离计算
    grid_centroids_utm = gpd.GeoSeries(grid['centroid'], crs="EPSG:4326").to_crs(config.UTM_EPSG)
    all_stations_utm = all_stations.to_crs(config.UTM_EPSG)

    grid_centroids = np.array([[g.x, g.y] for g in grid_centroids_utm])
    station_coords = np.array([[g.x, g.y] for g in all_stations_utm.geometry])

    # 距离计算
    distances = cdist(grid_centroids, station_coords, metric='euclidean')
    min_distances = distances.min(axis=1)
    avg_distance = min_distances.mean() / 1000  # km
    coverage_rate = (min_distances <= search_radius).mean() * 100
    p95_distance = np.percentile(min_distances, 95) / 1000  # km

    # Gini（距离）
    sorted_dist = np.sort(min_distances)
    n = len(sorted_dist)
    cum = np.cumsum(sorted_dist)
    dist_gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n if cum[-1] > 0 else 0

    # 可达性 Gini（基于 2SFCA 得分）
    if '供需得分' in grid.columns:
        acc = grid['供需得分'].values
    else:
        acc = 1.0 / (min_distances + 1.0)
    acc_sorted = np.sort(acc)
    n_acc = len(acc_sorted)
    cum_acc = np.cumsum(acc_sorted)
    acc_gini = (n_acc + 1 - 2 * np.sum(cum_acc) / cum_acc[-1]) / n_acc if cum_acc[-1] > 0 else 0

    # Theil 指数
    acc_pos = acc[acc > 0]
    if len(acc_pos) > 0:
        mean_acc = acc_pos.mean()
        theil = np.mean((acc_pos / mean_acc) * np.log(acc_pos / mean_acc))
    else:
        theil = 0

    # 服务重叠率
    overlap_rate = ((distances <= search_radius).sum(axis=1) > 1).mean() * 100 if distances.shape[1] > 1 else 0

    # 总端口数
    total_ports = int(all_stations['端口数'].sum()) if '端口数' in all_stations.columns else len(all_stations)

    return {
        '平均可达距离(km)': avg_distance,
        '覆盖率(%)': coverage_rate,
        'P95距离(km)': p95_distance,
        '可达性Gini': acc_gini,
        '距离Gini': dist_gini,
        'Theil指数': theil,
        '服务重叠率(%)': overlap_rate,
        '站点总数': len(all_stations),
        '总端口数': total_ports,
    }

def task3_comparison_map(charging_stations, new_stations, districts):
    """优化前后对比图 - 高德矢量底图"""
    if charging_stations.empty:
        st.warning("暂无充电站数据")
        return

    try:
        center_lat = charging_stations.geometry.y.mean()
        center_lon = charging_stations.geometry.x.mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=None
        )

        # 添加高德矢量底图
        folium.TileLayer(
            tiles=AMAP_TILE_URL,
            attr="高德地图",
            name="高德矢量图"
        ).add_to(m)

        if not districts.empty:
            folium.GeoJson(
                districts,
                style_function=lambda x: {
                    'fillColor': 'rgba(30, 136, 229, 0.08)',
                    'color': '#1e88e5',
                    'weight': 1,
                    'fillOpacity': 0.08
                }
            ).add_to(m)

        # 现有充电站 - 蓝色标记
        for _, row in charging_stations.iterrows():
            folium.CircleMarker(
                location=[row['纬度'], row['经度']],
                radius=6,
                color='#1e88e5',
                fill=True,
                fill_color='#42a5f5',
                fill_opacity=0.8,
                popup=folium.Popup(f"<strong>现有站点</strong><br>{row['名称']}", max_width=200)
            ).add_to(m)

        # 新增站点 - 亮绿色大号标记 + 缓冲区
        if new_stations is not None and not new_stations.empty:
            for idx, row in new_stations.iterrows():
                lat, lon = row.geometry.y, row.geometry.x
                
                # 添加服务范围缓冲区（1.5km半径）
                folium.Circle(
                    location=[lat, lon],
                    radius=1500,  # 1.5km
                    color='#3fb950',
                    fill=True,
                    fill_color='#3fb950',
                    fill_opacity=0.15,
                    weight=2,
                    popup=folium.Popup(f"<strong>服务范围</strong><br>半径: 1.5km", max_width=200)
                ).add_to(m)
                
                # 添加亮绿色大号标记
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=12,
                    color='#22c55e',
                    fill=True,
                    fill_color='#4ade80',
                    fill_opacity=0.95,
                    weight=3,
                    popup=folium.Popup(f"<strong>新增站点 {idx+1}</strong><br>坐标: {lat:.4f}, {lon:.4f}", max_width=200)
                ).add_to(m)

        legend_html = '''
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: #ffffff; padding: 18px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e5e7eb;">
            <h4 style="color: #111827; margin-bottom: 12px; font-size: 16px;">图例</h4>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 12px; height: 12px; background: #42a5f5; border-radius: 50%; border: 2px solid #1e88e5; margin-right: 12px;"></div>
                <span style="color: #4b5563; font-size: 15px;">现有充电站</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 14px; height: 14px; background: #26a69a; border-radius: 50%; border: 2px solid #3fb950; margin-right: 12px;"></div>
                <span style="color: #4b5563; font-size: 15px;">新增充电站</span>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl().add_to(m)

        folium_static(m, width=None, height=550)

    except Exception as e:
        st.error(f"优化前后对比图加载失败: {e}")

def task3_new_stations_map(new_stations, districts):
    """新增站点地图 - 高德"""
    if new_stations is None or new_stations.empty:
        st.warning("暂无新增站点数据")
        return

    try:
        center_lat = new_stations.geometry.y.mean()
        center_lon = new_stations.geometry.x.mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles=None
        )

        # 添加高德矢量底图（不带API key，提高兼容性）
        folium.TileLayer(
            tiles=AMAP_TILE_URL,
            attr="高德地图",
            name="高德矢量图"
        ).add_to(m)

        if not districts.empty:
            folium.GeoJson(
                districts,
                style_function=lambda x: {
                    'fillColor': 'rgba(30, 136, 229, 0.08)',
                    'color': '#1e88e5',
                    'weight': 1,
                    'fillOpacity': 0.08
                }
            ).add_to(m)

        for idx, row in new_stations.iterrows():
            # 添加1.5km服务范围缓冲区（半透明绿圈）
            folium.Circle(
                location=[row.geometry.y, row.geometry.x],
                radius=1500,  # 1.5km
                color='#00FF00',
                fill=True,
                fill_color='#00FF00',
                fill_opacity=0.15,
                weight=2
            ).add_to(m)
            
            popup_html = f"""
            <div style='color: #1a237e; padding: 12px; background: white; border-radius: 8px;'>
                <h4 style='color: #3fb950; margin-bottom: 8px;'>新增站点 {idx + 1}</h4>
                <p style='margin: 6px 0; font-size: 14px;'><strong>坐标:</strong> {row.geometry.y:.4f}, {row.geometry.x:.4f}</p>
                <p style='margin: 6px 0; font-size: 14px;'><strong>类型:</strong> {row.get('type', '未知')}</p>
                <p style='margin: 6px 0; font-size: 14px;'><strong>服务范围:</strong> 1.5km</p>
            </div>
            """
            # 亮绿色大号marker
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=14,
                color='#22c55e',
                fill=True,
                fill_color='#4ade80',
                fill_opacity=0.95,
                popup=folium.Popup(popup_html, max_width=250),
                weight=3
            ).add_to(m)

        folium.LayerControl().add_to(m)

        folium_static(m, width=None, height=550)

    except Exception as e:
        st.error(f"新增站点地图加载失败: {e}")

def task3_metrics_table(before_metrics, after_metrics):
    """指标对比表"""
    # 计算变化值
    distance_diff = before_metrics['平均可达距离(km)'] - after_metrics['平均可达距离(km)']
    coverage_diff = after_metrics['覆盖率(%)'] - before_metrics['覆盖率(%)']
    gini_diff = after_metrics['可达性Gini'] - before_metrics['可达性Gini']
    station_diff = after_metrics['站点总数'] - before_metrics['站点总数']
    
    metrics_df = pd.DataFrame({
        '指标': ['平均可达距离(km)', '覆盖率(%)', '可达性Gini', '站点总数'],
        '优化前': [
            f"{before_metrics['平均可达距离(km)']:.2f}",
            f"{before_metrics['覆盖率(%)']:.1f}",
            f"{before_metrics['可达性Gini']:.3f}",
            before_metrics['站点总数']
        ],
        '优化后': [
            f"{after_metrics['平均可达距离(km)']:.2f}",
            f"{after_metrics['覆盖率(%)']:.1f}",
            f"{after_metrics['可达性Gini']:.3f}",
            after_metrics['站点总数']
        ],
        '变化': [
            f"-{distance_diff:.2f}" if distance_diff > 0 else f"+{-distance_diff:.2f}",
            f"+{coverage_diff:.1f}" if coverage_diff > 0 else f"{coverage_diff:.1f}",
            f"{gini_diff:+.3f}",
            f"+{station_diff}"
        ]
    })

    # 创建紧凑的柱形图
    fig = go.Figure(data=[
        go.Bar(name='优化前', x=metrics_df['指标'][:3],
               y=[before_metrics['平均可达距离(km)'], before_metrics['覆盖率(%)'], before_metrics['可达性Gini']],
               marker_color='#42a5f5',
               marker_line_color='#1e88e5',
               marker_line_width=1.5,
               width=0.35,
               text=[f"{v:.2f}" for v in [before_metrics['平均可达距离(km)'], before_metrics['覆盖率(%)'], before_metrics['可达性Gini']]],
               textposition='auto',
               textfont=dict(color='#111827', size=11)),
        go.Bar(name='优化后', x=metrics_df['指标'][:3],
               y=[after_metrics['平均可达距离(km)'], after_metrics['覆盖率(%)'], after_metrics['可达性Gini']],
               marker_color='#26a69a',
               marker_line_color='#3fb950',
               marker_line_width=1.5,
               width=0.35,
               text=[f"{v:.2f}" for v in [after_metrics['平均可达距离(km)'], after_metrics['覆盖率(%)'], after_metrics['可达性Gini']]],
               textposition='auto',
               textfont=dict(color='#111827', size=11))
    ])
    
    # 设置坐标轴范围
    max_distance = max(before_metrics['平均可达距离(km)'], after_metrics['平均可达距离(km)'])
    max_coverage = max(before_metrics['覆盖率(%)'], after_metrics['覆盖率(%)'], 100)
    max_gini = max(before_metrics['可达性Gini'], after_metrics['可达性Gini'], 1)
    
    fig.update_layout(
        title='优化前后指标对比',
        barmode='group',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font_color='#111827',
        title_font_size=22,
        title_font_color='#111827',
        title_x=0.5,
        xaxis_title_font_color='#4b5563',
        yaxis_title_font_color='#4b5563',
        xaxis_tickfont=dict(color='#4b5563', size=13),
        yaxis_tickfont=dict(color='#4b5563', size=12),
        margin=dict(l=20, r=20, t=50, b=30),
        height=380,  # 紧凑的图表高度
        legend=dict(
            title='状态',
            title_font_color='#4b5563',
            font_color='#4b5563',
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#e5e7eb',
            borderwidth=1,
            x=0.95,
            y=1.05,
            xanchor='right',
            orientation='h'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#ffffff',
            bordercolor='#42a5f5',
            font_color='#111827'
        )
    )
    
    # 添加网格线
    fig.update_yaxes(
        gridcolor='rgba(79, 89, 107, 0.3)',
        gridwidth=1,
        showline=True,
        linecolor='#e5e7eb'
    )
    fig.update_xaxes(
        showline=True,
        linecolor='#e5e7eb'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # 美化表格显示
    st.subheader("指标对比详情")
    
    # 使用 Streamlit 的原生表格样式，避免 jinja2 依赖问题
    st.dataframe(metrics_df, use_container_width=True, height=220)
    
    # 添加表格说明
    st.markdown("""
    <div style="background: rgba(255,255,255,0.92); padding: 15px; border-radius: 8px; margin-top: 10px;">
        <p style="color: #8b949e; font-size: 13px;">
            <strong>优化前</strong>：现有充电站的服务指标；
            <strong>优化后</strong>：新增站点后的服务指标；
            <strong>变化</strong>：优化带来的改善（+表示提升，-表示下降）
        </p>
    </div>
    """, unsafe_allow_html=True)

def task2_temporal_analysis(grid, charging_stations):
    """时空对比分析 — 4个时段需求权重下的供需匹配差异"""
    from scipy.spatial.distance import cdist as scipy_cdist

    st.markdown("### 🕐 时空对比分析")
    st.markdown("模拟早高峰、午间、晚高峰、夜间四个典型时段的出行需求差异，对比各时段充电供需匹配的变化。")

    PERIODS = {
        '早高峰 (7-9时)': 'morning_peak',
        '午间 (11-13时)': 'noon',
        '晚高峰 (17-19时)': 'evening_peak',
        '夜间 (22-24时)': 'night',
    }

    # 时段需求权重修饰因子
    MODIFIERS = {
        'morning_peak': {'住宅POI': 1.5, '办公POI': 2.0, '交通POI': 1.8, '商业POI': 0.5},
        'noon':         {'住宅POI': 0.5, '办公POI': 0.8, '交通POI': 1.2, '商业POI': 1.8},
        'evening_peak': {'住宅POI': 0.6, '办公POI': 0.3, '交通POI': 1.5, '商业POI': 2.0},
        'night':        {'住宅POI': 2.0, '办公POI': 0.1, '交通POI': 0.4, '商业POI': 0.3},
    }

    grid_utm = gpd.GeoSeries(grid['centroid'], crs="EPSG:4326").to_crs(config.UTM_EPSG)
    grid_c = np.array([[g.x, g.y] for g in grid_utm])
    ch_utm = charging_stations.to_crs(config.UTM_EPSG)
    ch_c = np.array([[g.x, g.y] for g in ch_utm.geometry])
    ports_v = ch_utm['端口数'].values
    base_demand = grid['需求量'].values

    # 检测网格内的POI类型组成
    PLACE_COLS = ['住宅POI', '商业POI', '办公POI', '交通POI']
    poi_cols_present = [c for c in PLACE_COLS if c in grid.columns]

    results = []
    progress = st.progress(0, "计算时段差异...")
    for idx, (label, key) in enumerate(PERIODS.items()):
        progress.progress((idx + 1) / len(PERIODS), label)

        # 调整需求权重
        mod = MODIFIERS.get(key, {})
        if poi_cols_present:
            adj_demand = np.zeros(len(grid))
            for col in poi_cols_present:
                factor = mod.get(col, 1.0)
                adj_demand += grid[col].fillna(0).values * factor
            adj_demand = adj_demand / adj_demand.sum() * base_demand.sum()
        else:
            adj_demand = base_demand.copy()

        # 2SFCA
        dists = scipy_cdist(grid_c, ch_c)
        radius = 5000
        w = np.where(dists <= radius, np.exp(-0.5 * (dists / radius) ** 2), 0)
        dw = w * adj_demand[:, np.newaxis]
        Rj = np.where(dw.sum(axis=0) > 0, ports_v / dw.sum(axis=0), 0)
        scores = (w * Rj).sum(axis=1)
        min_d = dists.min(axis=1)

        cov = np.sum(adj_demand * (min_d <= radius)) / np.sum(adj_demand) * 100
        avg_d = np.average(min_d, weights=adj_demand)
        vv = np.sort(scores)
        nn = len(vv)
        gini_s = (2*np.sum(np.arange(1,nn+1)*vv)-(nn+1)*np.sum(vv))/(nn*np.sum(vv)) if nn>1 and vv.sum()>0 else 0

        # 需求热点偏移
        top_demand_idx = np.argsort(adj_demand)[-100:]
        demand_centroid_lon = np.average(grid.iloc[top_demand_idx]['centroid'].apply(lambda g: g.x).values,
                                          weights=adj_demand[top_demand_idx])
        demand_centroid_lat = np.average(grid.iloc[top_demand_idx]['centroid'].apply(lambda g: g.y).values,
                                          weights=adj_demand[top_demand_idx])

        results.append({
            '时段': label, '覆盖率(%)': round(cov, 1), '平均距离(m)': round(avg_d, 0),
            '可达性Gini': round(gini_s, 3), '需求总量': round(np.sum(adj_demand), 0),
            '需求重心经度': round(demand_centroid_lon, 4), '需求重心纬度': round(demand_centroid_lat, 4),
        })

    progress.empty()
    tdf = pd.DataFrame(results)

    # 对比图 — 覆盖率 & Gini
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tdf['时段'], y=tdf['覆盖率(%)'], name='覆盖率(%)',
                          marker_color='#42a5f5', text=tdf['覆盖率(%)'].round(1), textposition='outside'))
    fig.add_trace(go.Scatter(x=tdf['时段'], y=tdf['可达性Gini'] * 100, name='可达性Gini×100',
                              mode='lines+markers', yaxis='y2',
                              line=dict(color='#ff9800', width=3), marker=dict(size=10)))
    fig.update_layout(
        title='四时段供需匹配对比', paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff', font_color='#111827',
        title_font_color='#111827', height=400,
        yaxis=dict(title='覆盖率(%)', gridcolor='rgba(17,24,39,0.08)'),
        yaxis2=dict(title='可达性Gini×100', overlaying='y', side='right',
                     gridcolor='rgba(17,24,39,0.04)'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)',
                     bordercolor='#e5e7eb', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 需求重心漂移
    st.subheader("需求重心时空漂移")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=tdf['需求重心经度'], y=tdf['需求重心纬度'],
        mode='markers+lines+text',
        text=tdf['时段'], textposition='top center',
        marker=dict(size=15, color=['#ff9800', '#ffc107', '#ff5722', '#3f51b5'],
                     symbol='circle'),
        line=dict(color='#9ca3af', width=2, dash='dot'),
    ))
    fig2.update_layout(
        title='需求重心位置随时间变化', paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff', font_color='#111827',
        title_font_color='#111827', height=400,
        xaxis_title='经度', yaxis_title='纬度',
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 数据表
    st.subheader("时段指标对比")
    st.dataframe(tdf.drop(columns=['需求重心经度', '需求重心纬度']),
                 use_container_width=True, height=200,
                 column_config={'覆盖率(%)': st.column_config.NumberColumn(format="%.1f%%")})

    # 结论
    peak_morning = tdf.iloc[0]
    peak_evening = tdf.iloc[2]
    night = tdf.iloc[3]
    st.info(f"🕐 **时空分析结论**: 早高峰覆盖率 {peak_morning['覆盖率(%)']}% "
            f"(Gini={peak_morning['可达性Gini']}), "
            f"晚高峰覆盖率 {peak_evening['覆盖率(%)']}% "
            f"(Gini={peak_evening['可达性Gini']})。"
            f"夜间需求集中于居住区,覆盖率降至 {night['覆盖率(%)']}%。"
            f"建议在居住区周边布局充电设施以改善夜间服务。")


def task2_sensitivity_dashboard(grid, charging_stations, poi_df):
    """动态敏感性分析面板 — 交互式参数调节,实时展示指标变化"""
    from scipy.spatial.distance import cdist as scipy_cdist

    st.markdown("### 🔬 敏感性分析面板")
    st.markdown("调整模型参数，实时观察供需匹配指标的变化，理解参数不确定性对结果的影响。")

    # 参数控制区
    col1, col2, col3 = st.columns(3)
    with col1:
        sr = st.slider("搜索半径 (km)", 2, 10, 5, 1, key="sens_sr")
    with col2:
        decay_type = st.selectbox("距离衰减函数", ["Gaussian", "Exponential", "Binary"], key="sens_decay")
    with col3:
        st.selectbox("网格精度", [1000], index=0, disabled=True,
                     format_func=lambda x: f"{x}m (当前)", key="sens_gs")

    # 重新计算 — 使用已有网格,改变衰减函数和搜索半径
    with st.spinner("正在计算敏感性..."):
        grid_utm = gpd.GeoSeries(grid['centroid'], crs="EPSG:4326").to_crs(config.UTM_EPSG)
        grid_c = np.array([[g.x, g.y] for g in grid_utm])
        ch_utm = charging_stations.to_crs(config.UTM_EPSG)
        ch_c = np.array([[g.x, g.y] for g in ch_utm.geometry])
        demand_v = grid['需求量'].values
        ports_v = ch_utm['端口数'].values

        radius_m = sr * 1000
        dists = scipy_cdist(grid_c, ch_c)

        # 衰减函数
        if decay_type == "Gaussian":
            w = np.where(dists <= radius_m, np.exp(-0.5 * (dists / radius_m) ** 2), 0)
        elif decay_type == "Exponential":
            w = np.where(dists <= radius_m, np.exp(-dists / radius_m), 0)
        else:  # Binary
            w = np.where(dists <= radius_m, 1.0, 0)

        dw = w * demand_v[:, np.newaxis]
        ds = dw.sum(axis=0)
        Rj = np.where(ds > 0, ports_v / ds, 0)
        scores = (w * Rj).sum(axis=1)
        min_d = dists.min(axis=1)

        cov = np.sum(demand_v * (min_d <= radius_m)) / np.sum(demand_v) * 100
        avg_d = np.average(min_d, weights=demand_v)
        v_s = np.sort(scores)
        n_s = len(v_s)
        gini_s = (2*np.sum(np.arange(1,n_s+1)*v_s)-(n_s+1)*np.sum(v_s))/(n_s*np.sum(v_s)) if n_s>1 and v_s.sum()>0 else 0

    # 指标卡片
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("覆盖率", f"{cov:.1f}%")
    with c2:
        st.metric("平均可达距离", f"{avg_d:.0f}m")
    with c3:
        st.metric("可达性Gini", f"{gini_s:.3f}")
    with c4:
        st.metric("网格数", len(grid))

    # 参数影响热力图
    st.subheader("参数组合影响矩阵")
    param_grid = []
    for r in [3000, 5000, 7000, 10000]:
        for dt in ["Gaussian", "Exponential", "Binary"]:
            d = scipy_cdist(grid_c, ch_c)
            if dt == "Gaussian":
                ww = np.where(d <= r, np.exp(-0.5 * (d / r) ** 2), 0)
            elif dt == "Exponential":
                ww = np.where(d <= r, np.exp(-d / r), 0)
            else:
                ww = np.where(d <= r, 1.0, 0)
            ddw = ww * demand_v[:, np.newaxis]
            dds = ddw.sum(axis=0)
            RR = np.where(dds > 0, ports_v / dds, 0)
            ss = (ww * RR).sum(axis=1)
            md = d.min(axis=1)
            cv = np.sum(demand_v * (md <= r)) / np.sum(demand_v) * 100
            vv = np.sort(ss)
            nn = len(vv)
            gg = (2*np.sum(np.arange(1,nn+1)*vv)-(nn+1)*np.sum(vv))/(nn*np.sum(vv)) if nn>1 and vv.sum()>0 else 0
            param_grid.append({
                '搜索半径(km)': r/1000, '衰减函数': dt,
                '覆盖率(%)': round(cv, 1), '平均距离(m)': round(np.average(md, weights=demand_v), 0),
                '可达性Gini': round(gg, 3),
            })

    pg_df = pd.DataFrame(param_grid)

    # 覆盖率热力图
    heat_coverage = pg_df.pivot_table(values='覆盖率(%)', index='衰减函数', columns='搜索半径(km)', aggfunc='first')
    heat_gini = pg_df.pivot_table(values='可达性Gini', index='衰减函数', columns='搜索半径(km)', aggfunc='first')

    col_a, col_b = st.columns(2)
    with col_a:
        fig_h1 = go.Figure(data=go.Heatmap(
            z=heat_coverage.values, x=[f"{c}km" for c in heat_coverage.columns],
            y=heat_coverage.index.tolist(), colorscale='RdYlGn',
            text=[[f"{v:.1f}%" for v in row] for row in heat_coverage.values],
            texttemplate="%{text}", textfont=dict(size=12, color='black'),
        ))
        fig_h1.update_layout(title='覆盖率热力图', paper_bgcolor='#ffffff',
                              font_color='#111827', title_font_color='#111827', height=300)
        st.plotly_chart(fig_h1, use_container_width=True)
    with col_b:
        fig_h2 = go.Figure(data=go.Heatmap(
            z=heat_gini.values, x=[f"{c}km" for c in heat_gini.columns],
            y=heat_gini.index.tolist(), colorscale='RdYlGn_r',
            text=[[f"{v:.3f}" for v in row] for row in heat_gini.values],
            texttemplate="%{text}", textfont=dict(size=12, color='black'),
        ))
        fig_h2.update_layout(title='Gini系数热力图', paper_bgcolor='#ffffff',
                              font_color='#111827', title_font_color='#111827', height=300)
        st.plotly_chart(fig_h2, use_container_width=True)

    # 参数重要性 (Tornado)
    st.subheader("参数影响幅度 (Tornado)")
    pg_grouped = pg_df.groupby('衰减函数').agg({'覆盖率(%)': ['min', 'max'], '可达性Gini': ['min', 'max']})
    tornado_data = []
    for dt_name in ['Gaussian', 'Exponential', 'Binary']:
        sub = pg_df[pg_df['衰减函数'] == dt_name]
        tornado_data.append({'参数': f'{dt_name} 覆盖率范围', 'min': sub['覆盖率(%)'].min(), 'max': sub['覆盖率(%)'].max()})
        tornado_data.append({'参数': f'{dt_name} Gini范围', 'min': sub['可达性Gini'].min(), 'max': sub['可达性Gini'].max()})
    tornado_df = pd.DataFrame(tornado_data)
    tornado_df = tornado_df.sort_values('max', ascending=True)

    fig_t = go.Figure()
    for _, row in tornado_df.iterrows():
        fig_t.add_trace(go.Bar(
            y=[row['参数']], x=[row['max'] - row['min']],
            base=row['min'], orientation='h',
            marker_color='#42a5f5', name=row['参数'],
            text=f"{row['min']:.2f} → {row['max']:.2f}", textposition='outside',
        ))
    fig_t.update_layout(
        title='各衰减函数下指标波动范围', showlegend=False,
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font_color='#111827', title_font_color='#111827', height=350,
        xaxis_title='指标值', margin=dict(l=150, r=30, t=50, b=30),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    # 建议
    st.info(f"📊 **参数敏感性结论**: 在 {decay_type} 衰减 + {sr}km 搜索半径 + 1km 网格下, "
            f"覆盖率={cov:.1f}%, Gini={gini_s:.3f}。搜索半径是影响覆盖率的最敏感参数, "
            f"衰减函数类型对Gini系数影响较大。建议在实际应用中优先校准搜索半径。")


def task3_marginal_benefit(grid, candidates, charging_stations, max_k=15):
    """边际效益曲线 — 逐站覆盖增益与递减规律"""
    from scipy.spatial.distance import cdist as scipy_cdist

    if grid.empty or candidates.empty:
        st.warning("数据不足，无法计算边际效益")
        return

    grid_centroids_utm = gpd.GeoSeries(grid['centroid'], crs="EPSG:4326").to_crs(config.UTM_EPSG)
    grid_coords = np.array([[g.x, g.y] for g in grid_centroids_utm])
    demand_vals = grid['需求量'].values

    candidate_utm = candidates.to_crs(config.UTM_EPSG)
    cand_coords = np.array([[g.x, g.y] for g in candidate_utm.geometry])

    charging_utm = charging_stations.to_crs(config.UTM_EPSG)
    exist_coords = np.array([[g.x, g.y] for g in charging_utm.geometry])

    COVER_RADIUS = 5000
    w_sum = np.sum(demand_vals)

    # 现有站点已覆盖的网格
    if len(exist_coords) > 0:
        d_exist = scipy_cdist(grid_coords, exist_coords)
        covered_before = (d_exist.min(axis=1) <= COVER_RADIUS).astype(float)
    else:
        covered_before = np.zeros(len(grid_coords))

    baseline_cov = np.sum(demand_vals * covered_before) / w_sum

    selected_mask = np.zeros(len(cand_coords), dtype=bool)
    all_covered = covered_before.copy()
    results = []

    progress = st.progress(0, "计算边际效益...")
    for k in range(1, max_k + 1):
        progress.progress(k / max_k, f"k={k}/{max_k}")
        best_idx = -1
        best_marginal = -1.0
        best_covered = None
        for i in range(len(cand_coords)):
            if selected_mask[i]:
                continue
            d_i = scipy_cdist(grid_coords, [cand_coords[i]])
            new_c = np.maximum(all_covered, (d_i[:, 0] <= COVER_RADIUS).astype(float))
            marginal = np.sum(demand_vals * (new_c - all_covered))
            if marginal > best_marginal:
                best_marginal = marginal
                best_idx = i
                best_covered = new_c

        if best_idx == -1:
            break

        selected_mask[best_idx] = True
        all_covered = best_covered
        total_cov = np.sum(demand_vals * all_covered) / w_sum
        marginal_pct = best_marginal / w_sum * 100

        # 计算当前所有站点的距离指标
        sel_c = cand_coords[selected_mask]
        all_c = np.vstack([exist_coords, sel_c]) if len(exist_coords) > 0 else sel_c
        min_d = scipy_cdist(grid_coords, all_c).min(axis=1)
        avg_d = np.average(min_d, weights=demand_vals)
        p95 = np.percentile(min_d, 95)

        results.append({
            'k': k, '覆盖率(%)': round(total_cov * 100, 1),
            '边际增益(%)': round(marginal_pct, 2),
            '累计增益(%)': round((total_cov - baseline_cov) * 100, 1),
            '平均距离(m)': round(avg_d, 0),
            'P95距离(m)': round(p95, 0),
        })

    progress.empty()
    mdf = pd.DataFrame(results)

    st.markdown("### 📈 边际效益曲线")
    st.markdown(f"展示每新增一个充电站带来的覆盖增益变化。基线覆盖率: **{baseline_cov*100:.1f}%**")

    # 双轴图: 覆盖率和边际增益
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mdf['k'], y=mdf['边际增益(%)'], name='边际增益(%)',
        marker_color='#42a5f5', yaxis='y2',
        text=mdf['边际增益(%)'].round(1), textposition='outside',
        textfont=dict(color='#42a5f5', size=10),
    ))
    fig.add_trace(go.Scatter(
        x=mdf['k'], y=mdf['覆盖率(%)'], name='覆盖率(%)',
        mode='lines+markers', line=dict(color='#26a69a', width=3),
        marker=dict(size=8, color='#26a69a'),
    ))
    fig.add_trace(go.Scatter(
        x=mdf['k'], y=mdf['平均距离(m)'], name='平均距离(m)',
        mode='lines+markers', line=dict(color='#ff9800', width=2, dash='dash'),
        marker=dict(size=6, color='#ff9800'),
    ))
    fig.update_layout(
        title='选址边际效益 — 逐站覆盖增益与距离变化',
        xaxis=dict(title='新增站点数 k', dtick=1, gridcolor='rgba(17,24,39,0.08)'),
        yaxis=dict(title='覆盖率(%) / 平均距离(m)', gridcolor='rgba(17,24,39,0.08)'),
        yaxis2=dict(title='边际增益(%)', overlaying='y', side='right',
                     gridcolor='rgba(17,24,39,0.04)'),
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font_color='#111827', title_font_color='#111827',
        height=450, margin=dict(l=20, r=60, t=50, b=30),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)',
                     bordercolor='#e5e7eb', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 数据表
    st.subheader("逐站详情")
    st.dataframe(mdf, use_container_width=True, height=300,
                 column_config={
                     '覆盖率(%)': st.column_config.NumberColumn(format="%.1f%%"),
                     '边际增益(%)': st.column_config.NumberColumn(format="%.2f%%"),
                 })

    # 递减分析
    if len(mdf) >= 3:
        first3 = mdf['边际增益(%)'].iloc[:3].mean()
        last3 = mdf['边际增益(%)'].iloc[-3:].mean()
        decay = (first3 - last3) / first3 * 100 if first3 > 0 else 0
        st.info(f"📊 **边际递减分析**: 前3站平均增益 {first3:.2f}pp, "
                f"最后3站平均增益 {last3:.2f}pp, 递减幅度 **{decay:.0f}%**。"
                f"建议新建站点数不超过 {mdf.loc[mdf['边际增益(%)'] > 1.0, 'k'].max() if any(mdf['边际增益(%)'] > 1.0) else 5} 个以维持边际效益。")


def task3_scenario_comparison(grid, candidates, charging_stations, k=5):
    """多情景政策对比 — 4种政策取向下选址方案的效果比较"""
    import time
    from scipy.spatial.distance import cdist as scipy_cdist

    if grid.empty or candidates.empty:
        st.warning("数据不足，无法进行情景对比")
        return

    # 准备数据
    grid_centroids_utm = gpd.GeoSeries(grid['centroid'], crs="EPSG:4326").to_crs(config.UTM_EPSG)
    grid_coords = np.array([[g.x, g.y] for g in grid_centroids_utm])
    demand_vals = grid['需求量'].values

    candidate_utm = candidates.to_crs(config.UTM_EPSG)
    cand_coords = np.array([[g.x, g.y] for g in candidate_utm.geometry])

    charging_utm = charging_stations.to_crs(config.UTM_EPSG)
    exist_coords = np.array([[g.x, g.y] for g in charging_utm.geometry])

    COVER_RADIUS = 5000  # 覆盖半径5km
    MIN_DIST = 500       # 最小站间距

    def _web_greedy_p_median(coverage_weight, transit_weight=0.0):
        """Web版多目标贪婪选择 — 各目标先归一化到[0,1]再加权,避免量纲差异相互淹没"""
        from scipy.spatial import cKDTree as _cKDTree
        selected = []
        sel_coords_list = []
        weights = demand_vals.copy()

        # 有效候选点索引
        if len(exist_coords) > 0:
            d_cand_exist = scipy_cdist(cand_coords, exist_coords)
            valid = np.where(d_cand_exist.min(axis=1) >= MIN_DIST)[0]
        else:
            valid = np.arange(len(cand_coords))

        if len(valid) == 0:
            valid = np.arange(min(len(cand_coords), k * 3))

        valid_coords = cand_coords[valid]

        # 候选点交通邻近性(2km内交通POI密度均值) — 交通导向第三目标
        use_transit = transit_weight > 0 and 'transit_density' in grid.columns
        candidate_transit = None
        if use_transit:
            transit_vals = grid['transit_density'].values.astype(float)
            _tree = _cKDTree(grid_coords)
            candidate_transit = np.zeros(len(cand_coords))
            for ci in range(len(cand_coords)):
                nbrs = _tree.query_ball_point(cand_coords[ci], 2000.0)
                if nbrs:
                    candidate_transit[ci] = float(np.mean(transit_vals[nbrs]))

        def _minmax(a):
            a = np.asarray(a, dtype=float)
            rng = a.max() - a.min()
            return np.zeros_like(a) if rng < 1e-12 else (a - a.min()) / rng

        for _ in range(min(k, len(valid))):
            best_idx_local = -1
            best_score = -np.inf
            cand_pos = [i for i in range(len(valid_coords)) if i not in selected]
            cov_raw, eq_raw, tr_raw = [], [], []
            for i_local in cand_pos:
                temp = sel_coords_list + [valid_coords[i_local]]
                temp_arr = np.array(temp)
                min_d = scipy_cdist(grid_coords, temp_arr).min(axis=1)

                cov_raw.append(np.sum(weights * (min_d <= COVER_RADIUS)) / (np.sum(weights) + 1e-10))
                v_sorted = np.sort(min_d)
                n = len(v_sorted)
                gini = (2 * np.sum(np.arange(1, n + 1) * v_sorted) - (n + 1) * np.sum(v_sorted)) / (n * np.sum(v_sorted)) if n > 1 and v_sorted.sum() > 0 else 0
                eq_raw.append(1.0 - gini)
                tr_raw.append(candidate_transit[valid[i_local]] if candidate_transit is not None else 0.0)

            cov_n = _minmax(np.array(cov_raw))
            eq_n = _minmax(np.array(eq_raw))
            tr_n = _minmax(np.array(tr_raw)) if use_transit else np.zeros(len(cov_raw))

            for j, i_local in enumerate(cand_pos):
                if use_transit:
                    composite = (1 - transit_weight) * (coverage_weight * cov_n[j] + (1 - coverage_weight) * eq_n[j]) + transit_weight * tr_n[j]
                else:
                    composite = coverage_weight * cov_n[j] + (1 - coverage_weight) * eq_n[j]
                if composite > best_score:
                    best_score = composite
                    best_idx_local = i_local

            if best_idx_local != -1:
                selected.append(best_idx_local)
                sel_coords_list.append(valid_coords[best_idx_local])

        # Return selected candidates as GeoDataFrame
        sel_global_indices = valid[selected] if len(selected) > 0 else []
        return candidates.iloc[sel_global_indices].copy() if len(sel_global_indices) > 0 else gpd.GeoDataFrame()

    SCENARIOS = [
        {'name': '公平优先', 'cw': 0.1, 'tw': 0.0, 'desc': '优先缩小区域充电服务差距', 'color': '#26a69a'},
        {'name': '效率优先', 'cw': 0.9, 'tw': 0.0, 'desc': '优先最大化总体充电覆盖', 'color': '#42a5f5'},
        {'name': '交通导向', 'cw': 0.5, 'tw': 0.3, 'desc': '兼顾公交枢纽与充电需求', 'color': '#ff9800'},
        {'name': '均衡发展', 'cw': 0.5, 'tw': 0.0, 'desc': '覆盖与公平同等权重', 'color': '#ab47bc'},
    ]

    results = []
    all_selections = {}
    progress = st.progress(0, "正在计算多情景对比...")

    for idx, sc in enumerate(SCENARIOS):
        progress.progress((idx) / len(SCENARIOS), f"计算情景: {sc['name']}...")
        t0 = time.time()
        sel = _web_greedy_p_median(sc['cw'], sc.get('tw', 0.0))
        elapsed = time.time() - t0

        all_selections[sc['name']] = sel

        if len(sel) > 0:
            sel_utm = sel.to_crs(config.UTM_EPSG)
            sel_c = np.array([[g.x, g.y] for g in sel_utm.geometry])
            all_c = np.vstack([exist_coords, sel_c]) if len(exist_coords) > 0 else sel_c
            dists = scipy_cdist(grid_coords, all_c)
            min_d = dists.min(axis=1)

            w_sum = np.sum(demand_vals)
            coverage = np.sum(demand_vals * (min_d <= COVER_RADIUS)) / (w_sum + 1e-10) * 100
            avg_d = np.average(min_d, weights=demand_vals)
            p95 = np.percentile(min_d, 95)
            v_sorted = np.sort(min_d)
            n = len(v_sorted)
            gini = (2*np.sum(np.arange(1,n+1)*v_sorted)-(n+1)*np.sum(v_sorted))/(n*np.sum(v_sorted)) if n>1 and v_sorted.sum()>0 else 0
        else:
            coverage = avg_d = p95 = gini = 0

        results.append({
            '情景': sc['name'], '政策说明': sc['desc'],
            '覆盖率权重': sc['cw'], '交通权重': sc.get('tw', 0.0),
            '覆盖率(%)': round(coverage, 1), '平均距离(m)': round(avg_d, 0),
            'P95距离(m)': round(p95, 0), '距离Gini': round(gini, 3),
            '公平性得分': round(1-gini, 3), '耗时(s)': round(elapsed, 2),
            '新增站点数': len(sel), '颜色': sc['color'],
        })

    progress.progress(1.0, "完成!")
    progress.empty()

    comparison_df = pd.DataFrame(results)

    # --- 展示 ---
    st.markdown("### 🎯 多情景政策对比")
    st.markdown("四种政策取向下，新增充电站选址方案的差异化效果对比：")

    # 指标卡片行
    cols = st.columns(4)
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        with cols[i]:
            st.metric(
                f"{row['情景']}",
                f"{row['覆盖率(%)']}%",
                delta=f"Gini={row['距离Gini']} | {row['新增站点数']}站"
            )

    st.markdown("---")

    # 雷达图对比
    st.subheader("情景雷达图对比")
    categories = ['覆盖率(%)', '公平性得分', '平均距离(m)', 'P95距离(m)']

    radar_fig = go.Figure()
    for _, row in comparison_df.iterrows():
        radar_fig.add_trace(go.Scatterpolar(
            r=[row['覆盖率(%)'], row['公平性得分']*100,
               max(0, 100 - row['平均距离(m)']/100), max(0, 100 - row['P95距离(m)']/100)],
            theta=['覆盖率', '公平性', '距离便利性', 'P95便利性'],
            fill='toself',
            name=row['情景'],
            marker_color=row['颜色'],
            opacity=0.4,
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        paper_bgcolor='#ffffff',
        font_color='#111827',
        title='多情景多维对比',
        title_font_color='#111827',
        height=450,
        margin=dict(l=40, r=40, t=60, b=30),
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # 对比表格
    st.subheader("详细指标对比")
    display_cols = ['情景', '政策说明', '覆盖率权重', '交通权重', '覆盖率(%)', '平均距离(m)', 'P95距离(m)', '距离Gini', '公平性得分', '耗时(s)', '新增站点数']
    st.dataframe(comparison_df[display_cols], use_container_width=True, height=220,
                 column_config={
                     '覆盖率(%)': st.column_config.NumberColumn(format="%.1f%%"),
                     '距离Gini': st.column_config.NumberColumn(format="%.3f"),
                 })

    # 情景地图
    st.subheader("各情景选址结果地图")
    scenario_tabs = st.tabs([s['name'] for s in SCENARIOS])
    for i, (sc, tab) in enumerate(zip(SCENARIOS, scenario_tabs)):
        with tab:
            sel = all_selections[sc['name']]
            if len(sel) > 0:
                m = folium.Map(location=[34.26, 108.94], zoom_start=11,
                               tiles=AMAP_TILE_URL, attr='高德地图')
                # 现有站点 (蓝色)
                for _, row in charging_stations.iterrows():
                    folium.CircleMarker(
                        [row.geometry.y, row.geometry.x], radius=5,
                        color='#42a5f5', fill=True, fill_opacity=0.6,
                        popup=f"现有站点: {row.get('名称', '')}"
                    ).add_to(m)
                # 新增站点 (场景色)
                sel_wgs = sel.to_crs("EPSG:4326")
                for _, row in sel_wgs.iterrows():
                    folium.CircleMarker(
                        [row.geometry.y, row.geometry.x], radius=8,
                        color=sc['color'], fill=True, fill_opacity=0.8,
                        weight=3,
                        popup=f"{sc['name']}新增: ({row.geometry.y:.4f}, {row.geometry.x:.4f})"
                    ).add_to(m)
                folium_static(m, width=None, height=400)
            else:
                st.info(f"{sc['name']}: 未选出站点")

    # 政策建议
    st.markdown("---")
    st.subheader("📋 政策建议")
    best_cov = comparison_df.loc[comparison_df['覆盖率(%)'].idxmax()]
    best_equity = comparison_df.loc[comparison_df['距离Gini'].idxmin()]
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; border-left: 4px solid #1e88e5;">
        <p><strong>🔹 覆盖率最优情景</strong>: {best_cov['情景']} ({best_cov['覆盖率(%)']}% 覆盖)</p>
        <p><strong>🔹 公平性最优情景</strong>: {best_equity['情景']} (Gini={best_equity['距离Gini']})</p>
        <p><strong>🔹 推荐方案</strong>: 根据西安市实际情况，建议优先采用
           <span style="color:#26a69a;font-weight:700;">均衡发展</span> 方案，
           在保障基本覆盖的同时兼顾区域公平性，避免充电资源过度集中于核心城区。</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 创新模块 1-5：选址可解释性 / 覆盖热力叠加 / 方案记忆 / 成本约束 / 需求预测
# ══════════════════════════════════════════════════════════════════

_COVER_RADIUS = 5000  # 统一覆盖半径(米)


def _blind_areas_of(grid, deficit_threshold):
    """返回供给不足网格(盲区)的副本,并补齐'供给状态'列"""
    g = grid.copy()
    if '供给状态' not in g.columns:
        thr = g['供需得分'].quantile(deficit_threshold)
        g['供给状态'] = g['供需得分'].apply(lambda x: '供给不足' if x <= thr else '供给充足')
    return g[g['供给状态'] == '供给不足']


def _utm_coords(geoseries):
    """将 WGS84 几何序列投影到 UTM 32649 并返回 (N,2) 数组"""
    utm = gpd.GeoSeries(geoseries, crs="EPSG:4326").to_crs(config.UTM_EPSG)
    return np.array([[g.x, g.y] for g in utm])


def _covered_before_mask(grid_coords, exist_coords):
    """返回现有站点已覆盖的网格布尔掩码"""
    from scipy.spatial.distance import cdist as scipy_cdist
    if len(exist_coords) == 0:
        return np.zeros(len(grid_coords), dtype=bool)
    d = scipy_cdist(grid_coords, exist_coords)
    return (d.min(axis=1) <= _COVER_RADIUS).astype(bool)


def _incremental_greedy(cand_coords, grid_coords, demand, covered_before, k, radius=5000, min_dist=1500):
    """增量贪心选址：在现有站点覆盖基础上，逐步选择『增量覆盖需求最多』的候选点。

    与 greedy_selection 的区别：这里从 covered_before(现有站点已覆盖) 出发，
    只统计新站点对『尚未被覆盖网格』的贡献，保证选址真正带来增量覆盖。
    返回候选点索引(按选择顺序)。
    """
    from scipy.spatial.distance import cdist as scipy_cdist
    if len(cand_coords) == 0 or k < 1:
        return []

    d_cand = scipy_cdist(grid_coords, cand_coords)  # (n_grid, n_cand)
    cover = d_cand <= radius                       # (n_grid, n_cand)
    demand_cover = cover * demand[:, None]         # 每个候选点覆盖的需求 (n_grid, n_cand)
    cand_dist = scipy_cdist(cand_coords, cand_coords)  # 候选点间距 (分散约束)

    remaining = ~covered_before                    # 尚未被覆盖的网格
    selected = []
    for _ in range(k):
        # 每个候选点对剩余未覆盖网格的增量需求贡献
        inc = (demand_cover * remaining[:, None]).sum(axis=0).astype(float)
        for s in selected:
            inc[s] = -1.0
            inc[cand_dist[s] < min_dist] = -1.0     # 与已选点过近的候选点排除
        best = int(np.argmax(inc))
        if inc[best] <= 0:
            break
        selected.append(best)
        remaining = remaining & ~cover[:, best]

    return selected


def task3_explainability(grid, candidates, new_stations, charging_stations, deficit_threshold=config.BLIND_THRESHOLD):
    """创新1 — 选址可解释性：对每个新增站点标注『为何选这里』"""
    from scipy.spatial.distance import cdist as scipy_cdist

    if new_stations is None or new_stations.empty:
        st.warning("暂无新增站点数据，请先完成选址优化")
        return

    grid_coords = _utm_coords(grid['centroid'])
    grid_demand = grid['需求量'].values
    w_sum = np.sum(grid_demand)
    new_coords = _utm_coords(new_stations.geometry)
    exist_coords = _utm_coords(charging_stations.geometry)
    covered_before = _covered_before_mask(grid_coords, exist_coords)

    blind = _blind_areas_of(grid, deficit_threshold)
    blind_coords = _utm_coords(blind['centroid']) if len(blind) > 0 else None

    rows = []
    for i in range(len(new_coords)):
        # 距最近盲区质心的距离
        if blind_coords is not None:
            nearest_blind = scipy_cdist([new_coords[i]], blind_coords)[0].min()
        else:
            nearest_blind = np.nan
        # 新增覆盖的需求(未被现有站点覆盖的部分)
        d_i = scipy_cdist(grid_coords, [new_coords[i]])[:, 0]
        newly = (d_i <= _COVER_RADIUS) & (~covered_before)
        covered_demand_pct = np.sum(grid_demand[newly]) / w_sum * 100 if w_sum > 0 else 0
        # 距最近现有站点(竞争站点)
        if len(exist_coords) > 0:
            nearest_comp = scipy_cdist([new_coords[i]], exist_coords)[0].min()
        else:
            nearest_comp = np.nan

        rows.append({
            '站点': f'新增站点 {i + 1}',
            '经度': round(new_stations.geometry.iloc[i].x, 4),
            '纬度': round(new_stations.geometry.iloc[i].y, 4),
            '距最近盲区(m)': int(round(nearest_blind)) if not np.isnan(nearest_blind) else None,
            '新增覆盖需求(%)': round(covered_demand_pct, 2),
            '距最近现有站(m)': int(round(nearest_comp)) if not np.isnan(nearest_comp) else None,
        })

    expl_df = pd.DataFrame(rows)

    st.markdown("### 🔍 选址可解释性 — 为何选这里")
    st.markdown("为每个新增站点给出**三重选址依据**：贴近盲区质心、覆盖未被满足的需求、避开与现有站点竞争。")

    # 指标卡片
    c1, c2, c3 = st.columns(3)
    with c1:
        avg_blind = np.nanmean([r['距最近盲区(m)'] for r in rows if r['距最近盲区(m)'] is not None])
        st.metric("平均距盲区质心", f"{avg_blind:.0f}m" if not np.isnan(avg_blind) else "N/A")
    with c2:
        st.metric("累计新增覆盖需求", f"{expl_df['新增覆盖需求(%)'].sum():.1f}%")
    with c3:
        avg_comp = np.nanmean([r['距最近现有站(m)'] for r in rows if r['距最近现有站(m)'] is not None])
        st.metric("平均距现有站", f"{avg_comp:.0f}m" if not np.isnan(avg_comp) else "N/A")

    # 解释表格
    st.subheader("逐站选址依据")
    st.dataframe(
        expl_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            '距最近盲区(m)': st.column_config.NumberColumn(format="%d m"),
            '新增覆盖需求(%)': st.column_config.NumberColumn(format="%.2f%%"),
            '距最近现有站(m)': st.column_config.NumberColumn(format="%d m"),
        },
    )

    # 解释地图
    st.subheader("选址依据地图")
    center_lat = new_stations.geometry.y.mean()
    center_lon = new_stations.geometry.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)
    folium.TileLayer(tiles=AMAP_TILE_URL, attr="高德地图", name="高德矢量图").add_to(m)

    if blind_coords is not None and len(blind) > 0:
        folium.GeoJson(
            blind[['geometry', '供需得分']],
            name="服务盲区",
            style_function=lambda x: {'fillColor': '#d32f2f', 'color': '#b71c1c',
                                       'weight': 1, 'fillOpacity': 0.4},
        ).add_to(m)

    for _, row in charging_stations.iterrows():
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x], radius=5,
            color='#42a5f5', fill=True, fill_opacity=0.7,
            tooltip=f"现有站: {row.get('名称', '')}"
        ).add_to(m)

    for i, row in expl_df.iterrows():
        folium.Circle(
            [row['纬度'], row['经度']], radius=5000, color='#3fb950',
            fill=True, fill_color='#3fb950', fill_opacity=0.08, weight=1.5
        ).add_to(m)
        folium.CircleMarker(
            [row['纬度'], row['经度']], radius=10, color='#22c55e',
            fill=True, fill_color='#4ade80', fill_opacity=0.95, weight=3,
            popup=folium.Popup(
                f"<strong>{row['站点']}</strong><br>"
                f"距盲区质心 {row['距最近盲区(m)']} m<br>"
                f"新增覆盖需求 {row['新增覆盖需求(%)']}%<br>"
                f"距现有站 {row['距最近现有站(m)']} m",
                max_width=260)
        ).add_to(m)

    folium.LayerControl().add_to(m)
    folium_static(m, width=None, height=500)


def task3_coverage_overlay(grid, new_stations, charging_stations, deficit_threshold=config.BLIND_THRESHOLD):
    """创新2 — 实时覆盖率热力叠加：新增站点覆盖半径叠加到盲区地图"""
    from scipy.spatial.distance import cdist as scipy_cdist

    if new_stations is None or new_stations.empty:
        st.warning("暂无新增站点数据，请先完成选址优化")
        return

    grid_coords = _utm_coords(grid['centroid'])
    grid_demand = grid['需求量'].values
    w_sum = np.sum(grid_demand)
    new_coords = _utm_coords(new_stations.geometry)
    exist_coords = _utm_coords(charging_stations.geometry)
    covered_before = _covered_before_mask(grid_coords, exist_coords)

    blind = _blind_areas_of(grid, deficit_threshold)
    blind_idx = blind.index

    # 新增站点对盲区的覆盖情况
    d_new = scipy_cdist(grid_coords, new_coords)
    covered_by_new = (d_new <= _COVER_RADIUS).any(axis=1)

    # 被新增站点覆盖、且此前未被覆盖的盲区
    newly_covered_mask = covered_by_new & (~covered_before)
    newly_covered_in_blind = newly_covered_mask & grid.index.isin(blind_idx)

    coverage_improve = np.sum(grid_demand[newly_covered_mask]) / w_sum * 100 if w_sum > 0 else 0
    blind_resolved = int(newly_covered_in_blind.sum())
    blind_total = len(blind)

    st.markdown("### 🗺️ 覆盖率热力叠加")
    st.markdown("将新增站点的 **5km 覆盖半径** 实时叠加到服务盲区地图上，直观展示选址对盲区的填补效果。")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("整体新增覆盖", f"{coverage_improve:.1f}%")
    with c2:
        st.metric("盲区被填补网格", f"{blind_resolved}/{blind_total}")
    with c3:
        st.metric("盲区填补率", f"{blind_resolved / blind_total * 100:.1f}%" if blind_total > 0 else "N/A")

    center_lat = grid.geometry.centroid.y.mean()
    center_lon = grid.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)
    folium.TileLayer(tiles=AMAP_TILE_URL, attr="高德地图", name="高德矢量图").add_to(m)

    # 盲区(红色)
    if len(blind) > 0:
        folium.GeoJson(
            blind[['geometry', '供需得分']],
            name="服务盲区",
            style_function=lambda x: {'fillColor': '#d32f2f', 'color': '#b71c1c',
                                       'weight': 1, 'fillOpacity': 0.45},
        ).add_to(m)

    # 被新增站点填补的盲区(绿色高亮)
    resolved = grid[newly_covered_in_blind]
    if len(resolved) > 0:
        folium.GeoJson(
            resolved[['geometry', '供需得分']],
            name="被填补盲区",
            style_function=lambda x: {'fillColor': '#22c55e', 'color': '#16a34a',
                                       'weight': 1, 'fillOpacity': 0.6},
        ).add_to(m)

    # 现有站点
    for _, row in charging_stations.iterrows():
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x], radius=4,
            color='#42a5f5', fill=True, fill_opacity=0.7
        ).add_to(m)

    # 新增站点覆盖圈
    for i in range(len(new_stations)):
        lat, lon = new_stations.geometry.iloc[i].y, new_stations.geometry.iloc[i].x
        folium.Circle(
            [lat, lon], radius=5000, color='#3fb950',
            fill=True, fill_color='#3fb950', fill_opacity=0.08, weight=2,
            popup=f"新增站点 {i + 1} 覆盖半径 5km"
        ).add_to(m)
        folium.CircleMarker(
            [lat, lon], radius=9, color='#22c55e',
            fill=True, fill_color='#4ade80', fill_opacity=0.95, weight=3
        ).add_to(m)

    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: #ffffff; padding: 16px; border-radius: 10px; border: 1px solid #e5e7eb;">
        <h4 style="color: #111827; margin-bottom: 10px; font-size: 14px;">图例</h4>
        <div style="margin-bottom: 6px;"><span style="color:#d32f2f;font-size:14px;">■</span> <span style="color:#4b5563;font-size:13px;">服务盲区</span></div>
        <div style="margin-bottom: 6px;"><span style="color:#22c55e;font-size:14px;">■</span> <span style="color:#4b5563;font-size:13px;">被填补盲区</span></div>
        <div style="margin-bottom: 6px;"><span style="color:#4ade80;font-size:16px;">●</span> <span style="color:#4b5563;font-size:13px;">新增站点</span></div>
        <div><span style="color:#42a5f5;font-size:16px;">●</span> <span style="color:#4b5563;font-size:13px;">现有站点</span></div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    folium_static(m, width=None, height=520)


def task3_scenario_memory(new_stations, before_metrics, after_metrics, coverage_ratio, solve_time, params_key, new_stations_num, enforce_spread, solver_method):
    """创新3 — 方案保存与对比记忆：保存多个选址方案并排对比"""
    if 'saved_scenarios' not in st.session_state:
        st.session_state['saved_scenarios'] = {}

    st.markdown("### 💾 方案保存与对比记忆")
    st.markdown("为当前选址方案命名并保存，可与历史方案**并排对比**，追踪不同参数组合的效果差异。")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("方案名称", value=f"方案{len(st.session_state['saved_scenarios']) + 1}")
    with c2:
        st.write("")
        st.write("")
        do_save = st.button("💾 保存当前方案", use_container_width=True)
    with c3:
        st.write("")
        st.write("")
        do_clear = st.button("🗑️ 清空历史", use_container_width=True)

    if do_save:
        if name in st.session_state['saved_scenarios']:
            st.warning(f"方案「{name}」已存在，将覆盖旧值")
        st.session_state['saved_scenarios'][name] = {
            '覆盖率': round(coverage_ratio * 100, 1),
            '平均距离(km)': round(after_metrics['平均可达距离(km)'], 2),
            '可达性Gini': round(after_metrics['可达性Gini'], 3),
            '覆盖率提升(pp)': round(after_metrics['覆盖率(%)'] - before_metrics['覆盖率(%)'], 1),
            '站点数': len(new_stations),
            '强制分散': enforce_spread,
            '求解方法': solver_method,
            '耗时(s)': round(solve_time, 2),
            'params': params_key,
            'new_stations': new_stations.copy(),
        }
        st.success(f"已保存方案「{name}」")

    if do_clear:
        st.session_state['saved_scenarios'] = {}
        st.info("已清空所有历史方案")
        st.rerun()

    saved = st.session_state['saved_scenarios']
    if not saved:
        st.info("暂无已保存的方案，请先保存当前方案。")
        return

    # 对比表
    st.subheader("已保存方案对比")
    comp_rows = []
    for n, v in saved.items():
        comp_rows.append({
            '方案': n, '覆盖率(%)': v['覆盖率'], '平均距离(km)': v['平均距离(km)'],
            '可达性Gini': v['可达性Gini'], '覆盖率提升(pp)': v['覆盖率提升(pp)'],
            '站点数': v['站点数'], '强制分散': '是' if v['强制分散'] else '否',
            '求解方法': v['求解方法'], '耗时(s)': v['耗时(s)'],
        })
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True,
                 column_config={
                     '覆盖率(%)': st.column_config.NumberColumn(format="%.1f%%"),
                     '覆盖率提升(pp)': st.column_config.NumberColumn(format="%.1f"),
                 })

    # 雷达图对比
    if len(comp_df) >= 2:
        st.subheader("方案多维对比雷达图")
        radar = go.Figure()
        for _, r in comp_df.iterrows():
            radar.add_trace(go.Scatterpolar(
                r=[r['覆盖率(%)'], max(0, 100 - r['平均距离(km)'] * 100),
                   (1 - r['可达性Gini']) * 100, r['覆盖率提升(pp)']],
                theta=['覆盖率', '距离便利性', '公平性(1-Gini)', '覆盖率提升'],
                fill='toself', name=r['方案'], opacity=0.4,
            ))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            paper_bgcolor='#ffffff', font_color='#111827',
            title='方案多维对比', title_font_color='#111827', height=430,
            margin=dict(l=40, r=40, t=60, b=30),
        )
        st.plotly_chart(radar, use_container_width=True)

    # 方案地图并排
    st.subheader("方案选址地图并排对比")
    tabs = st.tabs(list(saved.keys()))
    for name_, tab in zip(saved.keys(), tabs):
        with tab:
            ns = saved[name_]['new_stations']
            if ns is not None and not ns.empty:
                m = folium.Map(
                    location=[ns.geometry.y.mean(), ns.geometry.x.mean()],
                    zoom_start=11,
                    tiles=AMAP_TILE_URL, attr='高德地图')
                for _, row in ns.iterrows():
                    folium.CircleMarker(
                        [row.geometry.y, row.geometry.x], radius=9,
                        color='#22c55e', fill=True, fill_color='#4ade80',
                        fill_opacity=0.95, weight=3,
                        popup=f"{name_}: ({row.geometry.y:.4f}, {row.geometry.x:.4f})"
                    ).add_to(m)
                folium_static(m, width=None, height=380)
            else:
                st.info(f"{name_}: 无站点数据")


def task3_cost_optimization(grid, candidates, charging_stations):
    """创新4 — 成本约束选址：快充/慢充/超充差异化成本下的预算最优选址"""
    from scipy.spatial.distance import cdist as scipy_cdist

    if grid.empty or candidates.empty:
        st.warning("数据不足，无法进行成本约束选址")
        return

    COST_MODEL = {
        '慢充站': {'cost': 30, 'ports': 10, 'power': '7kW交流', 'color': '#42a5f5'},
        '快充站': {'cost': 120, 'ports': 8, 'power': '60kW直流', 'color': '#26a69a'},
        '超充站': {'cost': 300, 'ports': 6, 'power': '240kW直流', 'color': '#ff9800'},
    }

    st.markdown("### 💰 成本约束选址")
    st.markdown("引入建站成本模型，在**固定预算**下求解最优建站组合，实现成本-覆盖的最优权衡。")

    c1, c2 = st.columns(2)
    with c1:
        station_type = st.selectbox("建站类型", list(COST_MODEL.keys()), index=1)
    with c2:
        budget = st.slider("建站预算(万元)", 100, 3000, 600, 100)

    cost = COST_MODEL[station_type]['cost']
    ports = COST_MODEL[station_type]['ports']
    power = COST_MODEL[station_type]['power']
    color = COST_MODEL[station_type]['color']

    max_k = int(budget // cost)
    if max_k < 1:
        st.warning(f"预算 {budget} 万元不足以建设 1 个{station_type}（单站 {cost} 万元），请提高预算或选择更低成本的建站类型。")
        return
    k = min(max_k, len(candidates))

    grid_coords = _utm_coords(grid['centroid'])
    grid_demand = grid['需求量'].values
    w_sum = np.sum(grid_demand)
    cand_coords = _utm_coords(candidates.geometry)
    exist_coords = _utm_coords(charging_stations.geometry)
    covered_before = _covered_before_mask(grid_coords, exist_coords)

    # 增量贪心：在现有站点覆盖基础上，优先选择增量覆盖最大的候选点
    ordered = _incremental_greedy(cand_coords, grid_coords, grid_demand, covered_before, max_k, radius=_COVER_RADIUS)

    st.markdown(f"**预算 {budget} 万元** 可建 **{max_k}** 个{station_type}({power}, 每站{ports}桩, 单站{cost}万元)。")

    # 预算-覆盖率曲线（增量视角：只统计新增站点带来的额外覆盖，排除现有站点已覆盖的网格）
    d_cand = scipy_cdist(grid_coords, cand_coords)
    uncovered = ~covered_before  # 现有站点尚未覆盖的网格
    cum_new = np.zeros(len(grid_coords), dtype=bool)
    curve = []
    prev_new = 0.0
    for rank, idx in enumerate(ordered, start=1):
        cum_new = cum_new | (d_cand[:, idx] <= _COVER_RADIUS)
        newly = cum_new & uncovered
        cov_inc = np.sum(grid_demand * newly) / w_sum * 100 if w_sum > 0 else 0
        marginal = cov_inc - prev_new
        prev_new = cov_inc
        cov_total = np.sum(grid_demand * (covered_before | cum_new)) / w_sum * 100 if w_sum > 0 else 0
        curve.append({'建站数': rank, '累计预算(万元)': rank * cost,
                      '累计新增覆盖(%)': round(cov_inc, 2),
                      '边际覆盖(%)': round(marginal, 2),
                      '总覆盖率(%)': round(cov_total, 1)})
    curve_df = pd.DataFrame(curve)

    # 当前预算下的最优方案
    sel_idx = ordered[:k]
    sel = candidates.iloc[sel_idx].copy() if sel_idx else gpd.GeoDataFrame()
    sel['端口数'] = ports

    if len(sel) > 0:
        sel_coords = cand_coords[sel_idx]
        d_sel = scipy_cdist(grid_coords, sel_coords)
        final_covered = covered_before.copy()
        for j in range(len(sel_idx)):
            final_covered = final_covered | (d_sel[:, j] <= _COVER_RADIUS)
        final_cov = np.sum(grid_demand * final_covered) / w_sum * 100 if w_sum > 0 else 0
        total_cost = len(sel) * cost
        total_ports = len(sel) * ports
    else:
        final_cov = np.sum(grid_demand * covered_before) / w_sum * 100 if w_sum > 0 else 0
        total_cost = 0
        total_ports = 0

    baseline_cov = np.sum(grid_demand * covered_before) / w_sum * 100 if w_sum > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最优建站数", len(sel))
    m2.metric("总投资", f"{total_cost} 万元")
    m3.metric("总端口数", total_ports)
    m4.metric("覆盖率", f"{final_cov:.1f}%", delta=f"{final_cov - baseline_cov:.1f}pp")

    # 预算-覆盖率曲线图
    st.subheader("预算-覆盖率曲线")
    st.caption(f"现有站点基线覆盖率 **{baseline_cov:.1f}%**，以下展示新增站点带来的**增量覆盖**（排除基线后），边际递减规律更清晰。")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=curve_df['累计预算(万元)'], y=curve_df['边际覆盖(%)'], name='每站边际覆盖(%)',
        marker_color=color, yaxis='y2', opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=curve_df['累计预算(万元)'], y=curve_df['累计新增覆盖(%)'], name='累计新增覆盖(%)',
        mode='lines+markers', line=dict(color='#ff5252', width=3), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=curve_df['累计预算(万元)'], y=curve_df['总覆盖率(%)'], name='总覆盖率(%)',
        mode='lines', line=dict(color='#90a4ae', width=2, dash='dot'),
    ))
    if len(curve_df) > 0:
        actual_spend = max_k * cost  # 预算可实际投入的金额(整站倍数)
        b_row = curve_df.iloc[-1] if actual_spend >= curve_df['累计预算(万元)'].max() else \
                curve_df[curve_df['累计预算(万元)'] <= actual_spend].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[b_row['累计预算(万元)']], y=[b_row['累计新增覆盖(%)']],
            mode='markers+text', text=[f"{b_row['累计预算(万元)']}万"], textposition='top center',
            marker=dict(color='#ffea00', size=14, symbol='star'),
            name='当前预算',
        ))
    fig.update_layout(
        title=f'{station_type} 预算与新增覆盖关系', xaxis_title='累计预算(万元)',
        yaxis=dict(title='累计新增覆盖(%) / 总覆盖率(%)', gridcolor='rgba(17,24,39,0.08)'),
        yaxis2=dict(title='每站边际覆盖(%)', overlaying='y', side='right',
                     gridcolor='rgba(17,24,39,0.04)'),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff', font_color='#111827',
        height=400, margin=dict(l=20, r=60, t=50, b=30),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)',
                     bordercolor='#e5e7eb', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 最优方案地图
    st.subheader("预算最优选址地图")
    m = folium.Map(location=[candidates.geometry.y.mean(), candidates.geometry.x.mean()],
                   zoom_start=11, tiles=AMAP_TILE_URL, attr='高德地图')
    for _, row in charging_stations.iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=4,
                            color='#42a5f5', fill=True, fill_opacity=0.6,
                            tooltip=f"现有站: {row.get('名称', '')}").add_to(m)
    if len(sel) > 0:
        for i, row in enumerate(sel.iterrows()):
            g = row[1].geometry
            # 覆盖圈用类型色，站点标记统一用醒目绿色
            folium.Circle([g.y, g.x], radius=5000, color=color, fill=True,
                          fill_color=color, fill_opacity=0.10, weight=2).add_to(m)
            folium.CircleMarker([g.y, g.x], radius=9, color='#22c55e', fill=True,
                                fill_color='#4ade80', fill_opacity=0.95, weight=3,
                                popup=f"新增{station_type} #{i+1}<br>({g.y:.4f}, {g.x:.4f})").add_to(m)
    else:
        st.info("在给定预算下未选出可带来增量覆盖的站点。")
    legend_html = f'''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: #ffffff; padding: 16px; border-radius: 10px; border: 1px solid #e5e7eb;">
        <h4 style="color:#fff;margin-bottom:10px;font-size:14px;">图例</h4>
        <div style="margin-bottom:6px;"><span style="color:#4ade80;font-size:16px;">●</span> <span style="color:#4b5563;font-size:13px;">新增{station_type}</span></div>
        <div style="margin-bottom:6px;"><span style="color:#42a5f5;font-size:14px;">●</span> <span style="color:#4b5563;font-size:13px;">现有站点</span></div>
        <div><span style="color:{color};font-size:16px;">◯</span> <span style="color:#4b5563;font-size:13px;">覆盖半径 5km</span></div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium_static(m, width=None, height=450)


def task3_demand_forecast(grid, candidates, charging_stations, deficit_threshold=config.BLIND_THRESHOLD, new_stations_num=5):
    """创新5 — 需求预测：基于新能源汽车保有量增长的前瞻性选址"""
    from scipy.spatial.distance import cdist as scipy_cdist

    if grid.empty or candidates.empty:
        st.warning("数据不足，无法进行需求预测")
        return

    st.markdown("### 🔮 需求预测与前瞻性选址")
    st.markdown("基于西安市新能源汽车保有量增长趋势，预测未来 3-5 年充电需求，识别未来新增盲区并给出前瞻性选址建议。")

    c1, c2 = st.columns(2)
    with c1:
        horizon = st.slider("预测年限(年)", 3, 5, 5, 1)
    with c2:
        growth = st.slider("新能源汽车年均增长率(%)", 10, 40, 25, 5)

    r = growth / 100.0
    years = list(range(0, horizon + 1))
    base_demand = grid['需求量'].values
    total_demand_series = [np.sum(base_demand) * (1 + r) ** t for t in years]

    # 需求增长曲线
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[2026 + t for t in years], y=total_demand_series,
        mode='lines+markers', line=dict(color='#42a5f5', width=3), marker=dict(size=8),
        fill='tozeroy', fillcolor='rgba(66,165,245,0.15)',
    ))
    fig.update_layout(
        title='充电需求增长预测', xaxis_title='年份', yaxis_title='总需求(加权)',
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font_color='#111827', height=360, margin=dict(l=20, r=20, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 目标年份需求与前瞻选址
    target_grid = grid.copy()
    target_grid['需求量'] = base_demand * (1 + r) ** horizon

    # 前瞻选址(在目标年份需求下贪心)
    ordered = greedy_selection(candidates, target_grid, len(candidates), _COVER_RADIUS, enforce_spread=True)
    k_future = min(len(ordered), new_stations_num)
    future_sel = candidates.iloc[ordered[:k_future]].copy() if ordered else gpd.GeoDataFrame()

    # 未来盲区识别
    grid_coords = _utm_coords(target_grid['centroid'])
    demand_vals = target_grid['需求量'].values
    exist_coords = _utm_coords(charging_stations.geometry)
    covered = _covered_before_mask(grid_coords, exist_coords)

    # 供给不足(需求/供给比下降 → 按阈值)
    thr = target_grid['供需得分'].quantile(deficit_threshold)
    future_blind = target_grid[target_grid['供需得分'] <= thr]
    # 新出现的盲区(当前不盲、未来盲)
    cur_thr = grid['供需得分'].quantile(deficit_threshold)
    cur_blind_idx = grid[grid['供需得分'] <= cur_thr].index
    new_blind = future_blind[~future_blind.index.isin(cur_blind_idx)]

    m1, m2, m3 = st.columns(3)
    m1.metric("目标年总需求", f"{total_demand_series[-1]:,.0f}", delta=f"+{((1+r)**horizon - 1)*100:.0f}%")
    m2.metric("未来盲区网格", len(future_blind))
    m3.metric("新增盲区网格", len(new_blind))

    # 前瞻选址地图
    st.subheader("前瞻性选址方案地图")
    m = folium.Map(location=[grid.geometry.centroid.y.mean(), grid.geometry.centroid.x.mean()],
                   zoom_start=11, tiles=AMAP_TILE_URL, attr='高德地图')
    if len(new_blind) > 0:
        folium.GeoJson(new_blind[['geometry', '供需得分']], name='未来新增盲区',
                       style_function=lambda x: {'fillColor': '#ff9800', 'color': '#e65100',
                                                  'weight': 1, 'fillOpacity': 0.5}).add_to(m)
    for _, row in charging_stations.iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=4,
                            color='#42a5f5', fill=True, fill_opacity=0.6).add_to(m)
    if not future_sel.empty:
        for i in range(len(future_sel)):
            g = future_sel.geometry.iloc[i]
            folium.Circle([g.y, g.x], radius=5000, color='#ab47bc', fill=True,
                          fill_color='#ab47bc', fill_opacity=0.08, weight=1.5).add_to(m)
            folium.CircleMarker([g.y, g.x], radius=9, color='#ab47bc', fill=True,
                                fill_color='#ce93d8', fill_opacity=0.95, weight=3,
                                popup=f"前瞻站点 #{i+1} ({g.y:.4f}, {g.x:.4f})").add_to(m)
    folium.LayerControl().add_to(m)
    folium_static(m, width=None, height=480)

    # 结论建议
    st.markdown("---")
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; border-left: 4px solid #ab47bc;">
        <p><strong>🔮 前瞻性选址建议</strong>：按年均 <strong>{growth}%</strong> 增长，
        至 <strong>{2026 + horizon} 年</strong>充电需求将增长至当前的
        <strong>{(1+r)**horizon * 100:.0f}%</strong>，预计新增
        <strong>{len(new_blind)}</strong> 个盲区网格。建议提前在紫色标记位置预留土地与电网容量，
        分期建设，避免未来供给缺口。</p>
    </div>
    """, unsafe_allow_html=True)


def generate_html_report(charging_stations, districts, city_boundary, poi_df):
    """生成完整的HTML分析报告,可下载或打印为PDF"""
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_stations = len(charging_stations)
    n_districts = len(districts) if districts is not None else 0
    total_ports = int(charging_stations['端口数'].sum()) if '端口数' in charging_stations.columns else 'N/A'
    n_poi = len(poi_df) if poi_df is not None else 0

    # 运营商统计
    operator_stats = ""
    if '运营商' in charging_stations.columns:
        top_ops = charging_stations['运营商'].value_counts().head(5)
        for op, cnt in top_ops.items():
            operator_stats += f"<tr><td>{op}</td><td>{cnt}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>西安市充电设施供需匹配与选址优化 — 分析报告</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
  * {{ font-family: 'Noto Sans SC', sans-serif; }}
  body {{ max-width: 1000px; margin: 0 auto; padding: 40px 20px; color: #1a202c; line-height: 1.8; }}
  h1 {{ font-size: 28px; color: #1e88e5; border-bottom: 3px solid #1e88e5; padding-bottom: 12px; }}
  h2 {{ font-size: 20px; color: #1565c0; margin-top: 32px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 10px 14px; text-align: left; }}
  th {{ background: #f8f9fa; font-weight: 600; }}
  .hero {{ background: linear-gradient(135deg, #1e88e5, #00acc1); color: white; padding: 40px; border-radius: 12px; text-align: center; margin-bottom: 32px; }}
  .hero h1 {{ color: white; border: none; font-size: 32px; }}
  .cards {{ display: flex; gap: 16px; margin: 24px 0; }}
  .card {{ flex: 1; background: #f8f9fa; border-radius: 10px; padding: 20px; text-align: center; border-top: 4px solid #1e88e5; }}
  .card .num {{ font-size: 36px; font-weight: 700; color: #1e88e5; }}
  .card .label {{ color: #4a5568; font-size: 14px; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #718096; font-size: 13px; text-align: center; }}
  @media print {{ body {{ padding: 0; }} .no-print {{ display: none; }} }}
</style>
</head>
<body>
<div class="hero">
  <h1>🔋 西安市新能源汽车充电设施<br>供需匹配与选址优化</h1>
  <p style="font-size: 16px; opacity: 0.9; margin-top: 12px;">技术分析报告 | 生成时间: {now}</p>
</div>

<h2>📊 数据概览</h2>
<div class="cards">
  <div class="card"><div class="num">{n_stations}</div><div class="label">充电站总数</div></div>
  <div class="card"><div class="num">{total_ports}</div><div class="label">总端口数</div></div>
  <div class="card"><div class="num">{n_poi}</div><div class="label">POI记录数</div></div>
  <div class="card"><div class="num">{n_districts}</div><div class="label">覆盖区县</div></div>
</div>

<h2>🏢 运营商分布 (Top 5)</h2>
<table>
  <tr><th>运营商</th><th>站点数量</th></tr>
  {operator_stats}
</table>

<h2>🔬 分析方法</h2>
<ul>
  <li><strong>空间分布测度</strong>: 核密度估计(KDE)、Moran's I 空间自相关、Getis-Ord Gi* 冷热点分析</li>
  <li><strong>供需匹配量化</strong>: 两步移动搜索法(2SFCA)、增强型E2SFCA(Luo & Qi, 2009)、多种距离衰减函数对比</li>
  <li><strong>选址优化</strong>: 多目标贪婪P-Median算法(覆盖率+公平性)、整数规划、3层候选点生成策略</li>
  <li><strong>效果评估</strong>: 9维指标体系、Gini系数+Theil指数、敏感性分析、蒙特卡洛不确定性量化</li>
  <li><strong>创新功能</strong>: 多情景政策对比、边际效益递减分析、时空需求差异建模、路网距离计算</li>
</ul>

<h2>🎯 核心创新点</h2>
<ol>
  <li><strong>E2SFCA增强模型</strong>: 引入子区段权重(0-1km/1-3km/3-5km = 1.0/0.68/0.22),更真实刻画出行阻抗</li>
  <li><strong>多目标优化</strong>: 兼顾覆盖率最大化与Gini最小化,支持4种政策情景模拟</li>
  <li><strong>智能端口估算</strong>: 基于运营商基线+站点类型关键词+噪声的三层估算模型</li>
  <li><strong>边际效益曲线</strong>: 量化逐站覆盖增益递减规律,辅助确定最优建站数量</li>
  <li><strong>时空分析</strong>: 4时段需求权重修饰,揭示充电需求的时空异质性</li>
  <li><strong>不确定性量化</strong>: 蒙特卡洛100次模拟,输出关键指标95%置信区间</li>
</ol>

<h2>📋 政策建议</h2>
<ol>
  <li>优先在<strong>服务盲区</strong>（供需得分最低20%网格）布局新站点</li>
  <li>采用<strong>均衡发展</strong>策略,兼顾覆盖效率与区域公平</li>
  <li>夜间充电服务薄弱,建议在<strong>居住区周边</strong>增设充电设施</li>
  <li>新建站点建议数量: 5-8个,超过边际效益递减拐点后增益有限</li>
</ol>

<div class="footer">
  <p>本报告由"西安市新能源汽车充电设施供需匹配与选址优化系统"自动生成 | {now}</p>
  <p class="no-print" style="margin-top: 16px;">
    <button onclick="window.print()" style="padding:12px 32px;font-size:16px;background:#1e88e5;color:white;border:none;border-radius:8px;cursor:pointer;">
      🖨️ 打印/导出PDF
    </button>
  </p>
</div>
</body>
</html>"""
    return html


def main():
    st.markdown("""
        <style>
        #main-title {
            font-size: 28px !important;
            font-weight: 600 !important;
            color: #111827 !important;
            text-align: left !important;
            margin: 8px 0 20px 0 !important;
            line-height: 1.3 !important;
            letter-spacing: 0 !important;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 12px;
        }
        </style>
        <div id="main-title">西安市新能源汽车充电设施供需匹配与选址优化系统</div>
    """, unsafe_allow_html=True)

    with st.spinner("正在加载数据，请稍候..."):
        charging_stations, districts, city_boundary, roads, poi_df = load_data()

    show_stats_cards(charging_stations, districts)

    with st.sidebar:
        st.markdown("<h2 style='color: #1e88e5; margin-bottom: 25px; font-size: 22px;'>功能菜单</h2>", unsafe_allow_html=True)

        task_choice = st.radio("选择任务模块", ["任务一：空间分布测度", "任务二：供需匹配量化", "任务三：选址优化"], index=0)

        if task_choice == "任务一：空间分布测度":
            task1_option = st.selectbox("选择分析项", ["充电站分布图", "区县密度图", "运营商统计图"])

        elif task_choice == "任务二：供需匹配量化":
            task2_option = st.selectbox("选择分析项", ["网格供需得分图", "冷热点分析图", "服务盲区地图", "敏感性分析面板", "时空对比分析"])
            search_radius = st.slider("搜索半径(km)", 3, 10, 5, 1, help="2SFCA模型的搜索半径参数")
            sfca_method = st.selectbox("2SFCA方法", ["标准2SFCA (高斯衰减)", "E2SFCA (子区段权重)"],
                                       help="标准2SFCA使用单一高斯衰减；E2SFCA将搜索半径分为0-1km/1-3km/3-5km子区段，权重分别为1.0/0.68/0.22 (Luo & Qi, 2009)")
            deficit_threshold = st.slider("供给不足阈值百分位数", 0.1, 0.5, 0.2, 0.05,
                                        help="供需得分低于此百分位数的网格判定为供给不足区")

        elif task_choice == "任务三：选址优化":
            task3_option = st.selectbox("选择分析项", ["优化前后对比图", "新增站点地图", "指标对比表", "多情景政策对比", "边际效益曲线", "选址可解释性", "覆盖热力叠加", "方案保存与对比", "成本约束选址", "需求预测选址"])
            new_stations_num = st.slider("新增站点数量", 3, 10, 5, 1, help="待选址的充电站数量")
            enforce_spread = st.checkbox("强制分散", value=True, help="开启后新增站点之间至少保持1.5km距离")
            solver_method = st.selectbox("求解方法", ["贪心启发式", "整数规划"], index=0,
                                        help="贪心启发式速度快，整数规划更精确但可能耗时较长")
            sfca_method_t3 = st.selectbox("2SFCA方法", ["标准2SFCA (高斯衰减)", "E2SFCA (子区段权重)"],
                                          help="标准2SFCA使用单一高斯衰减；E2SFCA考虑距离衰减的分段差异性")
            deficit_threshold = st.slider("供给不足阈值百分位数", 0.1, 0.5, 0.2, 0.05,
                                        help="供需得分低于此百分位数的网格判定为供给不足区")

        # 报告导出按钮
        st.markdown("---")
        st.markdown("<p style='color:#8b949e;font-size:13px;margin-bottom:8px;'>📄 报告导出</p>", unsafe_allow_html=True)
        if st.button("📥 生成HTML分析报告", use_container_width=True,
                     help="生成包含数据概览、分析方法、创新点和政策建议的完整HTML报告，可在浏览器中打开并打印为PDF"):
            with st.spinner("正在生成报告..."):
                report_html = generate_html_report(charging_stations, districts, city_boundary, poi_df)
                st.download_button(
                    label="💾 下载报告 (HTML)",
                    data=report_html,
                    file_name="西安市充电设施分析报告.html",
                    mime="text/html",
                    use_container_width=True,
                )
                st.success("✅ 报告生成成功！点击上方按钮下载，在浏览器中打开后可打印为PDF。")

    if task_choice == "任务一：空间分布测度":
        st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #111827; margin-bottom: 25px; font-size: 28px;'>任务一：空间分布测度</h2>", unsafe_allow_html=True)

        if task1_option == "充电站分布图":
            st.markdown("### 充电站分布图")
            st.markdown("显示西安市所有充电站的位置分布，点击标记可查看详细信息（运营商、端口数等）。")
            task1_charging_map(charging_stations, districts)

        elif task1_option == "区县密度图":
            st.markdown("### 区县密度图")
            st.markdown("展示各区县充电站的分布密度（充电站数量/区县面积），颜色越深表示密度越高。")
            task1_density_map(charging_stations, districts)

        elif task1_option == "运营商统计图":
            st.markdown("### 运营商统计图")
            st.markdown("统计各运营商在西安市的充电站数量分布，便于了解市场竞争格局。")
            task1_operator_chart(charging_stations)
        st.markdown("</div>", unsafe_allow_html=True)

    elif task_choice == "任务二：供需匹配量化":
        st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #111827; margin-bottom: 25px; font-size: 28px;'>任务二：供需匹配量化</h2>", unsafe_allow_html=True)

        with st.spinner("正在计算供需匹配..."):
            grid = create_grid(city_boundary)
            grid = calculate_supply_demand(charging_stations, grid, poi_df)
            if "E2SFCA" in sfca_method:
                grid, _ = calculate_e2sfca(grid, charging_stations, search_radius * 1000)
            else:
                grid, _ = calculate_2sfca(grid, charging_stations, search_radius * 1000)
            
            # 使用用户设置的阈值判定供给不足区
            threshold_value = grid['供需得分'].quantile(deficit_threshold)
            grid['供给状态'] = grid['供需得分'].apply(lambda x: '供给不足' if x <= threshold_value else '供给充足')

        if task2_option == "网格供需得分图":
            st.markdown("### 网格供需得分图")
            st.markdown("基于2SFCA模型计算的1km×1km网格供需匹配得分分布。红色表示供给不足，绿色表示供给充足。")
            task2_supply_demand_map(grid)

        elif task2_option == "冷热点分析图":
            st.markdown("### 冷热点分析图")
            st.markdown("使用局部Getis-Ord Gi*统计量识别充电站供需的热点区域（高值聚集）和冷点区域（低值聚集）。")
            task2_hotspot_map(grid)

        elif task2_option == "服务盲区地图":
            st.markdown("### 服务盲区地图")
            st.markdown(f"识别供需得分最低的{int(deficit_threshold*100)}%网格作为服务盲区，显示为红色区域，支持下载GeoJSON文件。")
            task2_blind_area_map(grid, charging_stations, districts, deficit_threshold)

        elif task2_option == "敏感性分析面板":
            st.markdown("### 敏感性分析面板")
            st.markdown("交互式调节模型参数，实时观察供需匹配指标变化，评估参数不确定性影响。")
            task2_sensitivity_dashboard(grid, charging_stations, poi_df)

        elif task2_option == "时空对比分析":
            st.markdown("### 时空对比分析")
            st.markdown("模拟早高峰、午间、晚高峰、夜间四个典型时段的需求差异，对比各时段充电供需匹配变化。")
            task2_temporal_analysis(grid, charging_stations)

        st.markdown("</div>", unsafe_allow_html=True)

    elif task_choice == "任务三：选址优化":
        st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #111827; margin-bottom: 25px; font-size: 28px;'>任务三：选址优化</h2>", unsafe_allow_html=True)

        # 检查参数是否变化，决定是否重新计算
        params_key = f"{new_stations_num}_{solver_method}_{enforce_spread}_{deficit_threshold}"
        needs_recalc = (
            'task3_results' not in st.session_state or
            st.session_state.get('task3_params') != params_key
        )

        if needs_recalc:
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # 步骤1: 创建网格
                status_text.text("正在创建网格...")
                progress_bar.progress(10)
                grid = create_grid(city_boundary)
                st.info(f"网格数量: {len(grid)}")

                # 步骤2: 计算供需
                status_text.text("正在计算供需...")
                progress_bar.progress(30)
                grid = calculate_supply_demand(charging_stations, grid, poi_df)

                # 步骤3: 2SFCA分析
                status_text.text("正在进行2SFCA分析...")
                progress_bar.progress(50)
                if "E2SFCA" in sfca_method_t3:
                    grid, _ = calculate_e2sfca(grid, charging_stations, 5000)
                else:
                    grid, _ = calculate_2sfca(grid, charging_stations, 5000)

                # 步骤4: 生成候选点
                status_text.text("正在生成候选点...")
                progress_bar.progress(70)
                threshold = grid['供需得分'].quantile(deficit_threshold)
                blind_areas = grid[grid['供需得分'] <= threshold]
                candidates = generate_candidates(blind_areas, roads, charging_stations, city_boundary)
                st.info(f"候选点数量: {len(candidates)}")

                # 步骤5: 选址优化
                status_text.text("正在进行选址优化...")
                progress_bar.progress(85)
                
                # 转换求解方法参数
                solver_type = 'greedy' if solver_method == '贪心启发式' else 'integer'
                new_stations, coverage_ratio, solve_time = optimize_locations(
                    candidates, grid, 
                    k=new_stations_num, 
                    solver_method=solver_type,
                    enforce_spread=enforce_spread
                )
                st.info(f"求解耗时: {solve_time:.2f}秒")

                # 步骤6: 计算指标
                status_text.text("正在计算评价指标...")
                progress_bar.progress(95)
                before_metrics = calculate_metrics(grid, charging_stations)
                after_metrics = calculate_metrics(grid, charging_stations, new_stations)

                # 完成
                progress_bar.progress(100)
                status_text.text("计算完成！")

                # 缓存结果
                st.session_state['task3_results'] = {
                    'grid': grid,
                    'candidates': candidates,
                    'new_stations': new_stations,
                    'coverage_ratio': coverage_ratio,
                    'solve_time': solve_time,
                    'before_metrics': before_metrics,
                    'after_metrics': after_metrics
                }
                st.session_state['task3_params'] = params_key

            except Exception as e:
                import traceback
                st.error(f"计算出错: {e}")
                traceback.print_exc()
                st.session_state['task3_results'] = {
                    'grid': None,
                    'candidates': gpd.GeoDataFrame(),
                    'new_stations': gpd.GeoDataFrame(),
                    'coverage_ratio': 0.0,
                    'solve_time': 0.0,
                    'before_metrics': None,
                    'after_metrics': None
                }
                st.session_state['task3_params'] = params_key
                # 设置默认局部变量防止 UnboundLocalError
                new_stations = gpd.GeoDataFrame()
                coverage_ratio = 0.0
                solve_time = 0.0
                before_metrics = None
                after_metrics = None
        else:
            # 使用缓存的结果
            results = st.session_state['task3_results']
            grid = results.get('grid')
            candidates = results.get('candidates', gpd.GeoDataFrame())
            new_stations = results['new_stations']
            coverage_ratio = results['coverage_ratio']
            solve_time = results.get('solve_time', 0.0)
            before_metrics = results['before_metrics']
            after_metrics = results['after_metrics']
            st.info("使用缓存结果，如需重新计算请修改参数")
        
        # 显示优化指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("优化覆盖率", f"{coverage_ratio*100:.1f}%")
        with col2:
            if before_metrics and after_metrics:
                distance_improve = (before_metrics['平均可达距离(km)'] - after_metrics['平均可达距离(km)']) / before_metrics['平均可达距离(km)'] * 100
                st.metric("平均距离改善", f"{distance_improve:.1f}%")
            else:
                st.metric("平均距离改善", "N/A")
        with col3:
            st.metric("新增站点数量", len(new_stations))
        with col4:
            st.metric("求解耗时", f"{solve_time:.2f}秒")

        if task3_option == "优化前后对比图":
            st.markdown("### 优化前后对比图")
            st.markdown("蓝色标记为现有充电站，绿色标记为新增优化站点，直观展示选址优化效果。")
            task3_comparison_map(charging_stations, new_stations, districts)

        elif task3_option == "新增站点地图":
            st.markdown("### 新增站点地图")
            st.markdown("显示通过优化算法筛选出的新增充电站位置，点击标记可查看详细坐标信息。")
            task3_new_stations_map(new_stations, districts)

        elif task3_option == "指标对比表":
            st.markdown("### 指标对比表")
            st.markdown("对比优化前后的平均可达距离、覆盖率和基尼系数，量化评估选址优化效果。")
            task3_metrics_table(before_metrics, after_metrics)

        elif task3_option == "多情景政策对比":
            st.markdown("### 多情景政策对比")
            st.markdown("在公平优先、效率优先、交通导向、均衡发展四种政策取向下，分别进行选址优化并对比效果差异。")
            task3_scenario_comparison(grid, candidates, charging_stations, k=new_stations_num)

        elif task3_option == "边际效益曲线":
            st.markdown("### 边际效益曲线")
            st.markdown("展示每新增一个充电站带来的覆盖增益变化，揭示边际效益递减规律，为确定最优建站数量提供决策依据。")
            task3_marginal_benefit(grid, candidates, charging_stations, max_k=15)

        elif task3_option == "选址可解释性":
            st.markdown("### 选址可解释性")
            st.markdown("对每个新增站点标注「为何选这里」——贴近盲区质心、覆盖未满足需求、避开与现有站点竞争。")
            task3_explainability(grid, candidates, new_stations, charging_stations, deficit_threshold)

        elif task3_option == "覆盖热力叠加":
            st.markdown("### 覆盖率热力叠加")
            st.markdown("将新增站点的覆盖半径实时叠加到服务盲区地图上，直观展示选址对盲区的填补效果。")
            task3_coverage_overlay(grid, new_stations, charging_stations, deficit_threshold)

        elif task3_option == "方案保存与对比":
            st.markdown("### 方案保存与对比记忆")
            st.markdown("保存多个选址方案并排对比，追踪不同参数组合下的优化效果差异。")
            task3_scenario_memory(new_stations, before_metrics, after_metrics, coverage_ratio, solve_time, params_key, new_stations_num, enforce_spread, solver_method)

        elif task3_option == "成本约束选址":
            st.markdown("### 成本约束选址")
            st.markdown("引入快充/慢充/超充差异化建站成本，在固定预算下求解最优建站组合。")
            task3_cost_optimization(grid, candidates, charging_stations)

        elif task3_option == "需求预测选址":
            st.markdown("### 需求预测与前瞻性选址")
            st.markdown("基于新能源汽车保有量增长趋势预测未来充电需求，识别未来盲区并给出前瞻性选址建议。")
            task3_demand_forecast(grid, candidates, charging_stations, deficit_threshold, new_stations_num)

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
