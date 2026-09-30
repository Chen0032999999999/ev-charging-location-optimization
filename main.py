"""
主入口文件
可以运行完整分析管道，或启动交互式 Web 应用
"""
import sys
from pathlib import Path

# 将 src 目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("=" * 60)
    print("西安市新能源汽车充电设施供需匹配与选址优化系统")
    print("=" * 60)
    print()
    print("请选择运行方式：")
    print("  1. 运行完整分析管道:  python src/run_all.py")
    print("  2. 启动 Web 应用:      streamlit run app.py")
    print("  3. 双击 YAOSU启动器.exe 一键启动")
