"""
主执行脚本
按顺序执行所有分析模块
"""
import sys
import importlib
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
config = importlib.import_module('config')
from logger import logger

def run_module(module_name):
    """运行指定模块，返回 (成功/失败, 耗时秒)"""
    t0 = time.time()
    logger.info(f"{'='*50}")
    logger.info(f"模块: {module_name}")
    try:
        module = importlib.import_module(module_name)
        module.main()
        elapsed = time.time() - t0
        logger.info(f"模块 {module_name}: 成功 ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"模块 {module_name}: 失败 ({elapsed:.1f}s) - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数:按顺序执行所有模块"""
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("西安市新能源汽车充电设施供需匹配与选址优化")
    logger.info("=" * 60)

    modules = [
        '01_preprocess_poi',
        '02_spatial_matching',
        '03_regional_analysis',
        '04_supply_demand_model',
        '05_location_optimization',
        '06_evaluation'
    ]

    success_count = 0
    for module in modules:
        if run_module(module):
            success_count += 1

    total_time = time.time() - t0
    logger.info(f"执行完成: {success_count}/{len(modules)} 个模块成功 (总耗时 {total_time:.1f}s)")
    logger.info(f"输出目录: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
