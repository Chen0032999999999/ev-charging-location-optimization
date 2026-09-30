"""
日志模块
统一的日志记录，同时输出到控制台和 run.log 文件
"""
import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "run.log"

# 重置 root logger，避免重复 handler
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("yaosu")
