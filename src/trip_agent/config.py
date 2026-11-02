"""全局配置加载。

从 .env 读取环境变量，提供统一的配置访问入口。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（config.py 位于 src/trip_agent/ 下，往上两级到项目根）
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 加载 .env（若存在）
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """集中管理所有配置项。"""

    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    PLANNER_MODEL: str = os.getenv("PLANNER_MODEL", "gpt-4o")

    # 高德
    AMAP_API_KEY: str = os.getenv("AMAP_API_KEY", "")
    AMAP_JS_KEY: str = os.getenv("AMAP_JS_KEY", "")

    # Unsplash（景点真实图片，可选；留空则回退到 Wikimedia 免费图源）
    UNSPLASH_ACCESS_KEY: str = os.getenv("UNSPLASH_ACCESS_KEY", "")

    # 飞猪（通过 flyai-cli 调用）
    FLYAI_API_KEY: str = os.getenv("FLYAI_API_KEY", "")


settings = Settings()
