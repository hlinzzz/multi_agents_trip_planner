"""工具层：接入并统一管理所有外部工具。

三类工具来源：
1. 12306 —— 通过 langchain-mcp-adapters 的 MultiServerMCPClient 接入 MCP 服务
2. 高德地图 —— 用 @tool 封装 Web 服务 HTTP API（POI 搜索 / 天气 / 路径规划）
3. 飞猪 —— 用 @tool 封装 flyai 开放平台 HTTP API

提供统一入口：
- build_tools(): 启动所有 MCP 客户端并返回全部工具（真实模式）
- build_mock_tools(): 返回 mock 工具（无凭证时本地跑通用）
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any

import requests
from langchain_core.tools import tool

from .config import settings

# ============================================================
# 高德地图工具（HTTP 直连，无需 MCP 进程）
# ============================================================

_AMAP_BASE = "https://restapi.amap.com/v3"


_AMAP_SEMAPHORE = threading.Semaphore(2)  # 限制高德并发，避免触发 QPS 限制


def _amap_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """调用高德 Web 服务接口（带并发限流 + 指数退避重试）。"""
    import time

    params = {**params, "key": settings.AMAP_API_KEY}
    last_err: Exception | None = None
    for attempt in range(4):
        with _AMAP_SEMAPHORE:
            try:
                resp = requests.get(f"{_AMAP_BASE}{path}", params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "1":
                    return data
                info = data.get("info", "")
                # 并发超限 → 退避重试
                if "EXCEEDED" in info or "CUQPS" in info or "LIMIT" in info:
                    last_err = RuntimeError(f"高德接口限流: {info}")
                else:
                    raise RuntimeError(f"高德接口错误: {info}")
            except requests.RequestException as e:
                last_err = e
        time.sleep(1.5 * (attempt + 1))  # 1.5s / 3s / 4.5s
    raise RuntimeError(f"高德接口调用失败: {last_err}")


@tool
def amap_search_poi(keywords: str, city: str = "", citylimit: str = "false") -> str:
    """搜索景点 / 地点 POI。

    Args:
        keywords: 搜索关键词，如 "故宫"、"西湖"、"美食"
        city: 城市名或城市编码，如 "北京"。留空则全国搜索。
        citylimit: 是否仅限指定城市，取值 "true" / "false"

    Returns:
        JSON 字符串，包含 POI 列表（名称、地址、坐标、评分等）。
    """
    data = _amap_get(
        "/place/text",
        {
            "keywords": keywords,
            "city": city,
            "citylimit": citylimit,
            "extensions": "all",
            "offset": 20,
            "page": 1,
        },
    )
    pois = data.get("pois", [])
    return json.dumps(
        {
            "count": int(data.get("count", 0)),
            "pois": [
                {
                    "name": p.get("name"),
                    "address": p.get("address"),
                    "location": p.get("location"),
                    "type": p.get("type"),
                    "rating": p.get("biz_ext", {}).get("rating"),
                    "cost": p.get("biz_ext", {}).get("cost"),
                }
                for p in pois
            ],
        },
        ensure_ascii=False,
    )


@tool
def amap_weather(city: str) -> str:
    """查询城市实时天气与未来预报。

    Args:
        city: 城市名称或城市编码（adcode），如 "北京" 或 "110000"

    Returns:
        JSON 字符串，包含实时天气和未来 4 天预报。
    """
    # 若传入城市名，先转 adcode
    code = city
    if not city.isdigit():
        geo = _amap_get(
            "/geocode/geo", {"address": city, "city": city}
        )
        geos = geo.get("geocodes", [])
        if not geos:
            return json.dumps({"error": f"未找到城市 {city}"}, ensure_ascii=False)
        code = geos[0]["adcode"]
    data = _amap_get(
        "/weather/weatherInfo", {"city": code, "extensions": "all"}
    )
    forecasts = data.get("forecasts", [])
    if not forecasts:
        return json.dumps({"error": "天气数据为空"}, ensure_ascii=False)
    return json.dumps(forecasts[0], ensure_ascii=False)


@tool
def amap_route_planning(
    origin: str, destination: str, strategy: str = "0"
) -> str:
    """查询两点间的驾车 / 公交路径规划（用于估算城市间交通）。

    Args:
        origin: 起点坐标 "经度,纬度"，如 "116.397428,39.90923"
        destination: 终点坐标 "经度,纬度"
        strategy: 驾车策略，默认 "0"（速度优先）

    Returns:
        JSON 字符串，包含路径距离、耗时、步骤。
    """
    data = _amap_get(
        "/direction/driving",
        {"origin": origin, "destination": destination, "strategy": strategy},
    )
    route = data.get("route", {})
    paths = route.get("paths", [])
    if not paths:
        return json.dumps({"error": "未规划出路径"}, ensure_ascii=False)
    p = paths[0]
    return json.dumps(
        {
            "distance_m": p.get("distance"),
            "duration_s": p.get("duration"),
            "steps_count": len(p.get("steps", [])),
        },
        ensure_ascii=False,
    )


@tool
def amap_search_food(
    city: str,
    keywords: str = "美食",
    location: str = "",
    radius: str = "3000",
) -> str:
    """搜索餐饮 / 美食推荐（高德 POI 周边美食）。

    Args:
        city: 城市名，如 "杭州"
        keywords: 美食关键词，如 "杭帮菜"、"小吃"、"火锅"，默认 "美食"
        location: 中心点坐标 "经度,纬度"，用于按位置搜附近美食，可为空
        radius: 搜索半径（米），默认 3000

    Returns:
        JSON 字符串，含餐厅列表（名称、地址、坐标、评分、人均消费、类型）。
    """
    params: dict[str, Any] = {
        "keywords": keywords,
        "city": city,
        "citylimit": "true",
        "types": "050000",  # 高德餐饮类型
        "extensions": "all",
        "offset": 20,
        "page": 1,
    }
    if location:
        params["location"] = location
        params["radius"] = radius
    data = _amap_get("/place/around" if location else "/place/text", params)
    pois = data.get("pois", [])
    return json.dumps(
        {
            "count": int(data.get("count", 0)),
            "restaurants": [
                {
                    "name": p.get("name"),
                    "address": p.get("address"),
                    "location": p.get("location"),
                    "type": p.get("type"),
                    "rating": p.get("biz_ext", {}).get("rating"),
                    "cost": p.get("biz_ext", {}).get("cost"),
                    "tel": p.get("tel"),
                }
                for p in pois
            ],
        },
        ensure_ascii=False,
    )


# ============================================================
# 飞猪工具（通过 flyai-cli 调用 Fliggy MCP 服务）
# ============================================================

def _find_flyai() -> list[str]:
    """定位 flyai 的调用方式，返回命令行前缀（node + bundle 脚本）。

    在 Windows 上 npm 全局安装的 flyai 是 .cmd / 无扩展名 shell 脚本，
    subprocess 直接执行会报 WinError 193。这里改为直接用 node 执行
    flyai 的 bundle 入口，跨平台稳定且不依赖 shell。
    """
    import shutil

    # 1. 环境变量指定 node + bundle 时直接用
    explicit = os.environ.get("FLYAI_BIN")
    if explicit:
        return [explicit]

    # 2. 探测 node 可执行文件
    node = os.environ.get("FLYAI_NODE") or shutil.which("node")
    if not node:
        node = os.path.expanduser(
            "~/.workbuddy/binaries/node/versions/22.22.2-6/node"
        )

    # 3. 定位 flyai-cli 的 bundle 入口
    bundle = os.environ.get("FLYAI_BUNDLE")
    if not bundle:
        bundle_candidates = [
            os.path.expanduser(
                "~/.workbuddy/binaries/node/versions/22.22.2-6/"
                "node_modules/@fly-ai/flyai-cli/dist/flyai-bundle.cjs"
            ),
            os.path.expanduser(
                "~/AppData/Roaming/npm/node_modules/@fly-ai/flyai-cli/"
                "dist/flyai-bundle.cjs"
            ),
        ]
        for c in bundle_candidates:
            if os.path.exists(c):
                bundle = c
                break

    if node and bundle and os.path.exists(bundle):
        return [node, bundle]

    # 兜底：直接返回 flyai 命令（非 Windows 或已配置 shell）
    return ["flyai"]


def _run_flyai(args: list[str]) -> str:
    """调用 flyai-cli，返回 stdout 的 JSON 字符串。"""
    import subprocess

    cmd = _find_flyai() + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
        )
    except FileNotFoundError:
        return json.dumps(
            {"error": "未找到 flyai 命令，请先执行 npm i -g @fly-ai/flyai-cli"},
            ensure_ascii=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "飞猪接口超时"}, ensure_ascii=False)

    stdout = (result.stdout or "").strip()
    if stdout:
        return stdout
    # 错误走 stderr
    return json.dumps(
        {"error": (result.stderr or "").strip()[:500] or "未知错误"},
        ensure_ascii=False,
    )


@tool
def fliggy_search_hotel(
    dest_name: str,
    check_in_date: str = "",
    check_out_date: str = "",
    poi_name: str = "",
    max_price: float = 0.0,
    hotel_stars: str = "",
    sort: str = "rate_desc",
) -> str:
    """搜索酒店（飞猪，实时真实房源与价格）。

    Args:
        dest_name: 目的地城市名，如 "杭州"、"北京"
        check_in_date: 入住日期 YYYY-MM-DD，可为空
        check_out_date: 离店日期 YYYY-MM-DD，可为空
        poi_name: 附近景点名，如 "西湖"，用于按位置筛选，可为空
        max_price: 每晚最高价格（元），0 表示不限
        hotel_stars: 星级，逗号分隔如 "4,5"，可为空
        sort: 排序，distance_asc/rate_desc/price_asc/price_desc/no_rank

    Returns:
        JSON 字符串，含 hotels 精简列表（名称、价格、星级、坐标、图片URL、地址、附近POI）。
    """
    args = ["search-hotel", "--dest-name", dest_name]
    if check_in_date:
        args += ["--check-in-date", check_in_date]
    if check_out_date:
        args += ["--check-out-date", check_out_date]
    if poi_name:
        args += ["--poi-name", poi_name]
    if max_price and max_price > 0:
        args += ["--max-price", str(int(max_price))]
    if hotel_stars:
        args += ["--hotel-stars", hotel_stars]
    if sort:
        args += ["--sort", sort]
    raw = _run_flyai(args)
    # 精简返回字段，保留前端展示需要的图片/坐标等
    try:
        data = json.loads(raw)
        items = data.get("data", {}).get("itemList", [])
        hotels = [
            {
                "name": h.get("name"),
                "price": h.get("price"),
                "star": h.get("star"),
                "brand": h.get("brandName"),
                "address": h.get("address"),
                "location": (
                    f"{h.get('longitude')},{h.get('latitude')}"
                    if h.get("longitude") and h.get("latitude") else None
                ),
                "image": h.get("mainPic"),
                "nearby": h.get("interestsPoi"),
                "url": h.get("detailUrl"),
            }
            for h in items
        ]
        return json.dumps({"count": len(hotels), "hotels": hotels}, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return raw


@tool
def fliggy_search_train(
    origin: str,
    destination: str = "",
    dep_date: str = "",
    max_price: float = 0.0,
    sort_type: str = "8",
    seat_class_name: str = "second class",
) -> str:
    """搜索火车票（飞猪，可作 12306 的价格对比补充）。

    Args:
        origin: 出发城市或车站，如 "北京"
        destination: 到达城市或车站，如 "上海"
        dep_date: 出发日期 YYYY-MM-DD，可为空
        max_price: 最高票价（元），0 表示不限
        sort_type: 排序，默认 "8"=直达优先。可选 1价格高到低/2推荐/3价格低到高/4历时短到长/5历时长到短/6出发早/7出发晚/8直达优先
        seat_class_name: 坐席类型，逗号分隔，如 "second class,first class"。
            默认 "second class"（二等座，高铁/动车）。可选 first class/business class/hard sleeper/soft sleeper

    Returns:
        JSON 字符串，含车次、时刻、票价、预订链接。
    """
    args = ["search-train", "--origin", origin]
    if destination:
        args += ["--destination", destination]
    if dep_date:
        args += ["--dep-date", dep_date]
    if max_price and max_price > 0:
        args += ["--max-price", str(int(max_price))]
    if seat_class_name:
        args += ["--seat-class-name", seat_class_name]
    if sort_type:
        args += ["--sort-type", str(sort_type)]
    return _run_flyai(args)


@tool
def fliggy_search_poi(
    city_name: str,
    keyword: str = "",
    category: str = "",
    poi_level: str = "",
) -> str:
    """搜索景点/门票（飞猪，可作高德景点搜索的补充，含门票价格、介绍、图片）。

    Args:
        city_name: 城市名，如 "北京"
        keyword: 景点关键词，如 "故宫"
        category: 景点分类，如 "自然风光"/"历史古迹"/"主题乐园"，可为空
        poi_level: 景点等级 1-5，可为空

    Returns:
        JSON 字符串，含 spots 精简列表（名称、等级、分类、门票、介绍、坐标、图片URL）。
    """
    args = ["search-poi", "--city-name", city_name]
    if keyword:
        args += ["--keyword", keyword]
    if category:
        args += ["--category", category]
    if poi_level:
        args += ["--poi-level", poi_level]
    raw = _run_flyai(args)
    # 精简返回字段，保留前端展示需要的介绍/图片/坐标
    try:
        data = json.loads(raw)
        items = data.get("data", {}).get("itemList", [])
        spots = [
            {
                "name": s.get("name"),
                "level": s.get("poiLevel"),
                "category": s.get("category"),
                "ticket": s.get("ticketInfo"),
                "free": s.get("freePoiStatus"),
                "rank": s.get("listRank"),
                "description": s.get("description"),
                "address": s.get("address"),
                "location": (
                    f"{s.get('longitude')},{s.get('latitude')}"
                    if s.get("longitude") and s.get("latitude") else None
                ),
                "image": s.get("mainPic"),
                "url": s.get("jumpUrl"),
            }
            for s in items
        ]
        return json.dumps({"count": len(spots), "spots": spots}, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return raw


# ============================================================
# 12306 MCP 工具接入
# ============================================================

def _build_12306_mcp_client():
    """构造 12306 MCP 客户端（stdio 模式）。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # 优先用 E:/python/python.exe（3.12，已装 mcp 1.30.0）
    python_exe = os.environ.get(
        "TRIP_PYTHON", "E:/python/python.exe"
    )
    return MultiServerMCPClient(
        {
            "12306": {
                "transport": "stdio",
                "command": python_exe,
                "args": ["-m", "mcp_12306.server"],
                "cwd": settings.MCP_12306_DIR,
                "env": {
                    "PYTHONPATH": settings.MCP_12306_SRC,
                    # 透传登录 Cookie 相关环境变量
                    "RAIL_EXPIRATION": os.getenv("RAIL_EXPIRATION", ""),
                    "RAIL_DEVICEID": os.getenv("RAIL_DEVICEID", ""),
                    "RAIL_USERID": os.getenv("RAIL_USERID", ""),
                    "BIGipServerotn": os.getenv("BIGipServerotn", ""),
                },
            }
        }
    )


# ============================================================
# 景点图片搜索（Unsplash 优先，Wikimedia 兜底）
# ============================================================

def search_attraction_photo(name: str) -> str | None:
    """根据景点名搜索真实图片 URL。

    优先用 Unsplash（需配置 UNSPLASH_ACCESS_KEY，图片质量高、专门拍地标）；
    无 key 时回退到 Wikimedia Commons（无需鉴权，免费）。

    Args:
        name: 景点名称，如 "故宫"、"颐和园"

    Returns:
        图片 URL，找不到返回 None。
    """
    if settings.UNSPLASH_ACCESS_KEY:
        url = _unsplash_photo(name)
        if url:
            return url
    return _wikimedia_photo(name)


def _unsplash_photo(name: str) -> str | None:
    """Unsplash 搜索景点图片（带鉴权）。"""
    try:
        resp = requests.get(
            "https://api.unsplash.com/search/photos",
            params={
                "query": f"{name} landmark",
                "per_page": 1,
                "client_id": settings.UNSPLASH_ACCESS_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0].get("urls", {}).get("regular")
    except Exception:
        pass
    return None


def _wikimedia_photo(name: str) -> str | None:
    """Wikimedia Commons 搜索景点图片（无需鉴权）。英文关键词优先，质量更稳定。"""
    # 常见景点中英文映射，英文搜索命中更精准（避免中文古籍扫描图）
    en_map = {
        "故宫": "Forbidden City Beijing", "天安门": "Tiananmen Square",
        "颐和园": "Summer Palace Beijing", "长城": "Great Wall of China",
        "八达岭": "Great Wall Badaling", "天坛": "Temple of Heaven Beijing",
        "西湖": "West Lake Hangzhou", "外滩": "The Bund Shanghai",
        "兵马俑": "Terracotta Army", "黄山": "Mount Huangshan",
        "张家界": "Zhangjiajie", "桂林": "Guilin landscape",
        "九寨沟": "Jiuzhaigou", "泰山": "Mount Tai",
        "布达拉宫": "Potala Palace", "莫高窟": "Mogao Caves",
        "鼓浪屿": "Gulangyu", "拙政园": "Humble Administrator Garden",
        "夫子庙": "Confucius Temple Nanjing", "秦淮河": "Qinhuai River",
        "黄鹤楼": "Yellow Crane Tower", "岳麓山": "Yuelu Mountain",
        "豫园": "Yu Garden", "东方明珠": "Oriental Pearl Tower",
        "恭王府": "Prince Gong Mansion", "雍和宫": "Yonghe Temple Beijing",
        "什刹海": "Shichahai Beijing", "南锣鼓巷": "Nanluoguxiang",
        "烟袋斜街": "Yandaixiejie Beijing", "簋街": "Guijie Beijing",
        "前门": "Qianmen Beijing", "大栅栏": "Dashilan Beijing",
        "天安门广场": "Tiananmen Square",
    }
    # 构造查询顺序：英文映射优先，中文名兜底
    queries = []
    for key, en in en_map.items():
        if key in name:
            queries.append(en)
    queries.append(name)

    for q in queries:
        try:
            resp = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": q,
                    "gsrnamespace": "6",
                    "gsrlimit": "8",
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime",
                    "iiurlwidth": "800",
                    "format": "json",
                },
                headers={"User-Agent": "TripPlanner/1.0"},
                timeout=10,
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            # 优先横向照片（jpg/jpeg，宽度>=500，横向图）
            landscape = None
            fallback = None
            for p in pages.values():
                info = (p.get("imageinfo") or [{}])[0]
                thumb = info.get("thumburl")
                if not thumb:
                    continue
                mime = info.get("mime", "")
                if mime and "jpeg" not in mime and "png" not in mime:
                    continue  # 跳过 djvu/pdf/svg 等非照片
                w = info.get("thumbwidth", 0)
                h = info.get("thumbheight", 0)
                if w >= 500 and w >= h:
                    return thumb  # 找到横向大图，直接返回
                if fallback is None:
                    fallback = thumb
                if landscape is None and w >= 500:
                    landscape = thumb
            if landscape:
                return landscape
            if fallback:
                return fallback
        except Exception:
            continue
    return None


# ============================================================
# 统一工具组装入口
# ============================================================

async def load_tools(include_12306: bool = False) -> list:
    """启动 MCP 客户端并返回全部真实工具。

    Args:
        include_12306: 是否加载 12306 MCP 工具。默认 False，因为 12306 无登录态
            时 query_tickets 可能长时间挂起（风控），拖垮整个编排。需要精确
            时刻表/余票时再开启，并先在 .env 配置 RAIL_* Cookie。

    返回的工具列表可直接作为 Agent 的工具集。调用方负责在结束后关闭客户端。
    """
    amap_tools = [amap_search_poi, amap_weather, amap_route_planning, amap_search_food]
    fliggy_tools = [fliggy_search_hotel, fliggy_search_train, fliggy_search_poi]
    if not include_12306:
        return amap_tools + fliggy_tools
    client = _build_12306_mcp_client()
    mcp_tools = await client.get_tools()
    return amap_tools + fliggy_tools + mcp_tools


def build_mock_tools() -> list:
    """返回 mock 工具（无凭证 / 离线时本地跑通用）。"""
    from langchain_core.tools import tool as _tool

    @_tool
    def mock_query_tickets(
        from_station: str, to_station: str, date: str
    ) -> str:
        """Mock：查询车次余票。"""
        return json.dumps(
            {
                "count": 2,
                "trains": [
                    {"train_no": "G101", "from": from_station, "to": to_station,
                     "depart_time": "08:00", "arrive_time": "13:00", "price": 553.5},
                    {"train_no": "G103", "from": from_station, "to": to_station,
                     "depart_time": "10:00", "arrive_time": "15:00", "price": 553.5},
                ],
            },
            ensure_ascii=False,
        )

    @_tool
    def mock_weather(city: str) -> str:
        """Mock：查询天气。"""
        return json.dumps(
            {"city": city, "weather": "晴", "temp": "22", "forecast": "晴朗"},
            ensure_ascii=False,
        )

    return [amap_search_poi, mock_weather, mock_query_tickets, fliggy_search_hotel]


async def verify_12306_connection() -> dict[str, Any]:
    """验证 12306 MCP 连通性，返回工具清单。"""
    client = _build_12306_mcp_client()
    tools = await client.get_tools()
    return {
        "ok": True,
        "tool_names": [t.name for t in tools],
    }
