"""旅行规划助手 Web 服务（标准库实现，零额外依赖）。

启动后访问 http://127.0.0.1:8000 使用前端表单，
提交后调用 planner 生成旅行规划。

用法：
    E:\\python\\python.exe web_server.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# 自动把 src 目录加入模块搜索路径，无需手动设置 PYTHONPATH
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

WEB_DIR = Path(__file__).resolve().parent / "web"

# 单例 planner，避免每次请求重复加载工具
_planner = None
_planner_lock = asyncio.Lock()


async def _get_planner():
    global _planner
    if _planner is None:
        async with _planner_lock:
            if _planner is None:
                from langchain_openai import ChatOpenAI

                from trip_agent.config import settings
                from trip_agent.planner import TripPlanner
                from trip_agent.tools import load_tools

                llm = ChatOpenAI(
                    model=settings.LLM_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL,
                    temperature=0,
                )
                planner_llm = ChatOpenAI(
                    model=settings.PLANNER_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL,
                    temperature=0,
                )
                tools = await load_tools()
                _planner = TripPlanner(llm, planner_llm, tools)
    return _planner


async def _plan(prompt: str) -> dict:
    """在同一事件循环内完成 planner 加载与执行，带整体超时保护。"""
    planner = await _get_planner()
    return await asyncio.wait_for(planner.ainvoke(prompt), timeout=180)


def _search_pois(city: str) -> list[dict]:
    """调用高德 POI 搜索，返回目的地景点坐标列表（供前端地图可视化）。"""
    from trip_agent.config import settings
    from trip_agent.tools import amap_search_poi

    if not city:
        return []
    try:
        raw = amap_search_poi.invoke({"keywords": "景点", "city": city})
        data = json.loads(raw)
        pois = data.get("pois", [])[:8]
        return [
            {
                "name": p.get("name"),
                "location": p.get("location"),
                "rating": p.get("rating"),
            }
            for p in pois
            if p.get("location")
        ]
    except Exception:
        return []


def _extract_json(text: str) -> dict | None:
    """从一段文本中提取第一个 JSON 对象，容错处理。"""
    if not text:
        return None
    import re

    # 去掉可能的 markdown 代码块包裹
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # 尝试直接解析
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # 提取最外层 {...}
    start = cleaned.find("{")
    if start == -1:
        return None
    # 用括号配对找到匹配的结束位置
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except Exception:
                    return None
    return None


def _parse_structured(structured: dict, city: str = "") -> dict:
    """把各 Agent 的结构化 JSON 字符串解析成前端可用的数据。

    Args:
        city: 目的地城市，用于对缺坐标的行程景点做高德 POI 兜底搜索。
    """
    result: dict = {
        "attractions": [],
        "weather": None,
        "hotels": [],
        "trains": [],
        "itinerary": None,
        "budget": None,
    }
    # 景点
    attr = _extract_json(structured.get("attractions", "")) or {}
    result["attractions"] = attr.get("attractions", [])
    # 天气
    result["weather"] = _extract_json(structured.get("weather", ""))
    # 酒店
    hotel = _extract_json(structured.get("hotels", "")) or {}
    result["hotels"] = hotel.get("hotels", [])
    # 交通
    train = _extract_json(structured.get("trains", "")) or {}
    result["trains"] = train.get("trains", [])
    # 行程
    itin = _extract_json(structured.get("itinerary", "")) or {}
    itinerary = itin.get("itinerary")
    if isinstance(itinerary, dict):
        itinerary = _match_itinerary_coords(itinerary, result["attractions"])
        if city:
            itinerary = _fill_missing_coords(itinerary, city)
        result["itinerary"] = itinerary
    # 预算
    result["budget"] = _extract_json(structured.get("budget", ""))
    return result


def _fill_missing_coords(itinerary: dict, city: str) -> dict:
    """对行程中仍缺坐标的景点，用高德 POI 搜索补坐标（兜底）。"""
    from trip_agent.tools import amap_search_poi

    # 收集所有缺坐标的景点名
    missing: list[tuple[str, str]] = []  # (day, name)
    for day, spots in itinerary.items():
        if not isinstance(spots, list):
            continue
        for s in spots:
            if isinstance(s, dict) and not s.get("location") and s.get("name"):
                missing.append((day, s["name"]))

    if not missing:
        return itinerary

    for day, name in missing:
        try:
            raw = amap_search_poi.invoke({"keywords": name, "city": city, "citylimit": "true"})
            data = json.loads(raw)
            pois = data.get("pois", [])
            if pois and pois[0].get("location"):
                loc = pois[0]["location"]
                for s in itinerary.get(day, []):
                    if isinstance(s, dict) and s.get("name") == name:
                        s["location"] = loc
        except Exception:
            continue
    return itinerary


def _match_itinerary_coords(itinerary: dict, attractions: list) -> dict:
    """为行程中的每个景点匹配坐标/图片/介绍（从 attractions 里按名称匹配）。

    供前端地图按天分色打点、景点贴照片。返回带 location/image/intro 的 itinerary。
    """
    # 构建名称 → 完整信息的映射
    info_map: dict[str, dict] = {}
    for a in attractions:
        name = (a.get("name") or "").strip()
        if not name:
            continue
        info_map[name] = a

    def _normalize(s: str) -> str:
        """去掉常见干扰词，便于模糊匹配。"""
        for w in ("风景名胜区", "景区", "风景区", "公园", "国家", "博物馆", "步行街", "历史街区", "园区", "（", "）", "(", ")", " "):
            s = s.replace(w, "")
        return s

    def _match(name: str) -> dict | None:
        if not name:
            return None
        # 精确匹配
        if name in info_map:
            return info_map[name]
        # 归一化后匹配
        nkey = _normalize(name)
        for k, v in info_map.items():
            if _normalize(k) == nkey:
                return v
        # 包含关系匹配（取最短的 key 更精准）
        candidates = []
        for k, v in info_map.items():
            nk = _normalize(k)
            if nk and (nk in nkey or nkey in nk):
                candidates.append((len(nk), v))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return None

    enriched: dict[str, Any] = {}
    for day, spots in itinerary.items():
        if not isinstance(spots, list):
            enriched[day] = spots
            continue
        new_spots = []
        for s in spots:
            if isinstance(s, dict):
                matched = _match(s.get("name", ""))
                if matched:
                    if not s.get("location"):
                        loc = matched.get("location") or matched.get("lnglat")
                        if loc:
                            s["location"] = loc
                    if not s.get("image"):
                        img = matched.get("image")
                        if img:
                            s["image"] = img
                    if not s.get("intro"):
                        intro = matched.get("intro") or matched.get("description")
                        if intro:
                            s["intro"] = intro
            new_spots.append(s)
        enriched[day] = new_spots
    return enriched


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path):
        data = path.read_bytes()
        self.send_response(200)
        if path.suffix == ".html":
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif path.suffix == ".js":
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
        elif path.suffix == ".css":
            self.send_header("Content-Type", "text/css; charset=utf-8")
        else:
            self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html")
        elif path == "/api/photo":
            self._handle_photo(parsed)
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_photo(self, parsed):
        """GET /api/photo?name=景点名 → 返回真实图片 URL。"""
        from urllib.parse import parse_qs

        from trip_agent.tools import search_attraction_photo

        qs = parse_qs(parsed.query)
        name = (qs.get("name") or [""])[0].strip()
        if not name:
            self._send_json({"success": False, "message": "缺少 name 参数"}, 400)
            return
        try:
            url = search_attraction_photo(name)
        except Exception:
            url = None
        self._send_json(
            {"success": url is not None, "message": "ok", "data": {"name": name, "photo_url": url}}
        )

    def do_POST(self):
        if urlparse(self.path).path != "/api/plan":
            self._send_json({"error": "not found"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json({"error": "请求体需为 JSON"}, 400)
            return

        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            self._send_json({"error": "prompt 不能为空"}, 400)
            return

        try:
            result = asyncio.run(_plan(prompt))
        except asyncio.TimeoutError:
            self._send_json({"error": "规划超时（超过180秒），请简化需求后重试"}, 504)
            return
        except Exception as e:
            self._send_json({"error": f"规划失败：{e}"}, 500)
            return

        destination_city = result.get("destination", "")
        # 额外抓取景点坐标，供前端地图可视化（作为分天标记的坐标补充）
        pois = _search_pois(destination_city) if destination_city else []

        from trip_agent.config import settings

        # 解析各 Agent 结构化结果
        structured_raw = result.get("structured") or {}
        structured = _parse_structured(structured_raw, city=destination_city)

        self._send_json(
            {
                "answer": result.get("final_answer", ""),
                "over_budget": result.get("over_budget"),
                "destination": destination_city,
                "start_city": result.get("start_city", ""),
                "days": result.get("days"),
                "people": result.get("people", 1),
                "budget": result.get("budget"),
                "pois": pois,
                "structured": structured,
                "amap_js_key": settings.AMAP_JS_KEY,
            }
        )

    def log_message(self, *args):
        pass


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8899), Handler)
    print("旅行规划助手已启动：http://127.0.0.1:8899")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
