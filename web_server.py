"""旅行规划助手 Web 服务（FastAPI 实现）。

启动后访问 http://127.0.0.1:8899 使用前端表单，
提交后调用 planner 生成旅行规划。自动交互式文档见 /docs。

用法：
    E:\\python\\python.exe web_server.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# 自动把 src 目录加入模块搜索路径，无需手动设置 PYTHONPATH
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

WEB_DIR = Path(__file__).resolve().parent / "web"

# 单例 planner，避免每次请求重复加载工具
_planner = None
_planner_lock = asyncio.Lock()

PLAN_TIMEOUT_S = 180


# ============================================================
# planner 生命周期
# ============================================================

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
    return await asyncio.wait_for(planner.ainvoke(prompt), timeout=PLAN_TIMEOUT_S)


# ============================================================
# 业务辅助函数（从原 http.server 版本原样迁移，逻辑不变）
# ============================================================

def _search_pois(city: str) -> list[dict]:
    """调用高德 POI 搜索，返回目的地景点坐标列表（供前端地图可视化）。"""
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

    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    if start == -1:
        return None
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
    """把各 Agent 的结构化 JSON 字符串解析成前端可用的数据。"""
    result: dict = {
        "attractions": [],
        "weather": None,
        "hotels": [],
        "trains": [],
        "itinerary": None,
        "budget": None,
        "hotel": None,
    }
    attr = _extract_json(structured.get("attractions", "")) or {}
    result["attractions"] = attr.get("attractions", [])
    result["weather"] = _extract_json(structured.get("weather", ""))
    hotel = _extract_json(structured.get("hotels", "")) or {}
    result["hotels"] = hotel.get("hotels", [])
    train = _extract_json(structured.get("trains", "")) or {}
    result["trains"] = train.get("trains", [])
    itin = _extract_json(structured.get("itinerary", "")) or {}
    itinerary = itin.get("itinerary")
    if isinstance(itinerary, dict):
        itinerary = _match_itinerary_coords(itinerary, result["attractions"])
        if city:
            itinerary = _fill_missing_coords(itinerary, city)
        result["itinerary"] = itinerary
    result["budget"] = _extract_json(structured.get("budget", ""))
    # 选定酒店：从预算结果里提取最终入住酒店，匹配 hotels 拿坐标，供行程展示与地图标记
    result["hotel"] = _resolve_selected_hotel(result.get("budget"), result.get("hotels"))
    return result


def _resolve_selected_hotel(budget: dict | None, hotels: list) -> dict | None:
    """从预算结果里提取选定酒店名，匹配酒店列表补全坐标/价格。

    预算 Agent 输出 hotel（酒店全名）+ hotel_price_per_night（每晚单价），
    这里按名称匹配 hotels 列表，拿到 location/image 等，返回一个完整酒店对象；
    匹配不到时退化为仅含名称的占位对象。
    """
    if not budget or not isinstance(budget, dict):
        return None
    hotel_name = (budget.get("hotel") or "").strip()
    if not hotel_name:
        # 兼容：预算里没写 hotel 字段时，从住宿明细里尝试提取
        hotel_name = ""
    if not hotel_name:
        return None

    def _norm(s: str) -> str:
        for w in ("酒店", "宾馆", "客栈", "民宿", " "):
            s = s.replace(w, "")
        return s

    target = _norm(hotel_name)
    matched = None
    for h in hotels:
        if not isinstance(h, dict):
            continue
        name = (h.get("name") or "").strip()
        if _norm(name) == target or target in _norm(name) or _norm(name) in target:
            matched = h
            break
    if matched:
        result = dict(matched)
    else:
        result = {"name": hotel_name}
    result["price_per_night"] = budget.get("hotel_price_per_night", result.get("price"))
    return result


def _fill_missing_coords(itinerary: dict, city: str) -> dict:
    """对行程中仍缺坐标的景点，用高德 POI 搜索补坐标（兜底）。"""
    from trip_agent.tools import amap_search_poi

    missing: list[tuple[str, str]] = []
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
    """为行程中的每个景点匹配坐标/图片/介绍（从 attractions 里按名称匹配）。"""
    info_map: dict[str, dict] = {}
    for a in attractions:
        name = (a.get("name") or "").strip()
        if not name:
            continue
        info_map[name] = a

    def _normalize(s: str) -> str:
        for w in ("风景名胜区", "景区", "风景区", "公园", "国家", "博物馆", "步行街", "历史街区", "园区", "（", "）", "(", ")", " "):
            s = s.replace(w, "")
        return s

    def _match(name: str) -> dict | None:
        if not name:
            return None
        if name in info_map:
            return info_map[name]
        nkey = _normalize(name)
        for k, v in info_map.items():
            if _normalize(k) == nkey:
                return v
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


# ============================================================
# 请求 / 响应模型
# ============================================================

class PlanRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="旅行规划需求描述")


# ============================================================
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时预热 planner（可选：失败不阻塞启动）
    try:
        await _get_planner()
        print("planner 已预热完成")
    except Exception as e:
        print(f"planner 预热失败（将在首个请求时重试）：{e}")
    yield


app = FastAPI(title="多智能体旅行规划助手", version="1.0.0", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    """把 HTTPException 统一成前端认识的 {error} 结构（原 http.server 版即如此）。"""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """参数校验失败也返回 {error}，并把 detail 压平成可读文案。"""
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        msg = first.get("msg", "参数校验失败")
        detail = f"{loc}: {msg}" if loc else msg
    else:
        detail = "参数校验失败"
    return JSONResponse(status_code=422, content={"error": detail})


@app.get("/api/photo")
async def api_photo(name: str = ""):
    """GET /api/photo?name=景点名 → 返回真实图片 URL。"""
    from trip_agent.tools import search_attraction_photo

    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="缺少 name 参数")
    try:
        url = search_attraction_photo(name)
    except Exception:
        url = None
    return {"success": url is not None, "message": "ok", "data": {"name": name, "photo_url": url}}


@app.post("/api/plan")
async def api_plan(req: PlanRequest):
    """POST /api/plan {prompt} → 生成旅行规划（结构化 + 正文）。"""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt 不能为空")

    try:
        result = await _plan(prompt)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="规划超时（超过180秒），请简化需求后重试")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"规划失败：{e}")

    destination_city = result.get("destination", "")
    pois = _search_pois(destination_city) if destination_city else []

    from trip_agent.config import settings

    structured_raw = result.get("structured") or {}
    structured = _parse_structured(structured_raw, city=destination_city)

    return {
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


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/{path:path}")
async def static_files(path: str):
    """兜底静态资源（截图等）。"""
    candidate = WEB_DIR / path
    if candidate.is_file() and WEB_DIR in candidate.resolve().parents:
        return FileResponse(candidate)
    raise HTTPException(status_code=404, detail="not found")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8899, log_level="info")


if __name__ == "__main__":
    main()
