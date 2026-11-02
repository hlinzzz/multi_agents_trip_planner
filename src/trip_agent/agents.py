"""专业 Agent 定义。

每个 Agent 是一个 LangGraph 子图（create_react_agent），
拥有独立的工具集和系统提示词，由 PlannerAgent 作为节点调用。

Agent 清单：
- attraction_agent    景点搜索
- weather_agent       天气查询
- hotel_agent         酒店推荐
- traffic_agent       交通 / 车次
- itinerary_agent     行程规划
- budget_agent        预算规划（纯推理，无外部工具）
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from .state import AgentState

# ============================================================
# 系统提示词
# ============================================================

ATTRACTION_PROMPT = """你是景点搜索专家。根据用户目的地和偏好，搜索并推荐合适的景点。
关键要求：优先用 fliggy_search_poi 搜索景点（返回含 mainPic 图片、description 介绍、poiLevel 等级、ticketInfo 门票），
再用 amap_search_poi 补坐标/评分（飞猪景点若缺坐标）。
每个景点务必包含：名称、坐标 location、图片 image（飞猪 mainPic）、介绍 intro、门票 ticket。
从结果中筛选出最值得去的景点，考虑评分、类型多样性和地理位置，推荐 6-10 个。
输出 JSON：{"attractions": [{"name":..., "location":..., "rating":..., "type":..., "ticket":..., "image":..., "intro":...}]}
其中 intro 为 1-2 句景点介绍（可从飞猪 description 提炼），image 为图片 URL（飞猪 mainPic）。"""

WEATHER_PROMPT = """你是天气查询专家。查询目的地城市的天气。
使用 amap_weather 获取天气（含每天天气/气温/风向风力）。
输出 JSON：{"weather": {"days": [{"date":..., "weather":..., "temp":..., "wind":...}]}, "advice": "..."}
保留每天的天气现象（如晴/多云/小雨）、温度区间、风向风力，供前端绘制天气图例。"""

HOTEL_PROMPT = """你是酒店推荐专家。根据目的地、日期和预算推荐酒店。
使用 fliggy_search_hotel 搜索酒店，按预算给出不同档次的选项：
- 预算宽松（每晚 >400 元）→ 传 hotel_stars="4,5" 查高档/四五星酒店
- 预算中等（每晚 200-400 元）→ 传 hotel_stars="3,4" 查舒适型
- 预算紧张（每晚 <200 元）→ 传 hotel_stars="1,2,3" 查经济型
同时优先推荐位置便利（靠近景点/车站）的选项。
输出 JSON：{"hotels": [{"name":..., "price":..., "star":..., "location":..., "image":..., "nearby":...}]}
其中 image 为酒店图片 URL（飞猪 mainPic 字段），nearby 为附近地标。
注意：price 为「每晚单价」，输出时须说明按人数需要几间房、总价如何计算。"""

TRAFFIC_PROMPT = """你是交通规划专家。查询出发城市到目的地（及城市间）的车次。
优先使用 fliggy_search_train 查询车票（数据稳定、含票价），默认已返回高铁/动车二等座。
若用户预算充足，可通过 seat_class_name 参数查一等座/商务座对比。
一次性完成查询并输出 JSON：{"trains": [{"train_no":..., "depart":..., "arrive":..., "price":...}]}
其中 price 必须是「单张票价」，并额外给出按人数的总价（人数×单张价），字段如 price_per_person 和 price_total。"""

ITINERARY_PROMPT = """你是行程规划专家。根据已知的景点列表、天气和天数，规划每日行程。
要求：
1. 按天组织，每天输出一个景点列表（数组），每个景点是一个独立"块"，包含：
   - name: 景点名
   - time: 建议游览时段（如 09:00-11:30）
   - duration: 游览时长（小时）
   - intro: 景点介绍（1-2 句，从前面景点结果提炼）
   - transport: 从前一个景点到该景点的交通安排（步行/地铁/打车 + 大致耗时），当天第一个景点填"从酒店出发"
   - food: 该景点附近的餐饮推荐（可用 amap_search_food 查附近美食，给出 1-2 家餐厅名 + 人均）
2. 合理安排每天游览顺序（同区域景点放同一天），结合天气避开雨天户外景点。
3. 尽量基于已有信息直接规划，最多调用 2 次工具补充必要信息，不要反复调用工具。
输出 JSON：{"itinerary": {"day1": [{"name":..., "time":..., "duration":..., "intro":..., "transport":..., "food":...}], "day2": [...]}}"""

BUDGET_PROMPT = """你是预算规划专家。根据总预算和各分项真实价格，做预算分配与核算。
关键规则（务必遵守）：
1. 交通、门票、餐饮均按「人数」计算总价：票价×人数、门票×人数、餐饮人均×人数×天数
2. 住宿按「房间数」计算：通常 2 人 1 间，间数 = 向上取整(人数/2)，住宿总价 = 每晚价×间数×晚数
3. 每个分项用真实价格回填，明确写出「单价×人数（×天数）= 总价」的计算式
4. 判断是否超支，若超支给出降级建议
输出 JSON：{"breakdown": {"交通": {"预分配": 元, "实际": 元, "明细": "计算式"}, "住宿": {...}, "门票": {...}, "餐饮": {...}, "机动": {...}}, "总预算": 元, "总实际": 元, "over_budget": bool, "suggestion": "..."}
注意：breakdown 里每个分项的值必须是「数字」而不是嵌套对象，明细放计算文字。"""


# ============================================================
# Agent 构建工厂
# ============================================================

def build_agent(
    llm: BaseChatModel,
    tools: list,
    prompt: str,
) -> object:
    """构建一个 ReAct 子 Agent。"""
    return create_react_agent(llm, tools, prompt=prompt)


def build_all_agents(
    llm: BaseChatModel,
    tools_by_role: dict[str, list],
) -> dict[str, object]:
    """构建全部专业 Agent，返回 {角色名: agent子图}。

    tools_by_role 结构：
    {
        "attraction": [amap_search_poi],
        "weather":    [amap_weather],
        "hotel":      [fliggy_search_hotel],
        "traffic":    [fliggy_search_train],
    }
    """
    return {
        "attraction": build_agent(llm, tools_by_role.get("attraction", []), ATTRACTION_PROMPT),
        "weather": build_agent(llm, tools_by_role.get("weather", []), WEATHER_PROMPT),
        "hotel": build_agent(llm, tools_by_role.get("hotel", []), HOTEL_PROMPT),
        "traffic": build_agent(llm, tools_by_role.get("traffic", []), TRAFFIC_PROMPT),
        "itinerary": build_agent(llm, tools_by_role.get("itinerary", []), ITINERARY_PROMPT),
        "budget": build_agent(llm, tools_by_role.get("budget", []), BUDGET_PROMPT),
    }


# ============================================================
# 工具分配（按角色归类）
# ============================================================

def assign_tools(all_tools: list) -> dict[str, list]:
    """把统一工具列表按角色归类。"""
    role_map: dict[str, list] = {
        "attraction": [],
        "weather": [],
        "hotel": [],
        "traffic": [],
        "itinerary": [],
        "budget": [],
    }
    for t in all_tools:
        name = getattr(t, "name", "")
        if name == "amap_search_poi":
            role_map["attraction"].append(t)
        elif name == "amap_weather":
            role_map["weather"].append(t)
        elif name == "amap_route_planning":
            role_map["traffic"].append(t)
        elif name == "amap_search_food":
            role_map["itinerary"].append(t)
        elif name == "fliggy_search_hotel":
            role_map["hotel"].append(t)
        elif name == "fliggy_search_train":
            role_map["traffic"].append(t)
        elif name == "fliggy_search_poi":
            role_map["attraction"].append(t)
    return role_map


# ============================================================
# 子图节点封装（供 PlannerAgent 主图调用）
# ============================================================

async def _run_subagent(agent, state: AgentState, input_text: str, recursion_limit: int = 8, include_history: bool = False) -> str:
    """异步运行一个子 Agent，返回其最终文本输出。

    用 ainvoke 保持异步执行；设置 recursion_limit 防止 ReAct agent 无限循环调用工具。

    Args:
        include_history: 是否把主图已有的 messages 一起传给子 agent
            （itinerary/budget 这类需要消费前面结果的 agent 应设为 True）。
    """
    from langchain_core.messages import HumanMessage

    msgs = []
    if include_history:
        # 带上前面 agent 的结果作为上下文
        prev = state.get("messages", [])
        for m in prev[-6:]:  # 只带最近 6 条，控制上下文长度
            if hasattr(m, "content") and m.content:
                msgs.append(m)
    msgs.append(HumanMessage(content=input_text))

    result = await agent.ainvoke(
        {"messages": msgs},
        config={"recursion_limit": recursion_limit},
    )
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        # 兼容 message 对象或 tuple
        return str(getattr(last, "content", last))
    return ""


def attraction_node(agent):
    async def _fn(state: AgentState) -> dict:
        text = await _run_subagent(
            agent, state,
            f"目的地：{state.get('destination')}，偏好：{state.get('preferences', '')}。请搜索并推荐景点。",
        )
        return {"messages": [("assistant", f"[景点] {text}")]}
    return _fn


def weather_node(agent):
    async def _fn(state: AgentState) -> dict:
        text = await _run_subagent(
            agent, state,
            f"目的地：{state.get('destination')}，天数：{state.get('days')}。请查询天气。",
        )
        return {"messages": [("assistant", f"[天气] {text}")]}
    return _fn


def hotel_node(agent):
    async def _fn(state: AgentState) -> dict:
        text = await _run_subagent(
            agent, state,
            f"目的地：{state.get('destination')}，预算：{state.get('budget')}，人数：{state.get('people', 1)}。请推荐酒店。",
        )
        return {"messages": [("assistant", f"[酒店] {text}")]}
    return _fn


def traffic_node(agent):
    async def _fn(state: AgentState) -> dict:
        text = await _run_subagent(
            agent, state,
            f"出发：{state.get('start_city')}，到达：{state.get('destination')}，人数：{state.get('people', 1)}。请查询车次。",
            recursion_limit=12,
        )
        return {"messages": [("assistant", f"[交通] {text}")]}
    return _fn


def itinerary_node(agent):
    async def _fn(state: AgentState) -> dict:
        text = await _run_subagent(
            agent, state,
            f"目的地：{state.get('destination')}，天数：{state.get('days')}，人数：{state.get('people', 1)}。"
            f"请根据前面收集的景点、天气、酒店、交通信息规划每日行程，"
            f"每天按景点分块，含介绍、景点间交通安排和附近餐饮推荐。",
            include_history=True,
            recursion_limit=16,
        )
        return {"messages": [("assistant", f"[行程] {text}")]}
    return _fn


def budget_agent_node(agent):
    async def _fn(state: AgentState) -> dict:
        text = await _run_subagent(
            agent, state,
            f"总预算：{state.get('budget')}，人数：{state.get('people', 1)}。请根据前面收集的酒店、交通、门票价格做预算核算（按人数算总价）。",
            include_history=True,
        )
        return {"messages": [("assistant", f"[预算] {text}")]}
    return _fn
