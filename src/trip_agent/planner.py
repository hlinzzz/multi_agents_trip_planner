"""PlannerAgent —— 主编排图（LangGraph StateGraph）。

职责：
1. 解析用户输入，填充目的地/天数/预算/偏好
2. 并行调用各专业 Agent（景点/天气/酒店/交通）
3. 串行：行程规划 -> 预算规划
4. 预算超支时条件路由回退（限制最大重试次数）
5. 汇总生成最终答复

支持两种运行模式：
- real：真实 LLM + 真实工具（需凭证）
- mock：mock LLM + mock 工具（本地验证编排逻辑）
"""
from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import (
    assign_tools,
    attraction_node,
    budget_agent_node,
    build_all_agents,
    hotel_node,
    itinerary_node,
    traffic_node,
    weather_node,
)
from .state import AgentState

MAX_RETRY = 2  # 预算回退最大重试次数


def _parse_over_budget(text: str) -> bool:
    """从 budget agent 输出中判断是否超支。

    优先解析 JSON 里的 over_budget 字段；否则回退到关键词检测。
    """
    if not text:
        return False
    # 尝试提取 JSON 片段
    import re

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            ov = data.get("over_budget")
            if isinstance(ov, bool):
                return ov
        except (json.JSONDecodeError, TypeError):
            pass
    # 回退：关键词检测
    if "超支" in text and "未超" not in text:
        return True
    if "over_budget" in text and re.search(r"over_budget[\"']?\s*[:=]\s*[\"']?true", text, re.IGNORECASE):
        return True
    return False


class TripPlanner:
    """多智能体旅行规划助手总入口。"""

    def __init__(self, llm: BaseChatModel, planner_llm: BaseChatModel, tools: list):
        self.llm = llm
        self.planner_llm = planner_llm
        self.tools = tools

        # 按角色归类工具
        role_tools = assign_tools(tools)
        # 构建各专业 Agent
        self.agents = build_all_agents(llm, role_tools)
        # 构建主图
        self.graph = self._build_graph()

    # ----------------------------------------------------------
    # 主图节点
    # ----------------------------------------------------------

    def _parse_input(self, state: AgentState) -> dict:
        """用 planner_llm 解析用户输入，填充结构化字段。"""
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            user_text = str(getattr(last, "content", last))
        else:
            user_text = ""
        parse_prompt = (
            "从用户输入中提取旅行需求，输出严格 JSON：\n"
            '{"destination": str, "start_city": str, "days": int, '
            '"people": int, "budget": float 或 null, "preferences": str}\n'
            "缺失的字段用空字符串/0 填充；人数缺失时默认 1；"
            "预算缺失时 budget 填 null（不要填 0）。\n"
            f"用户输入：{user_text}"
        )
        resp = self.planner_llm.invoke(parse_prompt)
        try:
            data = json.loads(resp.content)
        except (json.JSONDecodeError, TypeError):
            data = {"destination": "", "start_city": "", "days": 0,
                    "people": 1, "budget": None, "preferences": ""}
        budget_raw = data.get("budget")
        try:
            budget = float(budget_raw) if budget_raw not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            budget = None
        return {
            "destination": data.get("destination", ""),
            "start_city": data.get("start_city", ""),
            "days": int(data.get("days", 0) or 0),
            "people": int(data.get("people", 0) or 1),
            "budget": budget,
            "preferences": data.get("preferences", ""),
            "retry_count": 0,
        }

    async def _attraction(self, state: AgentState) -> dict:
        return await attraction_node(self.agents["attraction"])(state)

    async def _weather(self, state: AgentState) -> dict:
        return await weather_node(self.agents["weather"])(state)

    async def _hotel(self, state: AgentState) -> dict:
        return await hotel_node(self.agents["hotel"])(state)

    async def _traffic(self, state: AgentState) -> dict:
        return await traffic_node(self.agents["traffic"])(state)

    async def _itinerary(self, state: AgentState) -> dict:
        return await itinerary_node(self.agents["itinerary"])(state)

    async def _budget(self, state: AgentState) -> dict:
        result = await budget_agent_node(self.agents["budget"])(state)
        # 从 budget agent 输出里提取 over_budget 标记（优先解析 JSON 字段）
        msgs = result.get("messages", [])
        text = str(getattr(msgs[-1], "content", msgs[-1])) if msgs else ""
        over = _parse_over_budget(text)
        return {**result, "over_budget": over}

    async def _summarize(self, state: AgentState) -> dict:
        """汇总所有中间结果，生成最终答复（结构化 JSON + 正文）。

        返回 final_answer（纯文本/HTML 友好的 markdown 正文），同时把各 agent 的
        结构化结果原样沉淀到 state，供后端解析后交给前端渲染。
        """
        msgs = state.get("messages", [])
        # 提取各 agent 的 JSON 结果（带 [角色] 前缀）
        sections: dict[str, Any] = {}
        for m in msgs:
            content = str(getattr(m, "content", m))
            for key in ("景点", "天气", "酒店", "交通", "行程", "预算"):
                tag = f"[{key}]"
                if content.startswith(tag):
                    sections[key] = content[len(tag):].strip()
                    break
        summary_prompt = (
            "根据以下各 Agent 的结果，整合成一份完整的旅行规划正文（Markdown），包含：\n"
            "1. 行程安排（按天，每天列出景点、交通、餐饮）\n"
            "2. 交通方案 3. 酒店推荐 4. 预算清单 5. 天气提醒\n"
            "要求：语言流畅、适合阅读，不要输出 JSON。\n"
            f"各 Agent 结果：\n{msgs}"
        )
        resp = await self.planner_llm.ainvoke(summary_prompt)
        return {
            "final_answer": resp.content,
            # 结构化结果透传，供后端解析
            "structured": {
                "attractions": sections.get("景点", ""),
                "weather": sections.get("天气", ""),
                "hotels": sections.get("酒店", ""),
                "trains": sections.get("交通", ""),
                "itinerary": sections.get("行程", ""),
                "budget": sections.get("预算", ""),
            },
        }

    async def _inc_retry(self, state: AgentState) -> dict:
        """回退计数：每次超支回退时递增 retry_count。"""
        return {"retry_count": state.get("retry_count", 0) + 1}

    # ----------------------------------------------------------
    # 路由函数
    # ----------------------------------------------------------

    def _route_after_parallel(self, state: AgentState) -> Literal["itinerary"]:
        return "itinerary"

    def _route_after_budget(self, state: AgentState) -> Literal["inc_retry", "summarize"]:
        """超支且未达重试上限 -> 回退（先递增计数）；否则 -> summarize。"""
        if state.get("over_budget") and state.get("retry_count", 0) < MAX_RETRY:
            return "inc_retry"
        return "summarize"

    # ----------------------------------------------------------
    # 构图
    # ----------------------------------------------------------

    def _build_graph(self) -> CompiledStateGraph:
        g = StateGraph(AgentState)

        g.add_node("parse_input", self._parse_input)
        g.add_node("attraction", self._attraction)
        g.add_node("weather", self._weather)
        g.add_node("hotel", self._hotel)
        g.add_node("traffic", self._traffic)
        g.add_node("itinerary", self._itinerary)
        g.add_node("budget", self._budget)
        g.add_node("inc_retry", self._inc_retry)
        g.add_node("summarize", self._summarize)

        g.add_edge(START, "parse_input")

        # 并行分支：景点/天气/酒店/交通 互不依赖
        g.add_edge("parse_input", "attraction")
        g.add_edge("parse_input", "weather")
        g.add_edge("parse_input", "hotel")
        g.add_edge("parse_input", "traffic")

        # 并行结果汇聚到 itinerary（LangGraph 会在四个节点都完成后才继续）
        g.add_edge("attraction", "itinerary")
        g.add_edge("weather", "itinerary")
        g.add_edge("hotel", "itinerary")
        g.add_edge("traffic", "itinerary")

        # 行程 -> 预算
        g.add_edge("itinerary", "budget")

        # 预算：条件路由（超支回退 / 汇总）
        g.add_conditional_edges(
            "budget",
            self._route_after_budget,
            {"inc_retry": "inc_retry", "summarize": "summarize"},
        )

        # 回退：递增计数后重新推荐酒店
        g.add_edge("inc_retry", "hotel")

        g.add_edge("summarize", END)

        return g.compile()

    # ----------------------------------------------------------
    # 对外调用
    # ----------------------------------------------------------

    def invoke(self, user_input: str) -> dict:
        """同步调用（内部走 async，以支持 MCP 异步工具）。"""
        import asyncio

        return asyncio.run(self.ainvoke(user_input))

    async def ainvoke(self, user_input: str) -> dict:
        """异步调用。"""
        return await self.graph.ainvoke({"messages": [("user", user_input)]})
