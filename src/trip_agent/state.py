"""共享 State 定义。

所有 Agent 通过这个 TypedDict 共享数据，作为 LangGraph 图的状态总线。
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """多智能体共享状态。

    total=False 表示字段均可选，各节点只读写自己关心的部分。
    """

    # ---- 消息与最终答复 ----
    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: str
    structured: dict[str, Any]      # 各 Agent 结构化 JSON 结果（供前端渲染）

    # ---- 用户输入 ----
    destination: str          # 目的地城市
    start_city: str           # 出发城市
    days: int                 # 旅行天数
    people: int               # 出行人数
    budget: float             # 总预算（元）
    preferences: str          # 用户偏好（自由文本，如"偏好自然风光"）

    # ---- 各 Agent 中间结果 ----
    attractions: list[dict[str, Any]]   # 景点列表（含名称、坐标、评分）
    weather: dict[str, Any]             # 天气信息
    hotels: list[dict[str, Any]]        # 酒店候选
    trains: list[dict[str, Any]]        # 车次候选
    itinerary: dict[str, Any]           # 行程方案（按天组织）

    # ---- 预算 ----
    budget_breakdown: dict[str, Any]    # 预算明细
    over_budget: bool                   # 是否超支
    retry_count: int                    # 预算回退重试次数（防死循环）
