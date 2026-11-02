"""命令行入口。

用法：
    python -m trip_agent mock "帮我规划一次3天北京游，预算3000"
    python -m trip_agent real  "帮我规划一次3天北京游，预算3000"
    python -m trip_agent verify
"""
from __future__ import annotations

import asyncio
import sys


def _build_mock_llm():
    """构建 mock LLM（离线验证编排逻辑用）。"""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatResult

    from .tools import build_mock_tools

    class _SeqLLM(BaseChatModel):
        """按调用次数循环返回预设内容的假模型。"""

        sequence: list[str] = [
            '{"destination": "北京", "start_city": "", "days": 3, "budget": 3000.0, "preferences": "自然风光"}',
            "最终行程：3天北京游，包含故宫、长城等景点。",
        ]
        _count: int = 0

        @property
        def _llm_type(self) -> str:
            return "seq-fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            content = self.sequence[self._count % len(self.sequence)]
            self._count += 1
            return ChatResult(generations=[{"text": content, "message": AIMessage(content=content)}])

        def bind_tools(self, tools, **kwargs):
            return self

    return _SeqLLM(), _SeqLLM(), build_mock_tools()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    mode = args[0]

    if mode == "verify":
        from .tools import verify_12306_connection
        result = asyncio.run(verify_12306_connection())
        print("12306 MCP 连接成功，可用工具：")
        for name in result["tool_names"]:
            print(f"  - {name}")
        return

    user_input = args[1] if len(args) > 1 else "帮我规划一次旅行"

    from .planner import TripPlanner

    if mode == "mock":
        llm, planner_llm, tools = _build_mock_llm()
        planner = TripPlanner(llm, planner_llm, tools)
        result = planner.invoke(user_input)
        _print_result(result)
    elif mode == "real":
        asyncio.run(_run_real(user_input))
    else:
        print(f"未知模式：{mode}，可选 mock / real / verify")


async def _run_real(user_input: str) -> None:
    """真实模式：同一事件循环内完成工具加载与编排执行。"""
    from langchain_openai import ChatOpenAI

    from .config import settings
    from .planner import TripPlanner
    from .tools import load_tools

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
    planner = TripPlanner(llm, planner_llm, tools)
    result = await planner.ainvoke(user_input)
    _print_result(result)


def _print_result(result: dict) -> None:
    print("\n========== 最终规划 ==========")
    print(result.get("final_answer", "(无最终答复)"))
    print("\n========== 中间消息 ==========")
    for m in result.get("messages", []):
        content = getattr(m, "content", m)
        print(f"- {str(content)[:200]}")


if __name__ == "__main__":
    main()
