# 多智能体旅行规划助手

基于 **LangChain + LangGraph** 的多智能体旅行规划系统，集成高德地图、飞猪（flyai）、12306。

## 架构

```
用户输入 → PlannerAgent(LangGraph 状态机)
              ├─ 并行：景点搜索 / 天气 / 酒店 / 交通（互不依赖）
              ├─ 串行：行程规划 → 预算规划（消费前面结果，纯推理）
              └─ 预算超支 → 条件路由回退到酒店（限 MAX_RETRY 次）
```

| Agent | 职责 | 工具 |
|-------|------|------|
| 景点搜索 | POI 搜索、评分筛选 | 高德 `amap_search_poi` + 飞猪 `fliggy_search_poi` |
| 天气查询 | 实时+预报、穿衣建议 | 高德 `amap_weather` |
| 酒店推荐 | 房源、价格、位置 | 飞猪 `fliggy_search_hotel` |
| 交通/车次 | 车次、票价 | 飞猪 `fliggy_search_train`（12306 可选） |
| 行程规划 | 天数分配、顺序优化 | 纯推理（消费前面结果） |
| 预算规划 | 比例预分配、实时核算、超支预警 | 纯推理 |

## 目录结构

```
trip_update/
├── requirements.txt
├── .env.example
├── setup.bat
└── src/trip_agent/
    ├── config.py      # 配置加载
    ├── state.py       # 共享 State (TypedDict)
    ├── tools.py       # 工具层（高德 HTTP + 飞猪 flyai-cli + 12306 MCP）
    ├── agents.py      # 各专业 Agent 定义
    ├── planner.py     # PlannerAgent 主编排图
    └── __main__.py    # CLI 入口
```

## 快速开始

### 1. 安装依赖

```bash
# Python 依赖（用 3.12，系统 3.13 有冲突）
E:\python\python.exe -m pip install -r requirements.txt

# 飞猪 flyai CLI（酒店/机票/火车票/景点搜索）
npm i -g @fly-ai/flyai-cli
flyai config set FLYAI_API_KEY "你的飞猪Key"
```

### 2. 配置凭证

```bash
cp .env.example .env
# 填入 OPENAI_API_KEY(DeepSeek)、AMAP_API_KEY(高德Web服务Key)
```

### 3. 验证 12306 MCP（可选）

```bash
PYTHONPATH=src E:\python\python.exe -m trip_agent verify
```

### 4. 运行

```bash
# mock 模式（离线验证编排逻辑）
PYTHONPATH=src E:\python\python.exe -m trip_agent mock "帮我规划一次3天北京游"

# real 模式（真实 LLM + 真实工具）
PYTHONPATH=src E:\python\python.exe -m trip_agent real "帮我规划一次2天杭州游，预算2000元"
```

## 关键设计决策

1. **飞猪通过 flyai-cli 调用**（不是 HTTP API）：飞猪 flyai 开放平台基于 `@fly-ai/flyai-cli`，
   命令输出单行 JSON 到 stdout，封装成 LangChain 工具时用 `subprocess` 调用。
2. **12306 默认关闭**（`load_tools(include_12306=False)`）：12306 无登录态时 `query_tickets`
   可能因风控长时间挂起，拖垮编排。交通数据默认走飞猪 `fliggy_search_train`（更稳定、含票价），
   需要精确时刻表/余票时再开启 12306 并配置 `RAIL_*` Cookie。
3. **itinerary/budget 是纯推理 Agent**（无工具）：它们消费前面景点/天气/酒店/交通的结果，
   通过 `include_history=True` 把主图消息传给子 agent，避免"工具试错死循环"。
4. **全程 async**：12306 等 MCP 工具是异步 StructuredTool，仅支持 `ainvoke`，
   所以子 agent 和主编排都用 async 执行。
5. **子 agent 设 recursion_limit**：防止 ReAct agent 在真实模型下无限循环调用工具。

## 已知限制与下一步

1. **12306 票价需登录态**：配置 `.env` 中 `RAIL_*` Cookie 后才能拿准确票价。
2. **12306 有严格风控**：控制请求频率，勿用于抢票/囤票/批量刷票。
3. **预算回退是简化版**：当前用关键词检测超支，可改为让 budget agent 输出结构化 JSON 精确判断。
4. **高德有 QPS 限制**：已加并发信号量 + 指数退避重试，个人 Key 并发勿超 2。
5. **飞猪 CLI 每次 subprocess 启动较慢**（~2s），可考虑后续常驻进程或改用其底层 MCP 直连。
