# 多智能体旅行规划助手 🧳

基于 **LangChain + LangGraph** 的多智能体旅行规划系统，集成 **高德地图、飞猪（flyai）、12306、Unsplash**，提供从景点搜索、天气、酒店、交通、餐饮到预算核算的一站式旅行规划，并配有可视化 Web 界面与按天分色的地图标记。

> ✨ 支持 CLI 命令行与 Web 界面两种使用方式，端到端自动编排，预算超支自动回退重排。

## ✨ 功能特性

- **多智能体协作**：景点 / 天气 / 酒店 / 交通 四个专业 Agent 并行搜索，行程 / 预算 Agent 串行编排
- **可视化 Web 界面**：表单输入 → 模块化卡片结果（行程 / 地图 / 天气 / 酒店 / 交通 / 预算 / 景点），左侧导航栏点击跳转
- **高德地图可视化**：按天分色标记每个景点、路线连线、点击弹窗查看详情
- **真实景点图片**：Unsplash 高清图库（英文关键词映射命中），回退 Wikimedia、渐变占位图兜底
- **行程分块编排**：每天按景点分块，含时段、景点介绍、景点间交通安排、附近餐饮推荐
- **天气预报图例**：每日网格卡片 + 天气图标 + 温度风向 + 穿衣建议
- **预算智能核算**：按人数×票价 / 住宿间数×晚数 实时核算，超支自动回退并给出降级建议

## 🏗️ 架构

```
用户输入 → PlannerAgent (LangGraph 状态机)
              ├─ 并行：景点搜索 / 天气 / 酒店 / 交通（互不依赖）
              ├─ 串行：行程规划 → 预算规划（消费前面结果，纯推理）
              └─ 预算超支 → 条件路由回退到酒店（限 MAX_RETRY 次）
```

| Agent | 职责 | 工具 |
|-------|------|------|
| 景点搜索 | POI 搜索、评分筛选、门票 | 高德 `amap_search_poi` + 飞猪 `fliggy_search_poi` |
| 天气查询 | 实时 + 预报、穿衣建议 | 高德 `amap_weather` |
| 酒店推荐 | 房源、价格、位置 | 飞猪 `fliggy_search_hotel` |
| 交通/车次 | 车次、票价（按人数核算） | 飞猪 `fliggy_search_train`（12306 可选） |
| 餐饮推荐 | 附近美食、人均 | 高德 `amap_search_food` |
| 行程规划 | 天数分配、分块编排、交通/餐饮衔接 | 纯推理（消费前面结果） |
| 预算规划 | 比例预分配、实时核算、超支预警 | 纯推理 |

## 📁 目录结构

```
trip_update/
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板（复制为 .env 填入真实值）
├── setup.bat               # Windows 一键安装依赖
├── web_server.py           # Web 服务（http.server + API）
├── web/
│   └── index.html          # 前端页面（表单 + 模块化结果 + 高德地图）
└── src/trip_agent/
    ├── config.py           # 配置加载（.env）
    ├── state.py            # 共享 State (TypedDict)
    ├── tools.py            # 工具层（高德 HTTP + 飞猪 flyai-cli + 12306 MCP + 图片搜索）
    ├── agents.py           # 各专业 Agent 定义与提示词
    ├── planner.py          # PlannerAgent 主编排图（LangGraph StateGraph）
    └── __main__.py         # CLI 入口（mock / real / verify）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# Python 依赖（建议用 Python 3.12，系统 3.13 可能有依赖冲突）
pip install -r requirements.txt

# 飞猪 flyai CLI（酒店 / 机票 / 火车票 / 景点搜索）
npm i -g @fly-ai/flyai-cli
flyai config set FLYAI_API_KEY "你的飞猪Key"
```

### 2. 配置凭证

```bash
cp .env.example .env
# 填入真实值：
#   OPENAI_API_KEY      —— DeepSeek（或任意兼容 OpenAI 协议的 LLM）密钥
#   AMAP_API_KEY        —— 高德 Web 服务 Key（后端 POI / 天气 / 路径）
#   AMAP_JS_KEY         —— 高德 Web 端 JS Key（前端地图）
#   UNSPLASH_ACCESS_KEY —— Unsplash 图库（可选，用于真实景点图片）
```

各 Key 申请地址：
- DeepSeek：https://platform.deepseek.com
- 高德开放平台：https://lbs.amap.com
- Unsplash（可选）：https://unsplash.com/developers

### 3. 运行

**方式一：Web 界面（推荐）**

```bash
python web_server.py
# 浏览器打开 http://127.0.0.1:8899
```

**方式二：CLI 命令行**

```bash
# mock 模式（离线验证编排逻辑，无需凭证）
PYTHONPATH=src python -m trip_agent mock "帮我规划一次3天北京游"

# real 模式（真实 LLM + 真实工具）
PYTHONPATH=src python -m trip_agent real "帮我规划一次2天杭州游，预算2000元"

# 验证 12306 MCP 连接（可选）
PYTHONPATH=src python -m trip_agent verify
```

> Windows 下若 `PYTHONPATH=src` 语法报错，请用 `set PYTHONPATH=src` 或直接用 Web 界面（`web_server.py` 已自动处理路径）。

## 🔑 关键设计决策

1. **飞猪通过 flyai-cli 调用**（非 HTTP API）：飞猪 flyai 开放平台基于 `@fly-ai/flyai-cli`，命令输出单行 JSON 到 stdout，封装成 LangChain 工具时用 `subprocess` 调 `node.exe + flyai-bundle.cjs` 执行。
2. **12306 默认关闭**（`load_tools(include_12306=False)`）：无登录态时 `query_tickets` 可能因风控长时间挂起拖垮编排。交通数据默认走飞猪（更稳定、含票价），需精确时刻表/余票时再开启 12306 并配置 `RAIL_*` Cookie。
3. **itinerary/budget 是纯推理 Agent**（无工具）：消费前面景点/天气/酒店/交通结果，通过 `include_history=True` 传递历史，避免「工具试错死循环」。
4. **全程 async**：12306 等 MCP 工具是异步 StructuredTool，仅支持 `ainvoke`，故子 Agent 与主编排全用 async 执行。
5. **子 Agent 设 recursion_limit**：防止 ReAct Agent 在真实模型下无限循环调用工具。
6. **结构化输出**：各 Agent 输出 JSON，后端 `web_server.py` 统一解析为结构化数据（含坐标匹配、景点图片匹配），供前端模块化渲染与地图打点。

## 📝 已知限制

1. **12306 票价需登录态**：配置 `.env` 中 `RAIL_*` Cookie 后才能拿准确票价。
2. **12306 有严格风控**：控制请求频率，请勿用于抢票 / 囤票 / 批量刷票。
3. **高德有 QPS 限制**：已加并发信号量 + 指数退避重试，个人 Key 并发勿超 2。
4. **飞猪 CLI 每次 subprocess 启动较慢**（约 2s），可考虑后续常驻进程或改用其底层 MCP 直连。
5. **Unsplash 免费额度 50 次/小时**：超出或未配置时自动回退 Wikimedia / 渐变占位图。

## ⚠️ 安全提示

- 请勿将真实密钥提交到 Git。`.env` 已被 `.gitignore` 忽略，`.env.example` 中仅为占位符。
- 若曾误提交密钥，请立即在对应平台吊销并重置。
