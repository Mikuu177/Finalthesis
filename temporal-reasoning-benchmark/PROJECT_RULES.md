# 项目规则（Project Rules）

1. 总指挥 GPT 汇报机制（必须遵守）
- 目标：确保项目每一步有据可查、可审计、可指挥。
- 规则：每次关键操作后，向“总指挥 GPT”进行阶段性汇报，并在聊天回复的开头显式给出“向总指挥 GPT 汇报”部分，包含：
  - 当前阶段/背景（处于哪个里程碑/任务）
  - 本次完成事项（基于实际代码改动/新增文件清单/配置变更）
  - 产出物（路径与文件名：如 predictions.jsonl、summary.csv、配置）
  - 风险与依赖（阻塞点、待确认项）
  - 下一步建议与需要的指令（请总指挥明确拍板）
- 触发时机：
  - 创建/修改核心脚本（runner、adapters、scorer、prompt 构建器）
  - 变更配置（configs/*.yaml）与数据（data/*）
  - 完成一次实验运行（产生新的 run 目录）
  - 引入或移除一个模型/数据集/评测维度
  - 更新评分/规范化/工作流逻辑
- 汇报模板（建议）：
  - 向总指挥 GPT 汇报：<一句话目标/阶段>
  - 完成事项：<要点列表 + 关键文件路径>
  - 产出物：<文件与目录>
  - 风险/依赖：<列表>
  - 申请指令：<需要拍板的选项/参数/下一步>

2. 可复现与配置驱动
- 任何一次实验必须由单一配置文件完整描述（configs/baseline.yaml 等）。
- shots 抽样可复现（固定 seed）。
- 结果可重跑且支持断点续跑与缓存（.cache/ 与 outputs/runs/*）。

3. 安全与密钥管理
- 所有 API Key 通过 .env 管理，不得写入代码库。（示例放置 .env.example 或文档说明）
- models.yaml 使用 api_key_env 字段间接引用密钥名。

4. 评测与日志留存
- predictions.jsonl 必须包含：prompt、raw、pred、gold、correct、match、usage（可得时）、latency、error。
- summary.csv 至少包含：model、category、n、correct、accuracy。
- 分析脚本（src/analysis.py）用于多次运行的汇总导出。

5. 扩展性与模块化
- 新模型接入需通过 adapters/* 新增，runner 主逻辑不改。
- 新数据集需加载成统一 schema：{id, question, context(optional), gold, category}。
- 评分与规范化策略以模块形式扩展（normalize/score），可通过配置开关选择 strict/relaxed。

6. 语言与沟通
- 默认使用中文简体进行交流与文档撰写。
- 每次沟通遵循第 1 条“总指挥 GPT 汇报机制”。






