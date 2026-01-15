# Project Iteration Log (项目技术迭代记录)

目标用途：期末大作业报告的技术迭代说明，展示从 Demo 到工业级系统的演进路径。

## Timeline

| 版本号 | 核心变动 | 解决的问题 | 技术细节 |
| --- | --- | --- | --- |
| v1.0 | Bi-LSTM + Embedding 基线模型 | 建立端到端文本分类基线 | Keras 构建双向 LSTM；能够完成基础 Spam/Ham 识别，但对 OOV、同义词替换攻击、隐晦诈骗语义不敏感 |
| v2.0 | Transformer 迁移：DistilBERT Fine-tuning | 提升语义理解能力与长难句识别 | 使用 `DistilBertTokenizerFast` + `TFDistilBertForSequenceClassification`；Fine-tuning 显著增强语义理解，但对含链接短信过敏，False Positive 激增 |
| v3.0 (Current) | Hybrid Architecture：BERT + Regex + Whitelist | False Positive Reduction + 对抗测试闭环 | 规则引擎叠加模型评分；加入白名单强制覆盖；引入自动化审计进行 Adversarial Testing；新增 sextortion/stock_scam/money_mule 规则与数据增强，实现高难度测试集 100% Accuracy |

## Detailed Iteration Log

### v1.0 — 基准模型搭建（Bi-LSTM + Embedding）
- 版本号：v1.0
- 核心变动：使用 Keras 构建 Bi-LSTM 文本分类器，形成基线版本
- 解决的问题：完成短信 Spam/Ham 端到端识别流程
- 技术细节：
  - 使用 Embedding 将词汇映射为向量表示
  - Bi-LSTM 进行序列语义建模
  - 主要痛点：OOV、同义词替换攻击、隐晦诈骗语义表达

### v2.0 — 拥抱 Transformer（DistilBERT Fine-tuning）
- 版本号：v2.0
- 核心变动：弃用 LSTM，迁移到 DistilBERT 预训练模型
- 解决的问题：显著提升语义理解能力与长难句识别能力
- 技术细节：
  - 使用 `DistilBertTokenizerFast` 分词以缓解 OOV
  - 使用 `TFDistilBertForSequenceClassification` 进行二分类 Fine-tuning
  - 产生新问题：模型对链接过于敏感，将 Google Docs/Zoom 等正常链接误判为 SPAM，False Positive 增多

### v3.0 — 工业级混合架构与红蓝对抗（Hybrid Architecture）
- 版本号：v3.0（Current）
- 核心变动：引入“BERT + Regex Rule Engine + Whitelist”的混合架构
- 解决的问题：
  - False Positive Reduction：修复含正常链接的误判问题
  - Adversarial Testing：提升对隐蔽诈骗与变体攻击的鲁棒性
- 技术细节：
  - **混合评分机制（Hybrid Scoring）**：`calculate_hybrid_score()` 中将模型分数与规则风险因子叠加
  - **白名单强制覆盖（Whitelist Override）**：对 `google.com`, `docs.google.com`, `zoom.us`, `teams.microsoft.com` 等域名进行强制安全覆盖
  - **规则引擎补强（Regex Rule Engine）**：新增 `sextortion`, `stock_scam`, `money_mule` 等高危模式捕获
  - **数据增强（Data Augmentation）**：手工构造高风险样本追加到训练数据
  - **自动化审计（auto_audit.py）**：构建 Red/Blue Team 高难度样本集（30条），进行 Adversarial Testing

## Current Technical Snapshot (v3.0)
- Architecture：Hybrid Architecture（Fine-tuned DistilBERT + Regex Rule Engine + Whitelist）
- Evaluation：Adversarial Testing Accuracy 100% (30/30)
- Key Terms：Fine-tuning, False Positive Reduction, Adversarial Testing, Hybrid Architecture
