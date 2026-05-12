简体中文 | [English](README_en.md)

<div align="center">
  <img src="../docs/logo.png" alt="HitPawImage Logo" width="100%"/>
</div>

<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/HitPaw-Official"><img src="https://img.shields.io/badge/org-HitPaw--Official-orange" alt="Org"></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-brightgreen" alt="Python">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey" alt="Platform">
</p>

---

## 简介

**HitPawImage** 是 [HitPaw-Official](https://github.com/HitPaw-Official) 维护的所有图像相关 AI 算法的官方导航仓库。本仓库**不直接包含**算法源代码，而是作为导航枢纽，提供各独立算法仓库的链接、说明与快速上手指引。

所有算法仓库均独立进行版本管理、文档维护与部署。

---

## 算法列表

| 算法 | 描述 | 仓库链接 | 论文 |
|------|------|----------|------|
| **AIDraw** | 文本生图与草图转图像生成 | [HitPaw-Official/AIDraw](https://github.com/HitPaw-Official/AIDraw) | — |
| **AfsHumanParsing** | 细粒度人体部位语义分割 | [HitPaw-Official/AfsHumanParsing](https://github.com/HitPaw-Official/AfsHumanParsing) | — |
| **FaceAPP-Beautify** | 人脸检测、关键点对齐与美颜处理 | [HitPaw-Official/FaceAPP-Beautify](https://github.com/HitPaw-Official/FaceAPP-Beautify) | — |
| **ImageMatting** | 高精度前景/背景分离（抠图） | [HitPaw-Official/ImageMatting](https://github.com/HitPaw-Official/ImageMatting) | — |
| **NSFWImageClassification** | 安全内容检测与图像审核 | [HitPaw-Official/NSFWImageClassification](https://github.com/HitPaw-Official/NSFWImageClassification) | — |

---

## 仓库结构（各算法通用）

HitPaw-Official 下每个算法仓库均遵循以下目录结构：

```
AlgorithmName/
├── configs/          # 模型与训练配置
├── data/             # 数据集准备脚本
├── deploy/           # 导出、ONNX 及服务化工具
├── docs/             # 扩展文档与资源
├── models/           # 模型定义
├── tools/            # 训练 / 评估入口
├── README.md
└── requirements.txt
```

---

## 快速开始

请前往具体算法仓库查看环境配置说明，各仓库均遵循以下通用流程：

```bash
# 1. 克隆目标算法仓库
git clone https://github.com/HitPaw-Official/<AlgorithmName>.git
cd <AlgorithmName>

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载预训练权重（详见各算法仓库 README）

# 4. 执行推理 — 具体命令请参考各算法仓库的 README
```

---

## 相关导航仓库

| 仓库 | 领域 | 链接 |
|------|------|------|
| HitPawVideo | 视频 AI 算法 | [HitPaw-Official/HitPawVideo](https://github.com/HitPaw-Official/HitPawVideo) |
| HitPawVoice | 音频与语音算法 | [HitPaw-Official/HitPawVoice](https://github.com/HitPaw-Official/HitPawVoice) |

---

## 贡献指南

欢迎对 HitPaw-Official 下任意算法提交贡献。请在**具体算法仓库**中提交 Issue 或 Pull Request，而非本导航仓库。

如需讨论组织层面的事项，可在本仓库提交 Issue。

---

## 支持

- [官方网站](https://www.hitpaw.com/)
- [开发者平台](https://developer.hitpaw.com/)
- [获取 API Key](https://www.hitpaw.com/hitpaw-api.html)

---

## 开源协议

HitPaw-Official 下所有仓库均采用 [Apache 2.0 协议](../LICENSE)，各仓库另有说明者除外。
