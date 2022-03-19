# Wenlan-Video-Public

Wenlan-Video-Public是基于Wenlan 2.0（首个中文通用图文多模态大规模预训练模型）的文澜视频多模态预训练模型。

BriVL论文：[WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training](https://arxiv.org/abs/2103.06561)

Wenlan 2.0论文：[WenLan 2.0: Make AI Imagine via a Multimodal Foundation Model](https://arxiv.org/abs/2110.14378)

## 适用场景

适用场景示例：视频检索文本、文本检索视频、视频标注、视频零样本分类、作为其他下游多模态任务的输入特征等。

## 技术特色

1. Wenlan-Video-Public使用对比学习算法将视频和文本映射到了同一特征空间，可用于弥补视觉特征和文本特征之间存在的隔阂。
2. 基于视觉-语言弱相关的假设，除了能理解对视频的描述性文本外，也可以捕捉视频和文本之间存在的抽象联系。
3. 视觉编码器和文本编码器可分别独立运行，有利于实际生产环境中的部署。 
4. 三亿通用图文对+50万爱奇艺视频联合训练，强大的泛化性与通用性

## 3月20日开源，敬请期待
