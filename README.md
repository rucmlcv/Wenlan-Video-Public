# Wenlan-Video-Public

Wenlan-Video-Public是基于Wenlan 2.0（首个中文通用图文多模态大规模预训练模型）的文澜视频多模态预训练模型。

Wenlan 1.0论文：[WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training](https://arxiv.org/abs/2103.06561)

Wenlan 2.0论文：[WenLan 2.0: Make AI Imagine via a Multimodal Foundation Model](https://arxiv.org/abs/2110.14378)

## 适用场景

适用场景示例：视频检索文本、文本检索视频、视频标注、视频零样本分类、作为其他下游多模态任务的输入特征等。

## 技术特色

1. Wenlan-Video-Public使用对比学习算法将图像/视频和文本映射到了同一特征空间，可用于弥补视觉特征和文本特征之间存在的隔阂。
2. 基于视觉-语言弱相关的假设，除了能理解对图像/视频的描述性文本外，也可以捕捉图像/视频和文本之间存在的抽象联系。
3. 视觉编码器和文本编码器可分别独立运行，有利于实际生产环境中的部署。 
4. 三亿通用图文对+50万爱奇艺视频联合训练，强大的泛化性与通用性。


## 运行环境
```
python 3.8
torch 1.8
jsonlines
tqdm
easydict
torchvision
transformers
timm
```

## 模型下载
敬请期待

## 快速使用

1. 安装wenlan-video-public库
```
pip install wenlan-video-public==1.0.2
```

2. 导入模型

```
from wenlan_video import load_wenlan_model

# load_checkpoint 下载好的模型地址
# cfg_file github目录下moco_box.yaml文件地址
model = load_wenlan_model(load_checkpoint, cfg_file, device=device)
```

3. 读取视频/文本

```
from wenlan_video import wenlan_transforms
wenlan_transforms = wenlan_transforms()

# VIDEO_PATH为视频抽帧好的帧（jpg, png），帧数大于10，命名按顺序标号
video, video_boxes = wenlan_transforms.video_transform(VIDEO_PATH, device=device)
text, textMask = wenlan_transforms.text_transform(‘Hello Wenlan’, device=device)
```

4. 同时抽取视频/文本特征
```
videoFea, textFea = model(video, video_boxes, text.unsqueeze(0), textMask.unsqueeze(0))
```

5. 分别抽取视频/文本特征
```
videoFea = model.encode_video(videoFea, video_boxes)
textFea = model.encode_text(texts.unsqueeze(0), maskTexts.unsqueeze(0))
```

## Have Fun!
