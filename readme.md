# Sartorius-Cell-Instance-Segmentation Top-2-Percent-Solution

![avatar](https://storage.googleapis.com/kaggle-competitions/kaggle/30201/logos/header.png?t=2021-09-03-15-27-46)

https://www.kaggle.com/c/sartorius-cell-instance-segmentation

## 比赛介绍

本次竞赛中，你将在描绘神经系统疾病研究中常用的神经细胞类型的生物图像中检测并划定明显的感兴趣的对象。 更具体地说，你将使用相位对比显微镜图像来训练和测试你的模型，以便**对神经细胞进行实例分割**。成功的模型将以较高的精确度完成这一工作。

- 竞赛类型：本次竞赛属于**计算机视觉-实例分割**，所以推荐使用的模型或者库：Mask R-CNN / Detectron2 
- 赛题数据：赛题数据图像的数量不多，但注释对象的数量却相当多。 训练集：606张图片；隐藏的测试集：大约是240张图片。
- 评估标准：**MAP-IOU**  详见：[Sartorius - Cell Instance Segmentation | Kaggle](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/overview/evaluation)
- 推荐阅读 Kaggle 内的一篇 EDA（探索数据分析）来获取一些预备知识：[[TRAIN\] Sartorius Segmentation 🧬 EDA+EffDET [TF] | Kaggle](https://www.kaggle.com/dschettler8845/train-sartorius-segmentation-eda-effdet-tf)



## 数据说明

官方数据页面 [Sartorius - Cell Instance Segmentation | Kaggle](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/data)

- 赛题数据图像的数量不多，但注释对象的数量却相当多。 训练集：606张图片；隐藏的测试集：大约是240张图片。除了本次比赛的正式数据集，官方还在DataSets中提供了两份额外的数据集。
- train_semi_supervised - 在你想使用半监督方法的额外数据时提供的未标记的图像。
- LIVECell_dataset_2021 - LIVECell数据集的数据的镜像。LIVECell是本次比赛的前身数据集。你会发现SH-HY5Y细胞系的额外数据，以及其他几个在竞赛数据集中没有涉及的细胞系，这些数据可能对迁移学习有帮助。

注意：虽然最终预测不允许实例之间的重叠，但训练标签是提供完整的、包括重叠的部分。这是为了确保向模型提供每个物体的完整数据。而消除预测中的重叠部分是参赛者的任务。




## 解决方案思路

本次竞赛我们的方案采用了**Pretrain + Finetune**的形式，我们先在 LIVECell_dataset_2021 上做了pretrain，然后在官方竞赛数据集上做了finetune。

检测库上我们使用了 Facebook 的 **Detectron2**，模型上我们选择了 **mask_rcnn_R_50_FPN_3x** ，Detectron2的优势是集成较好，可配置的参数多，缺点是：自定义一些函数比较麻烦。

方案参考自 [@odede ](https://www.kaggle.com/markunys)的两篇notebook：

1. [Sartorius transfer learning train with livecell\] | Kaggle](https://www.kaggle.com/markunys/sartorius-transfer-learning-train-with-livecell)
2. [Sartorius transfer learning train\] | Kaggle](https://www.kaggle.com/markunys/sartorius-transfer-learning-train)

#### LIVECell pretrain：

- IMS_PER_BATCH = 1

- BASE_LR：0.001

- MAX_ITER = 260000

- BATCH_SIZE_PER_IMAGE：128

- SCORE_THRESH_TEST：0.4

- ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]

- ANCHOR_GENERATOR.SIZES = [[4], [8],[16],[64],[256]]

- MIN_SIZE_TRAIN = (480, 520,  560, 640, 672, 736, 800, 864, 928)

- MIN_SIZE_TEST = 928

#### Finetune

- IMS_PER_BATCH = 1
- BASE_LR：0.0011
- MAX_ITER = 24000
- SOLVER.STEPS = [13000,17000,20000,22000,23000]
- SOLVER.WARMUP_ITERS = 100
- SOLVER.GAMMA = 0.5
- BATCH_SIZE_PER_IMAGE = 128
- SCORE_THRESH_TEST：0.4

- ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]

- ANCHOR_GENERATOR.SIZES = [[4], [8],[16],[64],[256]]

- MIN_SIZE_TRAIN = (480, 520,  560, 640, 672, 736, 800, 864, 928)

- MIN_SIZE_TEST = 928





## 比赛上分历程

1. 使用 [@odede ](https://www.kaggle.com/markunys)的 LIVECell pretrain 和 train finetune，Public LB : 0.300
2. 调整finetune的学习率相关参数，Public LB : 0.305
3. 改成全量训练数据，跑10个seed并简单平均融合，Public LB : 0.1380
4. 尝试clean astro mask（[Sartorius - Cell Instance Segmentation | Kaggle](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/291371)），Public LB : 0.304 ，遂放弃使用
5. 调整finetune的SCORE_THRESH_TEST和BATCH_SIZE_PER_IMAGE ，Public LB : 0.309
6. 调整finetune的 MIN_SIZE_TRAIN 和 MIN_SIZE_TEST，Public LB : 0.311
7. 调整finetune的 ANCHOR_GENERATOR，Public LB : 0.315
8. 尝试做一些简单的数据增强，Public LB : 0.315，遂放弃使用
9. 精细化调整学习率参数，Public LB : 0.317
10. 将finetune的参数使用到pretrain阶段，然后再finetune，Public LB : 0.320
11. 训练5个Fold，其中最好的Fold，Public LB : 0.326
12. Ensemble NMS（[Ensemble Non-Maximum Supp - Detectron2 Inference\] | Kaggle](https://www.kaggle.com/ammarnassanalhajali/ensemble-non-maximum-supp-detectron2-inference)），Public LB : 0.332，Private LB : 0.340。



## 参数代码

#### Pretrain

```python
cfg.SOLVER.IMS_PER_BATCH = 1 # 每个batch包含的图片数量
cfg.SOLVER.BASE_LR = 0.001 # 初始学习率
cfg.SOLVER.MAX_ITER = 260000 # 迭代总次数
cfg.SOLVER.STEPS = [] # 学习率下降的step位置，这里为空，则学习率保持不变
cfg.SOLVER.CHECKPOINT_PERIOD = (len(DatasetCatalog.get('sartorius_train')) + len(DatasetCatalog.get('sartorius_test'))) // cfg.SOLVER.IMS_PER_BATCH  # 每个epoch都保存一次模型
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # 每个图像用于训练RPN的区域数
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10 # 标签类别数量
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .4 # 最小分数阈值（假设分数在[0, 1]范围内）；选择这个值是为了平衡获得高召回率和没有太多的低精度检测，因为低精度检测会减慢推理的后处理步骤（如NMS）。
cfg.TEST.EVAL_PERIOD = (len(DatasetCatalog.get('sartorius_train')) + len(DatasetCatalog.get('sartorius_test'))) // cfg.SOLVER.IMS_PER_BATCH  # 每个epoch都做一次验证

cfg.OUTPUT_DIR = f"./output/livecell_{suffix}" # 模型/日志 输出保存地址
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(cfg.OUTPUT_DIR)

cfg.SEED = 42 # 随机种子

cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]] # anchor长和宽的比例
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4], [8],[16],[64],[256]] # anchor的尺寸

cfg.INPUT.MIN_SIZE_TRAIN = (480, 520,  560, 640, 672, 736, 800, 864, 928)  # 训练阶段短边的最小值（多尺寸）
cfg.INPUT.MIN_SIZE_TEST = 928 # 测试阶段短边的最小值
cfg.INPUT.MAX_SIZE_TEST = 1333 # 测试阶段短边的最大值
cfg.INPUT.MAX_SIZE_TRAIN = 1333 # 训练阶段短边的最大值
```



#### Finetune

```python
cfg.merge_from_file(model_zoo.get_config_file(model_yaml)) # 使用 mask_rcnn_R_50_FPN_3x 模型
cfg.DATASETS.TRAIN = ("sartorius_train") # 创建训练集
cfg.DATASETS.TEST = ("sartorius_val",) # 创建验证集
cfg.DATALOADER.NUM_WORKERS = 10 # cpu线程数
cfg.MODEL.WEIGHTS = model_weights # checkpoint选择LiveCell的预训练

cfg.SOLVER.IMS_PER_BATCH = 1 # 每个batch包含的图片数量
cfg.SOLVER.BASE_LR = trial.suggest_float('BASE_LR', 0.0005, 0.01, log=True)  # 初始学习率 调参best:0.0011
cfg.SOLVER.MAX_ITER = trial.suggest_categorical('MAX_ITER', [20000, 24000]) # 迭代次数 调参best:24000
cfg.SOLVER.STEPS = [13000,17000,20000,22000,23000] # 学习率下降的step位置
cfg.SOLVER.WARMUP_ITERS = trial.suggest_categorical('WARMUP_ITERS', [100, 500, 1000]) # 升温迭代数 调参,best:100
cfg.SOLVER.GAMMA = 0.5 # 学习率下降倍数
cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER+1 # 保存checkpoint，仅最后

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = trial.suggest_categorical('BATCH_SIZE_PER_IMAGE', [96, 128])   # 每个图像用于训练RPN的区域数 调参best:128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 # 标签类别数量
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = trial.suggest_categorical('SCORE_THRESH_TEST', [0.3, 0.4, 0.5]) # 最小分数阈值（假设分数在[0, 1]范围内）；选择这个值是为了平衡获得高召回率和没有太多的低精度检测，因为低精度检测会减慢推理的后处理步骤（如NMS）。 调参best:0.4
cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH  # 每个epoch都做一次验证
cfg.TEST.DETECTIONS_PER_IMAGE = 1000 # 每张图片支持检测的最大实例数量
cfg.SEED = 42 # 随机种子

cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]] # anchor长和宽的比例
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4], [8],[16],[64],[256]] # anchor的尺寸

cfg.INPUT.MIN_SIZE_TRAIN = (480, 520,  560, 640, 672, 736, 800, 864, 928)  # 训练阶段短边的最小值（多尺寸）
cfg.INPUT.MIN_SIZE_TEST = 928 # 测试阶段短边的最小值
cfg.INPUT.MAX_SIZE_TEST = 1333 # 测试阶段短边的最大值
cfg.INPUT.MAX_SIZE_TRAIN = 1333 # 训练阶段短边的最大值
```



## 代码、数据集

+ 代码
  + sartorius-livecel-pretrain.ipynb
  + sartorius-train-finetune.py
  + sartorius-nms-inference.ipynb
+ 数据集
  - 官网图片数据： [Sartorius - Cell Instance Segmentation | Kaggle](https://www.kaggle.com/c/ventilator-pressure-prediction/data)
  - mask数据：见网盘

## TL;DR

竞赛是由Sartorius举办的，参赛者将对神经细胞进行实例分割。本次竞赛中我们团队选择的方案是 Detectron2 + Mask R-CNN。利用了LIVECell数据集做了pretrain后，对官方数据进行finetune起到了不错的效果。并且在参数选择上面，我们扩大了anchor的选择范围以及多尺度图像，达到了银牌的成绩，最后使用NMS对分数较好的模型做Ensemble。此外，我们还尝试了数据增强，生成clean mask代替源数据中的dirty mask，WBF 等，但没有起效果。最终我们获得了Private LB: 0.340 (Top2%) 的成绩。

