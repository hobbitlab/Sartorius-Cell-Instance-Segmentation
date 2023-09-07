import detectron2
from pathlib import Path
import random, cv2, os
import shutil
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import pycocotools.mask as mask_util
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.engine import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import detectron2.data.transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader

import json  
import optuna  
setup_logger()  

def precision_at(threshold, iou):
    '''
    计算TP, FP, FN
    '''
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    '''
    计算AP score
    '''    
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)


data_dir = '/home/xm/workspace/sartorius-cell-instance-segmentation'  # 数据集路径
output_dir = '/home/xm/workspace/output' # 输出路径
register_coco_instances('sartorius_train',{}, f'{data_dir}/coco_cell_train_fold4.json', data_dir) # 注册train数据集(这里可以调整5个Fold)
register_coco_instances('sartorius_val',{},f'{data_dir}/coco_cell_valid_fold4.json', data_dir) # 注册valid数据集(这里可以调整5个Fold)

SUFF = 122900 # 本次实验ID

# Optuna调参
def objective(trial):
    # 更新实验ID
    global SUFF 
    suffix = str(SUFF)
    SUFF += 1

    model_weights = '/home/xm/workspace/output/livecell_122501/model_0231215_3041.pth'  # LiveCell的预训练权重
    model_yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # 仍然使用 mask_rcnn_R_50_FPN_3x 模型

    cfg = get_cfg() # 生成detectron2的Config
    cfg.INPUT.MASK_FORMAT='bitmask' # label读取格式为bitmask
    # 读取数据集的元数据
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')
    # 新建Output文件夹
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.OUTPUT_DIR = f"{output_dir}/train_{suffix}"
    print("OUTPUT DIR:",cfg.OUTPUT_DIR)

    map_list = []
    class MAPIOUEvaluator(DatasetEvaluator):
        '''
        验证和测试阶段的MAP IOU计算
        '''
        def __init__(self, dataset_name):
            dataset_dicts = DatasetCatalog.get(dataset_name)
            self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
                
        def reset(self):
            self.scores = []

        def process(self, inputs, outputs):
            for inp, out in zip(inputs, outputs):
                if len(out['instances']) == 0:
                    self.scores.append(0)    
                else:
                    targ = self.annotations_cache[inp['image_id']]
                    self.scores.append(score(out, targ))

        def evaluate(self):
            mapiou = np.mean(self.scores)
            map_list.append(mapiou)
            return {"MaP IoU": mapiou}

    # 训练器
    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            '''验证器'''
            return MAPIOUEvaluator(dataset_name)

        def build_hooks(self):
            '''
            只保存分数最高的模型
            '''
            cfg = self.cfg.clone()
            hooks = super().build_hooks()
            # add the best checkpointer hook
            hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, DetectionCheckpointer(self.model, cfg.OUTPUT_DIR), "MaP IoU","max",))
            return hooks


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



    # 打印调参日志
    optuna_params_keys = f"suffix, BASE_LR, MAX_ITER, BATCH_SIZE_PER_IMAGE， SCORE_THRESH_TEST, WARMUP_ITERS, STEPS, max_map_iou"
    optuna_params_value = f"{suffix},  {cfg.SOLVER.BASE_LR:.7f},   {cfg.SOLVER.MAX_ITER},         {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE:>3d},                 {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f},  {cfg.SOLVER.WARMUP_ITERS:>4d},     {str(cfg.SOLVER.STEPS): <46},    "
    print(optuna_params_value)
    with open(f"{output_dir}/optuna.log","a+") as f:
        f.write(optuna_params_value)

    # 开始训练
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 保存训练日志
    max_map_iou = 0
    iteration = 0
    with open(f"{output_dir}/train_{suffix}/metrics.json","r") as f:
        json_file = f.readlines()
        for line in json_file:
            js = json.loads(line)
            if "MaP IoU" in js:
                if js["MaP IoU"] > max_map_iou:
                    max_map_iou = js["MaP IoU"]
                    iteration = js["iteration"]

    # 打印Map IoU分数                
    print("MaP IoU: ", max_map_iou, "iteration: ", iteration)

    old_filename = f"{output_dir}/train_{suffix}/model_best.pth"
    new_filename = f"{output_dir}/{suffix}_model_{model_yaml.split('/')[-1].replace('.yaml','')}_{iteration:07}_{max_map_iou*10000:.0f}.pth"
    shutil.copy(old_filename,  new_filename)
    torch.cuda.empty_cache()
    # 保存调参日志
    with open(f"{output_dir}/optuna.log","a+") as f:
        f.write(f"{max_map_iou:.5f}")
        f.write("\n")

    return max_map_iou

# optuna 参数选择
sampler = optuna.samplers.RandomSampler() # 纯随机采样
study = optuna.create_study(sampler=sampler, direction='maximize')  # Create a new study.
study.optimize(objective, n_trials=1)  # Invoke optimization of the objective function.