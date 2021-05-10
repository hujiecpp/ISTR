from detectron2.config import CfgNode as CN


def add_ISTR_config(cfg):
    """
    Add config for ISTR.
    """
    cfg.MODEL.ISTR = CN()
    cfg.MODEL.ISTR.NUM_CLASSES = 80
    cfg.MODEL.ISTR.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.ISTR.NHEADS = 8
    cfg.MODEL.ISTR.DROPOUT = 0.0
    cfg.MODEL.ISTR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.ISTR.ACTIVATION = 'relu'
    cfg.MODEL.ISTR.HIDDEN_DIM = 256
    cfg.MODEL.ISTR.NUM_CLS = 3
    cfg.MODEL.ISTR.NUM_REG = 3
    cfg.MODEL.ISTR.NUM_MASK = 3
    cfg.MODEL.ISTR.NUM_HEADS = 6

    cfg.MODEL.ISTR.MASK_DIM = 60


    # Dynamic Conv.
    cfg.MODEL.ISTR.NUM_DYNAMIC = 2
    cfg.MODEL.ISTR.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.ISTR.CLASS_WEIGHT = 2.0
    cfg.MODEL.ISTR.GIOU_WEIGHT = 2.0
    cfg.MODEL.ISTR.L1_WEIGHT = 5.0
    cfg.MODEL.ISTR.DEEP_SUPERVISION = True
    cfg.MODEL.ISTR.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.ISTR.MASK_WEIGHT = 2.0

    # Focal Loss.
    cfg.MODEL.ISTR.ALPHA = 0.25
    cfg.MODEL.ISTR.GAMMA = 2.0
    cfg.MODEL.ISTR.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # Matcher
    cfg.MODEL.ISTR.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.ISTR.IOU_LABELS = [0, 1]

    # Encoder
    cfg.MODEL.ISTR.PATH_COMPONENTS = "./projects/ISTR/LME/coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"
