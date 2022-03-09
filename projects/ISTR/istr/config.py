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
    # cfg.MODEL.ISTR.ACTIVATION = 'relu'
    cfg.MODEL.ISTR.HIDDEN_DIM = 256
    cfg.MODEL.ISTR.NUM_CLS = 3
    cfg.MODEL.ISTR.NUM_REG = 3
    cfg.MODEL.ISTR.NUM_MASK = 3
    cfg.MODEL.ISTR.NUM_HEADS = 6

    cfg.MODEL.ISTR.MASK_SIZE = 112
    cfg.MODEL.ISTR.MASK_FEAT_DIM = 256

    # Dynamic Conv.
    cfg.MODEL.ISTR.NUM_DYNAMIC = 2
    cfg.MODEL.ISTR.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.ISTR.CLASS_WEIGHT = 2.0
    cfg.MODEL.ISTR.GIOU_WEIGHT = 2.0
    cfg.MODEL.ISTR.L1_WEIGHT = 5.0
    cfg.MODEL.ISTR.DEEP_SUPERVISION = True
    cfg.MODEL.ISTR.NO_OBJECT_WEIGHT = 0.1

    cfg.MODEL.ISTR.MASK_WEIGHT = 5.0
    cfg.MODEL.ISTR.FEAT_WEIGHT = 1.0

    # Focal Loss.
    cfg.MODEL.ISTR.ALPHA = 0.25
    cfg.MODEL.ISTR.GAMMA = 2.0
    cfg.MODEL.ISTR.PRIOR_PROB = 0.01

    # Matcher
    cfg.MODEL.ISTR.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.ISTR.IOU_LABELS = [0, 1]

    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    cfg.LSJ_AUG = False

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    cfg.MODEL.ISTR.PATH_COMPONENTS = ''
    cfg.MODEL.ISTR.MASK_ENCODING_METHOD = 'AE'