import copy
import math

import torch
from torch import nn

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        self.box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.mask_pooler = self._init_mask_pooler(cfg, roi_input_shape)
        
        # Build heads.
        num_classes = cfg.MODEL.ISTR.NUM_CLASSES
        d_model = cfg.MODEL.ISTR.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.ISTR.DIM_FEEDFORWARD
        nhead = cfg.MODEL.ISTR.NHEADS
        dropout = cfg.MODEL.ISTR.DROPOUT
        self.num_heads = cfg.MODEL.ISTR.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout)
        self.head_series = _get_clones(rcnn_head, self.num_heads)
        self.return_intermediate = cfg.MODEL.ISTR.DEEP_SUPERVISION
        
        # Init parameters.
        self.num_classes = num_classes
        prior_prob = cfg.MODEL.ISTR.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler
    
    @staticmethod
    def _init_mask_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = 28 #cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return mask_pooler

    def forward(self, features, init_bboxes, init_features, targets = None, criterion = None, mask_E = None, mask_D = None):

        bboxes = init_bboxes
        
        proposal_features = init_features.clone()

        losses = {}

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_mask_logits = []
        inter_roi_feats = []
        
        for stage, rcnn_head in enumerate(self.head_series):
            
            if criterion != None or stage == self.num_heads-1:
                class_logits, pred_bboxes, proposal_features, mask_logits, ret_roi_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler, self.mask_pooler)

                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
                inter_mask_logits.append(mask_logits)
                inter_roi_feats.append(ret_roi_features)

            else:
                class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler, None)
                
                if stage >= 2:
                    inter_class_logits.append(class_logits)

            if criterion != None:
                output = {'pred_logits': class_logits, 'pred_boxes': pred_bboxes, 'pred_masks': mask_logits, 'pred_roi_feats': ret_roi_features}
                tmp_loss = criterion(output, targets, mask_E, mask_D, stage)
                losses.update(tmp_loss)

            bboxes = pred_bboxes.detach()

        if criterion != None:
            return losses

        inter_class_logits = torch.mean(torch.stack(inter_class_logits), 0)
        inter_pred_bboxes = torch.mean(torch.stack(inter_pred_bboxes), 0)
        inter_mask_logits = torch.mean(torch.stack(inter_mask_logits), 0)
        inter_roi_feats = torch.mean(torch.stack(inter_roi_feats), 0)

        # return class_logits[None], pred_bboxes[None], mask_logits[None], ret_roi_features[None]
        return inter_class_logits[None], inter_pred_bboxes[None], inter_mask_logits[None], inter_roi_feats[None]


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model
        self.cfg = cfg

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ELU(inplace=True)

        # cls.
        num_cls = cfg.MODEL.ISTR.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ELU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.ISTR.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ELU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        self.mask_module = nn.Sequential(
            nn.Conv2d(d_model, d_model, 4, 2, 1),
            nn.BatchNorm2d(d_model),
            nn.ELU(True),
            nn.Conv2d(d_model, d_model, 4, 2, 1),
            nn.BatchNorm2d(d_model),
            nn.ELU(True),
            nn.Conv2d(d_model, d_model, 7, 1),)

        if self.cfg.MODEL.ISTR.MASK_ENCODING_METHOD == 'AE':
            self.ret_roi_layer_1 = conv_block(in_ch=d_model, out_ch=64)
            self.ret_roi_layer_2 = conv_block(in_ch=64, out_ch=32)

        # pred.
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights


    def forward(self, features, bboxes, pro_features, pooler_box, pooler_mask):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]
        
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler_box(features, proposal_boxes)

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)        

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)

        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)

        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)

        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()

        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)

        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4)).view(N, nr_boxes, -1)

        if pooler_mask != None:
            proposal_boxes = list()
            for b in range(N):
                proposal_boxes.append(Boxes(pred_bboxes[b]))
            ret_roi_features = pooler_mask(features, proposal_boxes)

            mask_logits = self.mask_module(ret_roi_features)

            if self.cfg.MODEL.ISTR.MASK_ENCODING_METHOD == 'AE':
                ret_roi_features = self.ret_roi_layer_1(ret_roi_features)
                ret_roi_features = self.ret_roi_layer_2(ret_roi_features)
            
            return class_logits.view(N, nr_boxes, -1), pred_bboxes, obj_features, mask_logits.view(N, nr_boxes, -1), ret_roi_features.view(N, nr_boxes, -1)
        else:
            return class_logits.view(N, nr_boxes, -1), pred_bboxes, obj_features
    

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.ISTR.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.ISTR.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.ISTR.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ELU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)


        features = torch.bmm(features, param1)

        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)

        features = self.norm2(features)
        features_roi = self.activation(features)

        features = features_roi.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
