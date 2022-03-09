# coding:utf-8

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from MaskLoader import MaskLoader
from utils import (
    IOUMetric,
    transform,
    inverse_transform,
    direct_sigmoid,
    inverse_sigmoid
)
from PIL import Image

VALUE_MAX = 0.05
VALUE_MIN = 0.01
dataset_root = '/home/hujie/hujie-project/SparseR-CNN/datasets/'
parser = argparse.ArgumentParser(description='AE Mask Embedding')
parser.add_argument('--root', default='datasets', type=str)
parser.add_argument('--dataset_train', default='coco_2017_train', type=str)
parser.add_argument('--dataset_val', default='coco_2017_val', type=str)
parser.add_argument('--epoch', default=30, type=int)
# mask encoding params.
parser.add_argument('--embedding_size', default=64, type=int)
parser.add_argument('--mask_size', default=112, type=int)
parser.add_argument('--batch-size', default=1024, type=int)
args = parser.parse_args()


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, size, scale_factor):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size, scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# ----------- 28_256 ------------------
# class Encoder(nn.Module):
#     def __init__(self, mask_size, embedding_size):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, 4, 2, 1), # 14
#             nn.BatchNorm2d(64),
#             nn.ELU(True),
#             nn.Conv2d(64, 128, 4, 2, 1), # 7
#             nn.BatchNorm2d(128),
#             nn.ELU(True),
#             nn.Conv2d(128, embedding_size, 7, 1),
#             View((-1, embedding_size*1*1)),
#         )
#     def forward(self, x):
#         f = self.encoder(x)
#         return f

# class Decoder(nn.Module):
#     def __init__(self, mask_size, embedding_size):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             View((-1, embedding_size, 1, 1)),
#             nn.ConvTranspose2d(embedding_size, 128, 7, 1),
#             nn.BatchNorm2d(128),
#             nn.ELU(inplace=True),
#             up_conv(128, 64, None, 2),
#             up_conv(64, 32, None, 2),
#             nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),
#             View((-1, 1, mask_size, mask_size)),
#         )

#     def forward(self, x):
#         # for layer in self.decoder:
#         #     x = layer(x)
#         x = self.decoder(x)
#         return x

# ----------- 56_256 ------------
# class Encoder(nn.Module):
#     def __init__(self, mask_size, embedding_size):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(

#             nn.Conv2d(1, 32, 4, 2, 1), # 28
#             nn.BatchNorm2d(32),
#             nn.ELU(True),
#             nn.Conv2d(32, 64, 4, 2, 1), # 14
#             nn.BatchNorm2d(64),
#             nn.ELU(True),
#             nn.Conv2d(64, 128, 4, 2, 1), # 7
#             nn.BatchNorm2d(128),
#             nn.ELU(True),

#             nn.Conv2d(128, embedding_size, 7, 1),
#             View((-1, embedding_size*1*1)),
#         )
    
#     def forward(self, x):
#         f = self.encoder(x)
#         return f

# class Decoder(nn.Module):
#     def __init__(self, mask_size, embedding_size):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             View((-1, embedding_size, 1, 1)),
#             nn.ConvTranspose2d(embedding_size, 128, 7, 1),
#             nn.BatchNorm2d(128),
#             nn.ELU(inplace=True),
#             up_conv(128, 64, None, 2),
#             up_conv(64, 32, None, 2),
#             up_conv(32, 16, None, 2),
#             nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),
#             View((-1, 1, mask_size, mask_size)),
#         )

#     def forward(self, x, roi_feat=None):
#         # for i, layer in enumerate(self.decoder):
#         #     if i == 6 and roi_feat != None:
#         #         shape = x.shape
#         #         roi_feat = roi_feat.view(shape[0], shape[1], shape[2], shape[3])
#         #         x = x + roi_feat
#         #         del roi_feat
#         #     x = layer(x)
#         x = self.decoder(x)
#         return x

# ----------- 224_256 ------------
# class Encoder(nn.Module):
#     def __init__(self, mask_size, embedding_size):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, 4, 2, 1), # 112
#             nn.BatchNorm2d(32),
#             nn.ELU(True),
#             nn.Conv2d(32, 32, 4, 2, 1), # 56
#             nn.BatchNorm2d(32),
#             nn.ELU(True),
#             nn.Conv2d(32, 32, 4, 2, 1), # 28
#             nn.BatchNorm2d(32),
#             nn.ELU(True),
#             nn.Conv2d(32, 64, 4, 2, 1), # 14
#             nn.BatchNorm2d(64),
#             nn.ELU(True),
#             nn.Conv2d(64, 128, 4, 2, 1), # 7
#             nn.BatchNorm2d(128),
#             nn.ELU(True),

#             nn.Conv2d(128, embedding_size, 7, 1),
#             View((-1, embedding_size*1*1)),
#         )
    
#     def forward(self, x):
#         f = self.encoder(x)
#         return f

# class Decoder(nn.Module):
#     def __init__(self, mask_size, embedding_size):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             View((-1, embedding_size, 1, 1)),
#             nn.ConvTranspose2d(embedding_size, 128, 7, 1),
#             nn.BatchNorm2d(128),
#             nn.ELU(inplace=True),
#             up_conv(128, 64, None, 2),
#             up_conv(64, 32, None, 2),
#             up_conv(32, 16, None, 2),
#             up_conv(16, 16, None, 2),
#             up_conv(16, 16, None, 2),
#             nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),
#             View((-1, 1, mask_size, mask_size)),
#         )

#     def forward(self, x, roi_feat=None):
#         # for i, layer in enumerate(self.decoder):
#         #     if i == 6 and roi_feat != None:
#         #         shape = x.shape
#         #         roi_feat = roi_feat.view(shape[0], shape[1], shape[2], shape[3])
#         #         x = x + roi_feat
#         #         del roi_feat
#         #     x = layer(x)
#         x = self.decoder(x)
#         return x

# ----------- 112_64/128/512 ------------
class Encoder(nn.Module):
    def __init__(self, mask_size, embedding_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),  # 56
            nn.BatchNorm2d(16),
            nn.ELU(True),
            nn.Conv2d(16, 32, 4, 2, 1), # 28
            nn.BatchNorm2d(32),
            nn.ELU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # 14
            nn.BatchNorm2d(64),
            nn.ELU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # 7
            nn.BatchNorm2d(128),
            nn.ELU(True),

            nn.Conv2d(128, embedding_size, 7, 1),
            View((-1, embedding_size*1*1)),
        )

    def forward(self, x):
        f = self.encoder(x)
        return f

class Decoder(nn.Module):
    def __init__(self, mask_size, embedding_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            View((-1, embedding_size, 1, 1)),
            nn.ConvTranspose2d(embedding_size, 128, 7, 1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),

            up_conv(128, 64, None, 2),

            up_conv(64, 32, None, 2),

            up_conv(32, 16, None, 2),

            up_conv(16, 16, None, 2),

            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),

            View((-1, 1, mask_size, mask_size)),
        )

    def forward(self, x, roi_feat=None):
        # for i, layer in enumerate(self.decoder):
        #     if i == 6 and roi_feat != None:
        #         shape = x.shape
        #         roi_feat = roi_feat.view(shape[0], shape[1], shape[2], shape[3])
        #         x = x + roi_feat
        #         del roi_feat
        #     x = layer(x)
        x = self.decoder(x)
        return x

loss_MSE = nn.MSELoss()
E = Encoder(args.mask_size, args.embedding_size)
D = Decoder(args.mask_size, args.embedding_size)
E.cuda()
D.cuda()
optimizer = optim.Adam([{'params' : E.parameters()},
                        {'params' : D.parameters()}], lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10, 20], gamma=0.1)

mask_train_data = MaskLoader(root=dataset_root, dataset=args.dataset_train, size=args.mask_size)
mask_train_loader = DataLoader(mask_train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

mask_val_data = MaskLoader(root=dataset_root, dataset=args.dataset_val, size=args.mask_size)
mask_val_loader = DataLoader(mask_val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

best_iou = 0.0

def visualization(epoch, a, b):
    save_dir = os.path.join('./vis')
    os.makedirs(save_dir, exist_ok=True)
    res_mask = []

    for idx, (a_mask, b_mask) in enumerate(zip(a, b)):
        
        if idx > 0:
            break

        a_mask = a_mask.squeeze()
        a_mask = a_mask.detach().cpu().numpy() >= 0.5

        b_mask = b_mask.squeeze() # h,w
        b_mask = b_mask.detach().cpu().numpy() >= 0.5

        
        img = a_mask #np.concatenate((a_mask, b_mask), axis=1)    
        res_mask.append(img)
        
    res_mask = np.concatenate(res_mask, axis=0)
    
    img = Image.fromarray(res_mask)
    
    img.save(os.path.join(save_dir, 'vis_{}.png'.format(epoch)))

    print(adfs)



# Train
def train(epoch):
    E.train()
    D.train()
    for batch_idx, x in enumerate(mask_train_loader):
        x = x.cuda()
        f = E(x)
        x_rec = D(f)

        # print(f.shape, x_rec.shape)

        # loss_rec = loss_MSE(x_rec, x)

        eps = 1e-5
        n_inst = x.size(0)
        x_rec = x_rec.reshape(n_inst, -1)
        x = x.reshape(n_inst, -1)
        intersection = (x_rec * x).sum(dim=1)
        union = (x_rec ** 2.0).sum(dim=1) + (x ** 2.0).sum(dim=1) + eps
        loss_rec = 1. - (2 * intersection / union)
        loss_rec = loss_rec.mean()

        optimizer.zero_grad()
        loss_rec.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss_rec: {:.5f}'.format(
                    epoch, batch_idx * len(x), len(mask_train_loader.dataset),
                    100. * batch_idx / len(mask_train_loader),
                    loss_rec.item()))


# Test
def test(epoch = 0):
    global best_iou
    E.eval()
    D.eval()
    test_loss = 0
    IoUevaluate = IOUMetric(2)
    print("Start evaluation...")
    with torch.no_grad():
        for i, x in enumerate(mask_val_loader):
            x = x.cuda()
            f = E(x)
            x_rec = D(f)

            visualization(epoch, x, x_rec)

            x_rec = np.where(x_rec.cpu().numpy() >= 0.5, 1, 0)
            IoUevaluate.add_batch( x_rec, x.cpu().numpy() )

        _, _, _, mean_iou, _ = IoUevaluate.evaluate()
        print("The mIoU: {}".format(mean_iou))

    if mean_iou > best_iou:
        print('Best...')
        best_iou = mean_iou

        state = {
            'E': E.state_dict(),
            'D': D.state_dict(),
            'best_iou': mean_iou,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, './checkpoints/AE_{}_{}.t7'.format(args.mask_size, args.embedding_size))
        


if __name__ == "__main__":
    # build data loader.
    test()
    for epoch in range(args.epoch):
        train(epoch)
        test(epoch)
        scheduler.step()


