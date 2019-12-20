# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 15:42
# @Author  : uhauha2929
from torch import nn
import torch
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        @param alpha: 阿尔法α,类别权重
        当α是列表时，为各类别权重，当α为常数时，类别权重为[α, 1-α, 1-α, ....]，retainnet中设置为0.25
        @param gamma: 伽马γ,难易样本调节参数. retainnet中设置为2
        @param num_classes: 类别数量
        @param size_average: 损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        @param preds: 预测类别. size:[B,N,C] or [B,C]
        分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        @param labels: 实际类别. size:[B,N] or [B]
        @return: 损失
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # 取出对应label位置维度的概率值
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        # 取出每一类的权重
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.pow((1 - preds_softmax), self.gamma) * preds_logsoft
        loss = self.alpha * loss.squeeze()
        return loss.mean() if self.size_average else loss.sum()
