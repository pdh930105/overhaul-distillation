import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math

# 수식 (4), (5)의 공식을 연산하는 위한 함수
# traget = torch.max(target, margin)의 경우 수식 (2)의 margin_ReLU를 구현화한 것 (해당 연산의 결과가 수식 5의 σ(Ft) 를 의미)
# torch.nn.functional.mse_loss(source, target, reduction="none") 는 soruce와 target의 l2 distance를 계산하는 식과 동일

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

# student_net이 feature extraction 하기 위한 1x1 conv를 만들기 위한 함수
# (s_channel, t_channel, 1, 1)인 1x1 conv를 만드는 것으로 student의 activation feature가 teach의 activation feature와 동일한 shape를 가지게 함.
# 이 때 전제조건으로 둘의 activation map의 가로 세로는 동일해야 한다.
def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

# BN의 통계를 가지고서 margin을 계산하기 위한 함수
# 통계학적 지식 필요 해당 연산 결과 수식(3)의 margin이 나온다는 것 만 이해
def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()
        
        # model의 get_channel_num() 함수 참고
        # 해당 github 예제에선 teacher와 student 모두 3개의 feature distillation 사용
        # 해당하는 3개의 channel num을 가지고 오는 함수
        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        # 3개의 feature 들을 연결하기 위한 connector 선언
        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        
        # teacher_net의 margin을 미리 계산
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

    # forward를 통한 학습 준비
    # train_with_distill function에서 s_out을 가지고서 cross_entropy를 계산하고, loss_distill을 가지고 distillation_loss 를 구함.
    def forward(self, x):
        
        # models 안 의 extract_feature 확인
        # extract_feature를 통해 teacher_net의 feature 3개와 output, student의 feature 3개와 output 을 가지고 옮.
        t_feats, t_out = self.t_net.extract_feature(x, preReLU=True)
        s_feats, s_out = self.s_net.extract_feature(x, preReLU=True)
        feat_num = len(t_feats)

        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (feat_num - i - 1)

        return s_out, loss_distill
