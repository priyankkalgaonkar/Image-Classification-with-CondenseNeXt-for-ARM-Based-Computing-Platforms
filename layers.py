from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#Technicially removed from architecture but keeping it for reference only
class LearnedGroupConv(nn.Module):
    global_progress = 0.0
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, 
                 condense_factor=None, dropout_rate=0.):
        super(LearnedGroupConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor is None:
            self.condense_factor = self.groups
        ### Parameters that should be carefully used
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, "group number can not be divided by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor can not be divided by input channels"
        assert self.out_channels % self.groups == 0, "group number can not be divided by output channels"

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, None, self.conv.stride,
                        self.conv.padding, self.conv.dilation, 1)

    def _check_drop(self):
        progress = LearnedGroupConv.global_progress
        delta = 0
        ### Get current stage
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        ### Check for dropping
        if not self._reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
        if delta > 0:
            self._dropping(delta)
        return

    def _dropping(self, delta):
        weight = self.conv.weight * self.mask
        ### Sum up all kernels
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.transpose(0, 1).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            ### Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])
        
    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._reach_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.sum(0).clamp(min=1e-6).sqrt()
        return weight.sum()
#Deprecated code ends here

#My Depthwise Sep. + Groupwise Pruning Code Start
class PK_Dw_Conv(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, fiter_kernel, stride, padding, dropout_rate, k,cardinality=8):
        super(PK_Dw_Conv, self).__init__()

        self.dropout_rate = dropout_rate
        self.cardinality = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.group=self.out_channels//self.cardinality
        self.in_channel_per_group=in_channels//self.group
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu6 = nn.ReLU6(inplace=True)
        self.dwconv = nn.Conv2d(
			in_channels, 
			out_channels, 
			fiter_kernel, 
			stride, 
			padding, 
			bias=False,
			groups=self.group
        )

        self.group_pruning = (int)(self.in_channel_per_group*(self.cardinality-k)*0.125)
        self.tempo = self.group*(self.in_channel_per_group*self.cardinality- 8*self.group_pruning)
        
        self.pwconv = nn.Conv2d(
			self.tempo, 
			out_channels, 
			1, 
			1, 
			bias=False
        )

        self.dwconv2 = nn.Conv2d(
			self.tempo, 
			self.tempo, 
			fiter_kernel, 
			stride, 
			padding, 
			groups=self.tempo, 
			bias=False
        )
        
        self.register_buffer('index', torch.LongTensor(self.tempo))
        self.register_buffer('_mask_dw', torch.ones(self.dwconv.weight.size()))
        self.register_buffer('_count', torch.zeros(1))

    def _check_drop(self):
        progress = PK_Dw_Conv.global_progress
        if progress == 0:
            self.dwconv2.weight.data.zero_()
            self.pwconv.weight.data.zero_()
        if progress<300 :
            self.dwconv2.weight.data.zero_()
            self.pwconv.weight.data.zero_()
        if progress>300 :
            self.dwconv.weight.data.zero_()
            ### Check for dropping
        if progress == 37 or progress == 75 or progress == 112 or progress == 150 or progress == 187 or progress == 225 or progress == 262 or progress == 300:
            self._dropping_group(self.group_pruning)
        return
        
    def _dropping_group(self,delta):
        if PK_Dw_Conv.global_progress <= 300:
            weight=self.dwconv.weight*self.mask_dw
            weight=weight.view(self.group,self.cardinality,self.in_channel_per_group,3,3).abs().sum([3,4])
            for i in range(self.group):
                weight_tempo=weight[i,:,:].view(-1)
                di=weight_tempo.sort()[1][self.count:self.count+delta]
                for d in di.data:
                    out_ = d // self.in_channel_per_group
                    in_ = d % self.in_channel_per_group
                    self._mask_dw[i*self.cardinality+out_, in_, :, :].fill_(0)
            self.count = self.count + delta
        index=0
        if PK_Dw_Conv.global_progress == 300:
            self.pwconv.weight.data.zero_()
            for i in range(self.group):
                for j in range(self.cardinality):
                    for k in range(self.in_channel_per_group):
                        if self._mask_dw[i*self.cardinality+j,k,0,0]==1:
                            self.index[index]=i*self.in_channel_per_group+k
                            self.dwconv2.weight.data[index,:,:,:]=self.dwconv.weight.data[i*self.cardinality+j,k,:,:].view(1,3,3)
                            self.pwconv.weight.data[i*self.cardinality+j,index,:,:].fill_(1)
                            index=index+1
            assert index==self.tempo
            self.dwconv.weight.data.zero_()
    def forward(self, x):
        progress = PK_Dw_Conv.global_progress
        self._check_drop()
        if self.dropout_rate > 0:
            x = self.drop(x)
        
        ### Masked output

        if progress < 300:
            weight = self.dwconv.weight * self.mask_dw
            return F.conv2d(x,weight, None, self.dwconv.stride,
                            1, self.dwconv.dilation, self.group)
        else:
            x = torch.index_select(x, 1, Variable(self.index))
            x = self.dwconv2(x)
            self.pwconv.weight.data = self.pwconv.weight.data  # *self.mask_pw
            x = F.conv2d(x, self.pwconv.weight, None, self.pwconv.stride,
                         0, self.pwconv.dilation, 1)
            return x

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def mask_dw(self):
        return Variable(self._mask_dw)

    @property
    def mask_pw(self):
        return Variable(self._mask_pw)
    @property
    def pk_dw_loss(self):
        return 0
        if PK_Dw_Conv.global_progress >= 300:
            return 0
        weight = self.dwconv.weight * self.mask_dw
        weight_1=weight.abs().sum(-1).sum(-1).view(self.group,self.cardinality,self.in_channel_per_group)
        weight=weight.abs().sum([2,3]).view(self.group,-1)
        mask=torch.ge(weight,torch.topk(weight,self.k*self.in_channel_per_group,1,sorted=True)[0][:,self.k*self.in_channel_per_group-1]
                      .view(self.group,1).expand_as(weight)).view(self.group,self.cardinality,self.in_channel_per_group)\
            .sum(1).view(self.group,1,self.in_channel_per_group)
        mask = torch.exp((mask.float() - 1.5 * self.k) / (10)) - 1
        mask=mask.expand_as(weight_1)
        weight=(weight_1.pow(2)*mask).sum(1).clamp(min=1e-6).sum(-1).sum(-1)
        return weight
#My Depthwise Sep. + Groupwise Pruning Code End


def ShuffleLayer(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    ### reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    ### transpose
    x = torch.transpose(x, 1, 2).contiguous()
    ### flatten
    x = x.view(batchsize, -1, height, width)
    return x


class CondensingLinear(nn.Module):
    def __init__(self, model, drop_rate=0.5):
        super(CondensingLinear, self).__init__()
        self.in_features = int(model.in_features*drop_rate)
        self.out_features = model.out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.register_buffer('index', torch.LongTensor(self.in_features))
        _, index = model.weight.data.abs().sum(0).sort()
        index = index[model.in_features-self.in_features:]
        self.linear.bias.data = model.bias.data.clone()
        for i in range(self.in_features):
            self.index[i] = index[i]
            self.linear.weight.data[:, i] = model.weight.data[:, index[i]]

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.linear(x)
        return x


class CondensingConv(nn.Module):
    def __init__(self, model):
        super(CondensingConv, self).__init__()
        self.in_channels = model.conv.in_channels \
                         * model.groups // model.condense_factor
        self.out_channels = model.conv.out_channels
        self.groups = model.groups
        self.condense_factor = model.condense_factor
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=model.conv.kernel_size,
                              padding=model.conv.padding,
                              groups=self.groups,
                              bias=False,
                              stride=model.conv.stride)
        self.register_buffer('index', torch.LongTensor(self.in_channels))
        index = 0
        mask = model._mask.mean(-1).mean(-1)
        for i in range(self.groups):
            for j in range(model.conv.in_channels):
                if index < (self.in_channels // self.groups) * (i + 1) \
                         and mask[i, j] == 1:
                    for k in range(self.out_channels // self.groups):
                        idx_i = int(k + i * (self.out_channels // self.groups))
                        idx_j = index % (self.in_channels // self.groups)
                        self.conv.weight.data[idx_i, idx_j, :, :] = \
                            model.conv.weight.data[int(i + k * self.groups), j, :, :]
                        self.norm.weight.data[index] = model.norm.weight.data[j]
                        self.norm.bias.data[index] = model.norm.bias.data[j]
                        self.norm.running_mean[index] = model.norm.running_mean[j]
                        self.norm.running_var[index] = model.norm.running_var[j]
                    self.index[index] = j
                    index += 1

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class CondenseLinear(nn.Module):
    def __init__(self, in_features, out_features, drop_rate=0.5):
        super(CondenseLinear, self).__init__()
        self.in_features = int(in_features*drop_rate)
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.register_buffer('index', torch.LongTensor(self.in_features))

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.linear(x)
        return x


class CondenseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, groups=1):
        super(CondenseConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=self.groups,
                              bias=False)
        self.register_buffer('index', torch.LongTensor(self.in_channels))
        self.index.fill_(0)

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu6', nn.ReLU6(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))
