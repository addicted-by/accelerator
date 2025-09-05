import math
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist


def dist_reduce(x):
    try:
        dist.all_reduce(x)
    except Exception as e:
        print(e)


def standardize(x, num_instance):
    mean = torch.sum(x, dim=0, keepdim=True) / num_instance
    dist_reduce(mean) # sum all process's mean, that is global mean. read by rflai
    var = (x - mean) ** 2
    var = torch.sum(var, dim=0, keepdim=True) / num_instance
    dist_reduce(var) # sum all process's var, that is global var. read by rflai
    return (x - mean) / torch.sqrt(var)


def vam(x: np):
    num_instance, num_models = x.shape
    new_x = np.zeros_like(x)
    num_per_model = math.ceil(num_instance / num_models)
    candidate_cols = {*range(num_models)}
    unallocated_rows = {*range(num_instance)}
    label = np.zeros([num_instance])
    allocated_num = np.zeros([num_models])
    update_flag = True
    while len(unallocated_rows) > 0:
        if len(candidate_cols) == 1:
            label[[*unallocated_rows]] = [*candidate_cols][0]
            break
        if update_flag:
            for col in candidate_cols:
                other_cols = candidate_cols - {col}
                new_x[[*unallocated_rows], col] = x[[*unallocated_rows], col] - np.max(
                    x[[*unallocated_rows], :][:, [*other_cols]], axis=1)
            update_flag = False
        ind = np.unravel_index(np.argmax(new_x), new_x.shape)
        unallocated_rows.remove(ind[0])
        label[ind[0]] = ind[1]
        allocated_num[ind[1]] += 1
        new_x[ind[0], :] = -1
        if allocated_num[ind[1]] >= num_per_model:
            new_x[:, ind[1]] = -1
            candidate_cols.remove(ind[1])
            update_flag = True
    return label


def wgm(selection_logits: torch.tensor, start_ind: int, num_instance: int, alpha: float):
    batch_size, num_models = selection_logits.shape

    selection_logits = selection_logits.clone().detach()
    prob_s = torch.softmax(selection_logits, dim=1)

    global_probs = torch.zeros([num_instance, num_models]).cuda()
    global_probs[start_ind:start_ind + batch_size, :] = prob_s
    dist_reduce(global_probs)

    global_probs = np.array(global_probs.cpu())

    global_alloc = vam(global_probs)

    alloc = global_alloc[start_ind:start_ind + batch_size]
    alloc = torch.from_numpy(alloc).long().cuda()

    one_hot = torch.zeros_like(prob_s).cuda()
    one_hot.scatter_(1, alloc.view([batch_size, -1]), 1.0)
    uni_weight = 1 / num_models
    alloc_soft = alpha * one_hot + (1 - alpha) * uni_weight

    norm = num_instance / num_models
    loss_weight = alloc_soft / norm
    return loss_weight


def lgm(tcp: torch.tensor, start_ind: int, num_instance: int):
    batch_size, num_models = tcp.shape
    global_tcp = torch.zeros([num_instance, num_models]).cuda() # [n,c], n=gpu_nums, c=expert_nums; num_instance=gpu_nums*batch
    global_tcp[start_ind:start_ind + batch_size, :] = tcp
    dist_reduce(global_tcp) # sum all process's tcp, to get glocal tcp. read by rflai
    global_tcp = np.array(global_tcp.cpu())
    global_selection_label = vam(global_tcp)
    selection_label = global_selection_label[start_ind:start_ind + batch_size]
    selection_label = torch.from_numpy(selection_label).long().cuda()
    return selection_label


def vcm(tcp: torch.tensor):
    batch_size, num_models = tcp.shape
    mean_row = torch.sum(tcp, dim=1, keepdim=True) / num_models
    var_row = (tcp - mean_row) ** 2
    var_row = torch.sum(var_row, dim=1) / num_models
    dev_row = torch.sqrt(var_row)

    dev_row_sum = torch.sum(dev_row)
    dist_reduce(dev_row_sum)
    return dev_row / dev_row_sum


class Conv2d(nn.Module):
    def __init__(self, in_channel, ou_channel, basis_num=None, conv=nn.Conv2d, **kwargs):
        super(Conv2d, self).__init__()
        self.basis_num = basis_num
        if basis_num is None:
            self.conv = conv(in_channel, ou_channel, **kwargs)
        else:
            self.coe = None
            self.conv = conv(in_channel, ou_channel * self.basis_num, **kwargs)

    def set_coe(self, coe):
        self.coe = coe

    def forward(self, x):
        if self.basis_num is None:
            return self.conv(x)
        else:
            x = self.conv(x)
            b, c, h, w = x.shape
            x = x.view([b, -1, self.basis_num, h, w])
            x = x * self.coe[:, None, :, None, None]
            x = torch.sum(x, dim=2)
            return x


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channel, ou_channel, basis_num=None, kernel_size=2, stride=2, **kwargs):
        super(ConvTranspose2d, self).__init__()
        self.basis_num = basis_num
        if basis_num is None:
            self.dconv = nn.ConvTranspose2d(in_channel, ou_channel, kernel_size=kernel_size, stride=stride, **kwargs)
        else:
            self.coe = None
            self.dconv = nn.ConvTranspose2d(
                in_channel, ou_channel * basis_num, kernel_size=kernel_size, stride=stride, **kwargs
            )

    def set_coe(self, coe):
        self.coe = coe

    def forward(self, x):
        if self.basis_num is None:
            return self.dconv(x)
        else:
            x = self.dconv(x)
            b, c, h, w = x.shape
            x = x.view([b, -1, self.basis_num, h, w])
            x = x * self.coe[:, None, :, None, None]
            x = torch.sum(x, dim=2)
            return x


class CoePredictor(nn.Module):
    def __init__(self, in_chs, reduced_chs, out_chs, act_layer=nn.ReLU, basis_num=4, normalize=True):
        super(CoePredictor, self).__init__()
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, out_chs, 1, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.basis_num = basis_num
        self.normalize = normalize

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv_reduce(x)
        x = self.act1(x)
        x = self.conv_expand(x)
        if self.normalize:
            x = x.view([batch_size, -1, self.basis_num])
            x = self.softmax(x)
            x = x.view([batch_size, -1])
        return x


class ExpertSelector(nn.Module):
    def __init__(self, in_chs, reduced_chs, out_chs,
                 act_layer=nn.ReLU):
        super(ExpertSelector, self).__init__()
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, out_chs, 1, bias=True)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv_reduce(x)
        x = self.act1(x)
        x = self.conv_expand(x)
        x = x.flatten(1)
        return x