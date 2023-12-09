# 作者：York
# 时间：2022/5/30 18:18
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def get_activate(activate_function):
    if activate_function in ['sigmoid']:
        return nn.Sigmoid()
    elif activate_function in ['tanh']:
        return nn.Tanh()
    elif activate_function in ['relu']:
        return nn.ReLU()
    return None


def init_weight(modules, activation):
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation is None:
                m.weight.data = init.xavier_uniform_(m.weight.data)
            else:
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=init.calculate_gain(activation.lower()))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)


class CGCN(nn.Module):
    def __init__(self, in_put, out_put, activate_fun):
        super(CGCN, self).__init__()
        self.input = in_put
        self.output = out_put
        self.hidden = out_put * 2
        self.activate_function = get_activate(activate_fun)
        self.linear = nn.Linear(self.input, self.output, bias=False)
        init_weight(self.modules(), activate_fun)

    def forward(self, input, adj):
        output = self.linear(input)
        output = torch.matmul(adj, output)
        if self.activate_function is not None:
            output = self.activate_function(output)
        return output


class GTCAlign(nn.Module):
    def __init__(self, GCN_num_blocks, output_dim, s_feature, t_feature, activate_function, config):
        super(GTCAlign, self).__init__()
        self.GCN_num_blocks = GCN_num_blocks
        self.input_dim = s_feature.shape[1]
        input_dim = self.input_dim
        self.output_dim = output_dim
        self.s_feature = s_feature
        self.t_feature = t_feature
        self.activate_function = activate_function
        self.GCNs = []
        for i in range(self.GCN_num_blocks):
            self.GCNs.append(CGCN(input_dim, output_dim, activate_function))
            input_dim = self.GCNs[-1].output
        self.GCNs = nn.ModuleList(self.GCNs)
        init_weight(self.modules(), activate_function)

    def forward(self, adj, net='s'):
        if net in ['s']:
            input = self.s_feature
        elif net in ['t']:
            input = self.t_feature
        emb_input = input.clone()
        outputs = [emb_input]

        for i in range(self.GCN_num_blocks):
            GCN_output_i = self.GCNs[i](emb_input, adj)
            outputs.append(GCN_output_i)
            emb_input = GCN_output_i

        return outputs




class StableFactor(nn.Module):
    """
    Stable factor following each node
    """
    def __init__(self, num_source_nodes, num_target_nodes, cuda=True):
        """
        :param num_source_nodes: Number of nodes in source graph
        :param num_target_nodes: Number of nodes in target graph
        """
        super(StableFactor, self).__init__()
        self.alpha_source = torch.ones(num_source_nodes)
        self.alpha_target = torch.ones(num_target_nodes)
        self.score_max = 0
        self.alpha_source_max = None
        self.alpha_target_max = None
        self.use_cuda = cuda

    def forward(self, A_hat, net='s'):
        if net == 's':
            self.alpha = self.alpha_source
        else:
            self.alpha = self.alpha_target
        alpha_colum = self.alpha.reshape(len(self.alpha), 1)
        if self.use_cuda:
            alpha_colum = alpha_colum.cuda()
        A_hat_new = (alpha_colum * (A_hat * alpha_colum).t()).t()
        return A_hat_new