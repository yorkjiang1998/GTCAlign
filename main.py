# 作者：York
# 时间：2022/5/30 18:22
import itertools
from torch import optim
import json
import test
from config import Config
import argparse
from refine import *
import os
from model import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from time import time

def get_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('-d', "--dataset", type=str, default="douban")
    parse.add_argument('-a', "--alpha", type=float, default=0.01)
    parse.add_argument('-k', "--gcn_block", type=int, default=2)
    parse.add_argument('-out', "--output_dim", type=int, default="128")
    parse.add_argument('-r_e', "--r_epochs", type=int, default=80)
    parse.add_argument("--top_k", type=int, default=5)
    parse.add_argument("--alphas", type=list, default=[1, 1, 1])
    parse.add_argument("--stable_xs", type=list, default=[1.1, 1.0001])
    parse.add_argument("--topo_diff", type=list, default=[1.0, 1.006])
    return parse.parse_args()


def run():
    args = get_parse()
    config_file = './config/' + args.dataset + '.ini'
    config = Config(config_file)
    cuda = not config.no_cuda and torch.cuda.is_available()
    use_seed = not config.no_seed

    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    #———————————————————————————————————————— Dataset Loading
    print('Dataset Loading')
    config.name = 's'
    config.n = config.n1
    s_sadj = load_graph(config)
    s_feature = load_data(config)

    config.name = 't'
    config.n = config.n2
    t_sadj = load_graph(config)
    t_feature = load_data(config)
    print('Dataset end')
    #————————————————————————————————————————— Model
    model = GTCAlign(args.gcn_block, args.output_dim, s_feature, t_feature, 'tanh', config)
    s_topo, t_topo = get_topological_all(args.dataset)
    refine_alignment(model, s_sadj, t_sadj, args,s_topo, t_topo)


if __name__ == '__main__':
    run()


