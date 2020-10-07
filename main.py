import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch_optimizer as optim

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from time import gmtime, strftime
import argparse

from IPython.display import clear_output
import matplotlib.pyplot as plt

from utils.env import FrameEnv
from utils.plot import Plotter
from model import Critic, DiscreteActor, Beta, Reinforce
from reinforce import ChooseREINFORCE

cuda = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
parser.add_argument("--frame_size", type=int, default=10, help="frame size for training")
parser.add_argument("--batch_size", type=int, default=10, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim", type=int, default=32, help="dimension of embedding")
parser.add_argument("--policy_input_dim", type=int, default=1024, help="input dimension for policy/value net")
parser.add_argument("--policy_hidden_dim", type=int, default=4096, help="hidden dimension for policy/value net")
parser.add_argument("--data_path", type=str, default="/Users/JingboLiu/Desktop/nicf-pytorch/data/ml-1m")
parser.add_argument("--model_path", type=str, default="/Users/JingboLiu/Desktop/nicf-pytorch/models")
parser.add_argument("--plot_every", type=int, default=100, help="how many steps to plot the result")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
args = parser.parse_args()

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.embedding_path = os.path.join(args.data_path, 'ml20_pca128.pkl')
args.rating_path = os.path.join(args.data_path, 'ratings.csv')

cudnn.benchmark = True


if __name__=='__main__':

    env = FrameEnv(embedding_path, rating_path, num_items, frame_size=10, 
                               batch_size=25, num_workers=1, test_size=0.05)

    beta_net   = Beta().to(cuda)
    value_net  = Critic(args.policy_input_dim, args.policy_hidden_dim, num_items, args.dropout).to(cuda)
    policy_net = DiscreteActor(args.policy_input_dim, args.policy_hidden_dim, num_items, ).to(cuda)

    policy_net.action_source = {'pi': 'beta', 'beta': 'beta'}

    reinforce = Reinforce(policy_net, value_net)
    reinforce = reinforce.to(cuda)

    # reinforce.writer = SummaryWriter()
    plotter = Plotter(reinforce.loss_layout, [['value', 'policy']],)

    def select_action_corr(state, action, K, writer, step, **kwargs):
        # note here I provide beta_net forward in the arguments
        return reinforce.nets['policy_net']._select_action_with_TopK_correction(state, beta_net.forward, action,
                                                                                K=K, writer=writer,
                                                                                step=step)

    reinforce.nets['policy_net'].select_action = select_action_corr
    reinforce.params['reinforce'] = ChooseREINFORCE(ChooseREINFORCE.reinforce_with_TopK_correction)
    reinforce.params['K'] = 10

    
    for epoch in range(n_epochs):
        for batch in tqdm(env.train_dataloader):
            loss = reinforce.update(batch)
            reinforce.step()
            if loss:
                plotter.log_losses(loss)
            if reinforce._step % args.plot_every == 0:
                clear_output(True)
                print('step', reinforce._step)
                plotter.plot_loss()
            if reinforce._step > 1000:
                pass