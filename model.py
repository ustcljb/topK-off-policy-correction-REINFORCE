import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from reinforce import ChooseREINFORCE

class Beta(nn.Module):
    def __init__(self):
        super(Beta, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1290, num_items),
            nn.Softmax()
        )
        self.optim = optim.RAdam(self.net.parameters(), lr=1e-5, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, state, action):
        
        predicted_action = self.net(state)
        
        loss = self.criterion(predicted_action, action.argmax(1))
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return predicted_action.detach()


class Critic(nn.Module):

    """
    Vanilla critic. Takes state and action as an argument, returns value.
    """

    def __init__(self, input_dim, hidden_size, action_dim, dropout):
        super(Critic, self).__init__()

        self.drop_layer = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(input_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):

        value = torch.cat([state, action], 1)
        value = F.relu(self.linear1(value))
        value = self.drop_layer(value)
        value = F.relu(self.linear2(value))
        value = self.drop_layer(value)
        value = self.linear3(value)
        return value


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size, init_w=0):
        super(DiscreteActor, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_dim)

        self.saved_log_probs = []
        self.rewards = []
        self.correction = []
        self.lambda_k = []

        self.action_source = {"pi": "pi", "beta": "beta"}
        self.select_action = self._select_action

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores)

    def gc(self):
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.correction[:]
        del self.lambda_k[:]

    def _select_action(self, state, **kwargs):

        # for reinforce without correction only pi_probs is available.
        # the action source is ignored, since there is no beta

        pi_probs = self.forward(state)
        pi_categorical = Categorical(pi_probs)
        pi_action = pi_categorical.sample()
        self.saved_log_probs.append(pi_categorical.log_prob(pi_action))
        return pi_probs

    def pi_beta_sample(self, state, beta, action, **kwargs):
        # 1. obtain probabilities
        # note: detach is to block gradient
        beta_probs = beta(state.detach(), action=action)
        pi_probs = self.forward(state)

        # 2. probabilities -> categorical distribution.
        beta_categorical = Categorical(beta_probs)
        pi_categorical = Categorical(pi_probs)

        # 3. sample the actions
        # See this issue: https://github.com/awarebayes/RecNN/issues/7
        # usually it works like:
        # pi_action = pi_categorical.sample(); beta_action = beta_categorical.sample();
        # but changing the action_source to {pi: beta, beta: beta} can be configured to be:
        # pi_action = beta_categorical.sample(); beta_action = beta_categorical.sample();
        available_actions = {
            "pi": pi_categorical.sample(),
            "beta": beta_categorical.sample(),
        }
        pi_action = available_actions[self.action_source["pi"]]
        beta_action = available_actions[self.action_source["beta"]]

        # 4. calculate stuff we need
        pi_log_prob = pi_categorical.log_prob(pi_action)
        beta_log_prob = beta_categorical.log_prob(beta_action)

        return pi_log_prob, beta_log_prob, pi_probs

    def _select_action_with_correction(
        self, state, beta, action, writer, step, **kwargs
    ):
        pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample(state, beta, action)

        # calculate correction
        corr = torch.exp(pi_log_prob) / torch.exp(beta_log_prob)

        writer.add_histogram("correction", corr, step)
        writer.add_histogram("pi_log_prob", pi_log_prob, step)
        writer.add_histogram("beta_log_prob", beta_log_prob, step)

        self.correction.append(corr)
        self.saved_log_probs.append(pi_log_prob)

        return pi_probs

    def _select_action_with_TopK_correction(
        self, state, beta, action, K, writer, step, **kwargs
    ):
        pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample(state, beta, action)

        # calculate correction
        corr = torch.exp(pi_log_prob) / torch.exp(beta_log_prob)

        # calculate top K correction
        l_k = K * (1 - torch.exp(pi_log_prob)) ** (K - 1)

        writer.add_histogram("correction", corr, step)
        writer.add_histogram("l_k", l_k, step)
        writer.add_histogram("pi_log_prob", pi_log_prob, step)
        writer.add_histogram("beta_log_prob", beta_log_prob, step)

        self.correction.append(corr)
        self.lambda_k.append(l_k)
        self.saved_log_probs.append(pi_log_prob)

        return pi_probs

def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


class Algo:
    def __init__(self):
        self.nets = {
            "value_net": None,
            "policy_net": None,
        }

        self.optimizers = {"policy_optimizer": None, "value_optimizer": None}

        self.params = {"Some parameters here": None}

        self._step = 0

        self.debug = {}

        # by default it will not output anything
        # use torch.SummaryWriter instance if you want output
        self.writer = utils.misc.DummyWriter()

        self.device = torch.device("cpu")

        self.loss_layout = {
            "test": {"value": [], "policy": [], "step": []},
            "train": {"value": [], "policy": [], "step": []},
        }

        self.algorithm = None

    def update(self, batch, learn=True):
        return self.algorithm(
            batch,
            self.params,
            self.nets,
            self.optimizers,
            device=self.device,
            debug=self.debug,
            writer=self.writer,
            learn=learn,
            step=self._step,
        )

    def to(self, device):
        self.nets = {k: v.to(device) for k, v in self.nets.items()}
        self.device = device
        return self

    def step(self):
        self._step += 1


class Reinforce(Algo):
    def __init__(self, policy_net, value_net):

        super(Reinforce, self).__init__()

        self.algorithm = update.reinforce_update

        # these are target networks that we need for ddpg algorigm to work
        target_policy_net = copy.deepcopy(policy_net)
        target_value_net = copy.deepcopy(value_net)

        target_policy_net.eval()
        target_value_net.eval()

        # soft update
        soft_update(value_net, target_value_net, soft_tau=1.0)
        soft_update(policy_net, target_policy_net, soft_tau=1.0)


        # define optimizers
        value_optimizer = optim.Ranger(
            value_net.parameters(), lr=1e-5, weight_decay=1e-2
        )
        policy_optimizer = optim.Ranger(
            policy_net.parameters(), lr=1e-5, weight_decay=1e-2
        )

        self.nets = {
            "value_net": value_net,
            "target_value_net": target_value_net,
            "policy_net": policy_net,
            "target_policy_net": target_policy_net,
        }

        self.optimizers = {
            "policy_optimizer": policy_optimizer,
            "value_optimizer": value_optimizer,
        }

        self.params = {
            "reinforce": ChooseREINFORCE(ChooseREINFORCE.basic_reinforce),
            "K": 10,
            "gamma": 0.99,
            "min_value": -10,
            "max_value": 10,
            "policy_step": 10,
            "soft_tau": 0.001,
        }

        self.loss_layout = {
            "test": {"value": [], "policy": [], "step": []},
            "train": {"value": [], "policy": [], "step": []},
        }