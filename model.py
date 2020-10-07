import torch
import torch.nn as nn
import torch_optimizer as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from reinforce import ChooseREINFORCE, reinforce_update

class Beta(nn.Module):
    def __init__(self):
        super(Beta, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, num_items),
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

    def __init__(self, input_dim, hidden_size, action_dim, dropout, init_w):
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

    def to(self, device):
        self.nets = {k: v.to(device) for k, v in self.nets.items()}
        self.device = device
        return self

    def step(self):
        self._step += 1


class Reinforce:
    def __init__(self, policy_net, value_net):

        super(Reinforce, self).__init__()

        self.algorithm = reinforce_update

        # define optimizers
        value_optimizer = optim.Ranger(
            value_net.parameters(), lr=1e-5, weight_decay=1e-2
        )
        policy_optimizer = optim.Ranger(
            policy_net.parameters(), lr=1e-5, weight_decay=1e-2
        )

        self.nets = {
            "value_net": value_net,
            "policy_net": policy_net,
        }

        self.optimizers = {
            "policy_optimizer": policy_optimizer,
            "value_optimizer": value_optimizer,
        }

        self.params = {
            "reinforce": ChooseREINFORCE(ChooseREINFORCE.reinforce_with_TopK_correction),
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

    def update(self, batch, learn=True):
        return reinforce_update(
            batch,
            self.params,
            self.nets,
            self.optimizers,
            device=self.device,
            debug=self.debug,
            writer=self.writer,
            learn=learn,
            step=self._step
        )