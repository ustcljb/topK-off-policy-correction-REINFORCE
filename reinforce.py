import torch

from recnn import utils
from recnn import data
import gc


def temporal_difference(reward, done, gamma, target):
    return reward + (1.0 - done) * gamma * target


def value_update(
    batch,
    params,
    nets,
    optimizer,
    device=torch.device("cpu"),
    debug=None,
    writer=utils.DummyWriter(),
    learn=False,
    step=-1,
):
    """
    Everything is the same as in ddpg_update
    """

    state, action, reward, next_state, done = data.get_base_batch(batch, device=device)

    with torch.no_grad():
        next_action = nets["target_policy_net"](next_state)
        target_value = nets["target_value_net"](next_state, next_action.detach())
        expected_value = temporal_difference(
            reward, done, params["gamma"], target_value
        )
        expected_value = torch.clamp(
            expected_value, params["min_value"], params["max_value"]
        )

    value = nets["value_net"](state, action)

    value_loss = torch.pow(value - expected_value.detach(), 2).mean()

    if learn:
        optimizer["value_optimizer"].zero_grad()
        value_loss.backward(retain_graph=True)
        optimizer["value_optimizer"].step()

    elif not learn:
        debug["next_action"] = next_action
        writer.add_figure(
            "next_action", utils.pairwise_distances_fig(next_action[:50]), step
        )
        writer.add_histogram("value", value, step)
        writer.add_histogram("target_value", target_value, step)
        writer.add_histogram("expected_value", expected_value, step)

    return value_loss

    
class ChooseREINFORCE:
    def __init__(self, method=None):
        if method is None:
            method = ChooseREINFORCE.basic_reinforce
        self.method = method

    @staticmethod
    def reinforce_with_TopK_correction(policy, returns, *args, **kwargs):
        policy_loss = []
        for l_k, corr, log_prob, R in zip(
            policy.lambda_k, policy.correction, policy.saved_log_probs, returns
        ):
            policy_loss.append(l_k * corr * -log_prob * R)  # <- this line here
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss

    def __call__(self, policy, optimizer, learn=True):
        R = 0

        returns = []
        for r in policy.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 0.0001)

        policy_loss = self.method(policy, returns)

        if learn:
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        policy.gc()
        gc.collect()

        return policy_loss

def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


def reinforce_update(
    batch,
    params,
    nets,
    optimizer,
    device=torch.device("cpu"),
    debug=None,
    writer=utils.DummyWriter(),
    learn=True,
    step=-1,
):

    # Due to its mechanics, reinforce doesn't support testing!
    learn = True

    state, action, reward, next_state, done = data.get_base_batch(batch)

    predicted_probs = nets["policy_net"].select_action(
        state=state, action=action, K=params["K"], learn=learn, writer=writer, step=step
    )
    writer.add_histogram("predicted_probs_std", predicted_probs.std(), step)
    writer.add_histogram("predicted_probs_mean", predicted_probs.mean(), step)
    mx = predicted_probs.max(dim=1).values
    writer.add_histogram("predicted_probs_max_mean", mx.mean(), step)
    writer.add_histogram("predicted_probs_max_std", mx.std(), step)
    reward = nets["value_net"](state, predicted_probs).detach()
    nets["policy_net"].rewards.append(reward.mean())

    value_loss = value_update(
        batch,
        params,
        nets,
        optimizer,
        writer=writer,
        device=device,
        debug=debug,
        learn=True,
        step=step,
    )

    if step % params["policy_step"] == 0 and step > 0:
        policy_loss = params["reinforce"](
            nets["policy_net"],
            optimizer["policy_optimizer"],
        )

        losses = {
            "value": value_loss.item(),
            "policy": policy_loss.item(),
            "step": step,
        }

        utils.write_losses(writer, losses, kind="train" if learn else "test")

        return losses
