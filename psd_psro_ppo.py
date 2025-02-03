import random
import torch

class ModelPPO():
    """
    Parameters
    ----------
    network : CommanderNetwork
    learning_rate : float, optional
    clip_param : float, optional
    vf_clip_param : int, optional
    vf_loss_coef : float, optional
    entropy_coef : float, optional
    """

    batch_size = 512
    buffer_capacity = int(1e4)
    lr = 5e-3
    epsilon = 0.05
    gamma = 1
    target_update = 5

    def __init__(self,
                 network,
                 learning_rate: float = 5e-4,
                 clip_param: float = 0.1,
                 vf_clip_param: int = 10,
                 vf_loss_coef: float = 1,
                 entropy_coef: float = 0.02,
                 device_='cpu',
                 **kwargs):
        self._network = network
        self._loss_fn = PPOLoss(clip_param, vf_clip_param, vf_loss_coef, entropy_coef)

        self.optimizer = torch.optim.Adam(self._network.parameters(), lr=learning_rate, eps=1e-5)

        self.buffer = []
        self.buffer_count = 0

        self.device = device

    def store_transition(self, transition):
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(transition)
            self.buffer_count += 1
        else:
            index = int(self.buffer_count % self.buffer_capacity)
            self.buffer[index] = transition
            self.buffer_count += 1

    def clean_buffer(self):
        self.buffer = []

    @property
    def network(self):
        return self._network

    def update(self, anchor=None, div_weight=None):
        if len(self.buffer) < self.batch_size:
            return

        sample_data = random.sample(self.buffer, self.batch_size)

        action_names = [f'action_dim_{i}' for i in range(3)]

        feature = torch.vstack([t.feature for t in sample_data]).to(self.device)  # (bs, 11)
        old_logit_dict = {action_name: torch.vstack([t.logit_dict[action_name] for t in sample_data]) for action_name in action_names}  # 每个value的形状为(bs, 11)
        old_act_dict = {action_name: torch.vstack([t.act_dict[action_name] for t in sample_data]).squeeze() for action_name in action_names}  # 每个value的形状为(bs, )
        old_value = torch.tensor([t.value for t in sample_data]).to(torch.float).to(self.device)  # (bs, )
        done = torch.tensor([t.done for t in sample_data]).to(self.device)
        reward = torch.tensor([t.reward for t in sample_data]).to(torch.float).to(self.device)
        advantage = torch.tensor([t.advantage for t in sample_data]).to(torch.float).to(self.device)  # (bs, )

        # PPO loss
        out = self._network({'feature': feature})
        logit_dict = out['logits']
        decoder_mask = {action_names[i]: torch.ones(self.batch_size) for i in range(3)}
        neglogp_dict = self._network.negative_logp(logit_dict, old_act_dict, decoder_mask)
        neglogp = sum(neglogp_dict.values())  # (bs, )
        old_neglogp_dict = self._network.negative_logp(old_logit_dict, old_act_dict, decoder_mask)
        old_neglogp = sum(old_neglogp_dict.values())

        entropy_dict = self._network.entropy(logit_dict, decoder_mask)
        entropy = torch.mean(sum(entropy_dict.values()), dim=0, keepdim=False)  # (1, )

        target_value = advantage + old_value
        normalized_advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-8)
        value = out['value']

        loss, policy_loss, value_loss, clip_prob, ratio_diff = self._loss_fn(
            old_neglogp=old_neglogp,
            neg_logp=neglogp,
            advantage=normalized_advantage,
            old_value=old_value,
            value=value,
            target_value=target_value,
            entropy=entropy)

        if not anchor is None:
            act_prob_main = {}
            for action_name in [f'action_dim_{i}' for i in range(3)]:
                act_prob_main[action_name] = torch.softmax(out['logits'][action_name], dim=-1)
            with torch.no_grad():
                out_anchor = anchor.network({'feature': feature})
                act_prob_anchor = {}
                for action_name in [f'action_dim_{i}' for i in range(3)]:
                    act_prob_anchor[action_name] = torch.softmax(out_anchor['logits'][action_name], dim=-1)

            kl_loss = 0
            for action_name in action_names:
                kl_loss += - kl_divergence(act_prob_main[action_name], act_prob_anchor[action_name])
            kl_loss = kl_loss / 3.0 / self.batch_size * div_weight

            loss += kl_loss

        mean_advantage = torch.mean(advantage, dim=0, keepdim=False)
        kl_dict = self._network.kl(logit_dict, old_logit_dict, decoder_mask)
        kl = torch.mean(sum(kl_dict.values()))
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), 10)

        self._optimizer.step()

        summary = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "kl_loss": kl_loss,
            "entropy": entropy,
            "clip_prob": clip_prob,
            "ratio_diff": ratio_diff,
            "advantage": mean_advantage,
            "kl": kl,
        }

        return summary