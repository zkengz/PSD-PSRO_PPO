import torch
import torch.nn as nn
from typing import Dict, Any


def initialized_linear(in_features: int, out_features: int):
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_normal_(layer.weight)
    return layer

class Network(nn.Module):
    def __init__(self, inputs_length=20, action_space=[5, 5, 5]):
        super(Network, self).__init__()

        self._encoder = nn.Sequential(
            initialized_linear(inputs_length, 128), nn.ReLU(), nn.LayerNorm(128),
            initialized_linear(128, 256), nn.ReLU(), nn.LayerNorm(256)
        )

        self._decoder_dict = {}
        action_names = ['action_dim_0', 'action_dim_1', 'action_dim_2']
        for action_name, action_dim in zip(action_names, action_space):
            _dense_sequence = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, action_dim)
            )
            setattr(self, f"decoder_{action_name}", _dense_sequence)
            self._decoder_dict[action_name] = _dense_sequence

        self.value_seqs = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        encoder_output = self._encoder(inputs_dict['feature_name'])
        decoder_logits_dict = {}
        decoder_action_dict = {}

        for action_name, decoder in self._decoder_dict.items():
            logits = decoder(encoder_output)
            decoder_logits_dict[action_name] = logits

            # epsilon-greedy
            eps = 1e-10
            U = torch.rand(logits.shape, dtype=logits.dtype)
            if torch.cuda.is_available():
                U = U.cuda()
            U = -torch.log(-torch.log(U + eps) + eps)
            action = torch.argmax(logits + U, dim=-1)

            decoder_action_dict[action_name] = action

        value = torch.squeeze(self.value_seqs(encoder_output), dim=-1)

        predict_output_dict = {
            'logits': decoder_logits_dict,
            'action': decoder_action_dict,
            'value': value,
        }

        return predict_output_dict


    def negative_logp(self, logits_dict, action_dict, decoder_mask):
        neg_logp_dict = {}

        for action_name, decoder in self._decoder_dict.items():
            neg_logp = nn.functional.cross_entropy(
                input=logits_dict[action_name],
                target=action_dict[action_name],
                reduction="none"
            )
            neg_logp_dict[action_name] = torch.squeeze(neg_logp)

            neg_logp_dict[action_name] *= decoder_mask[action_name]

        return neg_logp_dict

    def entropy(self, logits_dict, decoder_mask):
        entropy_dict = {}
        for action_name, decoder in self._decoder_dict.items():
            _entropy = softmax_entropy_with_logits(logits_dict[action_name])
            entropy_dict[action_name] = torch.squeeze(_entropy) * decoder_mask[action_name]
        return entropy_dict

    def kl(self, logits_dict, other_logits_dict, decoder_mask):
        kl_dict = {}
        for action_name, decoder in self._decoder_dict.items():
            _kl = softmax_kl_with_logits(logits_dict[action_name], other_logits_dict[action_name])
            kl_dict[action_name] = torch.squeeze(_kl) * decoder_mask[action_name]
        return kl_dict

    def softmax_kl_with_logits(self, logits, other_logits):
        pass

    def softmax_entropy_with_logits(self, logits):
        pass