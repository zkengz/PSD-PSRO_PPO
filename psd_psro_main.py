import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from psd_psro_ppo import ModelPPO
from network_torch import Network

np.set_printoptions(precision=4)
Transition = namedtuple('Transition', ['feature', 'act_dict', 'logit_dict', 'value','reward', 'advantage', 'done'])


def torch_load_npz(network, model_path):
    npz_file = np.load(model_path)
    weights = [npz_file[file] for file in npz_file]
    with torch.no_grad():
        for target_p, p in zip(network.parameters(), weights):
            target_p.copy_(torch.from_numpy(p))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def new_agent(device):
    return ModelPPO(Network(), device=device)

def cal_gae(rewards, values, dones, gamma=0.99, lamb=0.95):
    """
    Parameters
    ----------
    rewards : List
    values : List
    dones : List
    gamma : float, optional
    lamb : float, optional

    Returns
    -------
    List
        advantage
    """
    values.append(torch.tensor(0.0))
    advantages = []
    adv = 0.0
    for i in reversed(range(len(rewards))):
        next_nonterminal = 1 - dones[i]
        delta = rewards[i] + gamma * values[i + 1] * next_nonterminal - values[i]
        adv = delta + gamma * lamb * next_nonterminal * adv
        advantages.append(adv)
    return list(reversed(advantages))

def simulate_one(env, oppo_policies, oppo_id_pools, main_agent, buffer_to_add):
    n_steps_rec = [0]
    with torch.no_grad():
        for oppo_id in oppo_id_pools:  # sample one episode for each opponent
            oppo_policy = oppo_policies[oppo_id]
            obs_data = env.reset_both_nn()
            feature_list = []
            act_dict_list = []
            logit_dict_list = []
            value_list = []
            reward_list = []
            agent_action_dict = {'red': {"action_dim_0": 0.1,
                                              "action_dim_1": 0.1,
                                              "action_dim_2": 0.1},
                                 'blue': {"action_dim_0": 0.1,
                                               "action_dim_1": 0.1,
                                               "action_dim_2": 0.1}}
            while True:
                feature_red_ = torch.from_numpy(np.array(obs_data['red'].obs['feature']).astype(np.float32)).unsqueeze(0)
                out_red = oppo_policy({'feature': feature_red_})
                feature_blue_ = torch.from_numpy(np.array(obs_data['blue'].obs['feature']).astype(np.float32)).unsqueeze(0)
                out_blue = main_agent.network({'feature': feature_blue_})
                for key in agent_action_dict['red']:
                    agent_action_dict['red'][key] = int(out_red['action'][key])
                for key in agent_action_dict['blue']:
                    agent_action_dict['blue'][key] = int(out_blue['action'][key])
                feature_list.append(feature_blue_)
                obs_data, game_done = env.step_both_nn(agent_action_dict)
                act_dict_list.append(out_blue['action'])
                logit_dict_list.append(out_blue['logits'])
                value_list.append(out_blue['value'])
                reward_list.append(obs_data['blue'].extra_info_dict['reward'])
                if game_done:
                    if not env.env.game_state['winner'] == 'Red' and not env.env.game_state['winner'] == 'Blue':
                        return None
                    if env.env.game_state['types_blue_win'] == 2:
                        reward_list[-1] += 100
                    break
            n_steps = len(act_dict_list)
            n_steps_rec.append(n_steps_rec[-1] + n_steps)  # record the length of each episode
            for n in range(n_steps):
                if n < n_steps - 1:
                    trans = Transition(feature_list[n], act_dict_list[n], logit_dict_list[n], value_list[n], reward_list[n], None, 0)
                else:
                    trans = Transition(feature_list[n], act_dict_list[n], logit_dict_list[n], value_list[n], reward_list[n], None, 1)
                buffer_to_add.append(trans)
            return n_steps_rec


class PSD_PSRO_SOLVER(object):
    def __init__(self, args, red_model_path_list, blue_model_path_list, load_info=None):
        env_config = {
           'scenario_name': '1v1_game',
        }
        self.env = Env(args.env_config)
        self.device = args.device
        self.sims_per_entry = args.sims_per_entry
        self.total_iters = args.total_iters
        self.oracle_iters = args.oracle_iters
        self.learn_step = 10
        self.fsp_eps = 1e-3
        self.div_weight = args.div_weight
        self.red_model_path_list = red_model_path_list
        self.blue_model_path_list = blue_model_path_list
        self.red_policy_set = [Network()]
        for red_model_path in red_model_path_list:
            red_network = Network()
            torch_load_npz(red_network, red_model_path)
            self.red_policy_set.append(red_network)
        if len(blue_model_path_list) == 0:
            self.blue_policy_set = [new_agent(self.device)]
        else:
            for blue_model_path in blue_model_path_list:
                blue_policy = ModelPPO(Network(), device=self.device)
                torch_load_npz(blue_policy.network, blue_model_path)
                self.blue_policy_set.append(blue_policy)
        rows = len(self.red_policy_set)
        cols = len(self.blue_policy_set)
        self.meta_games = np.zeros((rows, cols))  # blue winrate matrix, with rows as red policies and cols as blue policies 
        for idx in range(len(self.red_policy_set)):
            self.sim_game(self.red_policy_set[idx], self.blue_policy_set[0])
        self.ne = {'red': np.ones(rows) / rows,
                   'blue': np.ones(cols) / cols}
        self.hist = []
        self.save_every = 1
        self.evaluate_every = 2
        DEBUG_ = False
        if DEBUG_:
            self.oracle_iters = 100
            self.learn_step = 2

    def fictitious_play(self, meta_game, init_strategy=None, max_iters=5000):
        # returns nash equilibrium and final exploitability
        if init_strategy is None:
            m = meta_game.shape[0]
            n = meta_game.shape[1]
            red_strategy = np.ones(m) / m
            blue_strategy = np.ones(n) / n
        else:
            red_strategy = init_strategy['red']
            blue_strategy = init_strategy['blue']
        exp = 1
        exps = []
        it = 0
        while abs(exp) > self.fsp_eps:
            average_red = np.mean(red_strategy, axis=0)
            average_blue = np.mean(blue_strategy, axis=0)
            red_weighted_payouts = meta_game.dot(average_blue.T).T
            br_red = np.zeros_like(red_weighted_payouts)
            br_red[np.argmax(red_weighted_payouts)] = 1
            blue_weighted_payouts = average_red.dot(meta_game)
            br_blue = np.zeros_like(blue_weighted_payouts)
            br_blue[np.argmax(blue_weighted_payouts)] = 1
            exp_red = average_red.dot(meta_game).dot(br_blue)
            exp_blue = br_red.dot(meta_game).dot(average_blue)
            exp = exp_red - exp_blue
            exps.append(exp)
            red_strategy = np.vstack((red_strategy, br_red))
            blue_strategy = np.vstack((blue_strategy, br_blue))
            it += 1
            if it > max_iters:
                break
        ne = {'red': average_red, 'blue': average_blue}
        return ne, exps[-1]

    def sim_game(self, network_red, network_blue, sims_per_entry=None):
        if sims_per_entry is None:
            sims_per_entry = self.sims_per_entry
        if isinstance(network_red, ModelPPO):
            network_red = network_red.network
        if isinstance(network_blue, ModelPPO):
            network_blue = network_blue.network
        blue_winrate = 0
        agent_action_dict = {'red': {"action_dim_0": 0.1,
                                          "action_dim_1": 0.1,
                                          "action_dim_2": 0.1},
                             'blue': {"action_dim_0": 0.1,
                                           "action_dim_1": 0.1,
                                           "action_dim_2": 0.1}}
        with torch.no_grad():
            for _ in range(sims_per_entry):
                obs_data = self.env.reset_both_nn()
                while True:
                    feature_red_ = torch.from_numpy(np.array(obs_data['red'].obs['feature']).astype(np.float32)).unsqueeze(0)
                    out_red = network_red({'feature': feature_red_})
                    feature_blue_ = torch.from_numpy(np.array(obs_data['blue'].obs['feature']).astype(np.float32)).unsqueeze(0)
                    out_blue = network_blue({'feature': feature_blue_})
                    for key in agent_action_dict['red']:
                        agent_action_dict['red'][key] = int(out_red['action'][key])
                    for key in agent_action_dict['blue']:
                        agent_action_dict['blue'][key] = int(out_blue['action'][key])
                    obs_data, game_done = self.env.step_both_nn(agent_action_dict)
                    if game_done:
                        if self.env.env.game_state['winner'] == 'Red':
                            blue_winrate += 0
                        elif self.env.env.game_state['winner'] == 'Blue':
                            blue_winrate += 1
                        else:
                            continue
                        break
        return blue_winrate / sims_per_entry

    def update_meta_game(self, new_blue_policy):
        m = self.meta_game.shape[0]
        n = self.meta_game.shape[1]
        new_col = np.zeros(m)
        for idx in range(m):
            new_col[idx] = self.sim_game(self.red_policy_set[idx], new_blue_policy)

        meta_game = np.zeros((m, n + 1))
        meta_game[:, :n] = self.meta_game
        meta_game[:, n] = new_col
        self.meta_game = meta_game

        print("meta_game:\n", self.meta_game)

    def approximate_BR(self, oppo_policies, ne, main_agent=None, iterations=None): # approximate best response
        if main_agent is None:
            main_agent = new_agent(self.device)
        if iterations is None:
            iterations = self.oracle_iters

        oppo_id_pools = np.random.choice(len(oppo_policies), p=ne['red'], size=iterations)
        for idx in range(iterations // self.learn_step):
            buffer_to_add = []
            n_steps_rec = simulate_one(self.env,
                                    oppo_policies, oppo_id_pools[idx * self.learn_step:(idx + 1) * self.learn_step],
                                    main_agent, buffer_to_add)

        # calculate KL reward and add it to original reward
        features = torch.cat([trans.feature for trans in buffer_to_add], dim=0).to(self.device)
        logits = {action_name: [] for action_name in ['action_dim_{i}' for i in range(3)]}
        for trans in buffer_to_add:
            for action_name in ['action_dim_{i}' for i in range(3)]:
                logits[action_name].append(trans.logit_dict[action_name])

        kl_sum_all = []
        for blue_pol in self.blue_policy_set:
            with torch.no_grad():
                other_outs = blue_pol.network({'feature': features})
                kl_sum = 0
                for action_name in ['action_dim_{i}' for i in range(3)]:
                    kl_sum += F.kl_div(torch.log(torch.cat(logits[action_name], dim=0), -1),
                                            other_outs['logits'][action_name], reduction='sum') / self.learn_step
                kl_sum_all.append(kl_sum)

        kl_sum_all = torch.stack(kl_sum_all, axis=0)
        min_idx = kl_sum_all.sum(1).argmin().item()

        for n_i in range(len(n_steps_rec) - 1):
            psd_score = kl_sum_all[min_idx, n_steps_rec[n_i]:n_steps_rec[n_i + 1]].sum().item()
            psd_reward = 100 * psd_score * self.div_weight
            for n_j in range(n_steps_rec[n_i] + 1, n_steps_rec[n_i + 1] - 1):
                buffer_to_add[n_j] = buffer_to_add[n_j]._replace(reward=buffer_to_add[n_j].reward + psd_reward)
                psd_reward *= 0.99  # gamma

        # calculate advantage by GAE
        for n_i in range(len(n_steps_rec) - 1):
            reward_list = [buffer_to_add[n_j].reward for n_j in range(n_steps_rec[n_i], n_steps_rec[n_i + 1])]
            value_list = [buffer_to_add[n_j].value for n_j in range(n_steps_rec[n_i], n_steps_rec[n_i + 1])]
            done_list = [buffer_to_add[n_j].done for n_j in range(n_steps_rec[n_i], n_steps_rec[n_i + 1])]
            advantage_list = cal_gae(reward_list, value_list, done_list)
            for n_j in range(n_steps_rec[n_i], n_steps_rec[n_i + 1]):
                buffer_to_add[n_j] = buffer_to_add[n_j]._replace(advantage=advantage_list[n_j - n_steps_rec[n_i]])

        for trans in buffer_to_add:
            main_agent.store_transition(trans)

        summary = main_agent.update(anchor=self.blue_policy_set[min_idx], div_weight=self.div_weight)
        print("loss info:\n", summary)

        if idx > 0 and int(idx * self.learn_step) % 5000 == 0:  # save model
            model_dict_path = os.path.join("./model_logs", "_seed" + str(args.seed) + "_div" + str(self.div_weight))
            if not os.path.exists(model_dict_path):
                os.makedirs(model_dict_path)
            state_dict = main_agent.network.state_dict()
            arrays_list = []
            ordered_keys = list(state_dict.keys())
            for key in ordered_keys:
                tensor = state_dict[key]
                arrays_list.append(tensor.cpu().detach().numpy())
            save_path = os.path.join(model_dict_path, f"blue_model_{len(self.blue_policy_set)-1}_{int(idx * self.learn_step)}.npz")
            np.savez(save_path, *arrays_list)

        return main_agent

    def add_new(self, blue_policy):
        self.update_meta_game(blue_policy)
        self.blue_policy_set.append(blue_policy)

    def update_ne_exp(self, it):
        self.ne, sub_exp = self.fictitious_play(self.meta_game)
        self.hist_ne.append(copy.deepcopy(self.ne))

        if it > 0 and (it % self.save_every == 0):
            model_dict_path = os.path.join("./model_logs", "_seed" + str(args.seed) + "_div" + str(args.div_weight))
            if not os.path.exists(model_dict_path):
                os.makedirs(model_dict_path)

        red_model_pools = []
        blue_model_pools = []
        for policy in self.red_policy_set:
            pol = copy.deepcopy(policy)
            pol.buffer = None
            red_model_pools.append(pol)
        for policy in self.blue_policy_set:
            pol = copy.deepcopy(policy)
            pol.buffer = None
            blue_model_pools.append(pol)

        res = {
            'red_models': red_model_pools,
            'blue_models': blue_model_pools,
            'ne': self.hist_ne,
            'meta_game': self.meta_game,
        }

        save_path = os.path.join(model_dict_path, f"models_{it}.pkl")
        pkl.dump(res, open(save_path, "wb"))

    def run(self):
        main_policy = new_agent(self.device)
        it = 0
        start_it_time = time.time()
        while it < self.total_iters:
            clock_time = [time.time()]
            self.update_ne_exp(it)
            clock_time.append(time.time())
            self.blue_policy_set.append(main_policy)
            clock_time.append(time.time())
            main_policy = self.approximate_BR(
                self.red_policy_set, self.ne, main_policy)
            clock_time.append(time.time())

            main_policy = self.approximate_BR(
                self.red_policy_set, self.ne, main_policy)
            clock_time.append(time.time())

            self.blue_policy_set.pop()
            self.add_new(copy.deepcopy(main_policy))  # main_policy inherited from the last iteration
            clock_time.append(time.time())

            it += 1
            print("=====Iter: %d finish, Duration: %.4f minutes" %
                (it, (time.time() - start_it_time) / 60))
            start_it_time = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--sims_per_entry", type=int, default=100)
    parser.add_argument("--total_iters", type=int, default=5)
    parser.add_argument("--div_weight", type=float, default=0.1)

    args = parser.parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"
    print(args)

    red_model_path_list = [
        './red_models/model_1.npz',
        './red_models/model_2.npz',
        './red_models/model_3.npz',
    ]
    blue_model_path_list = []

    set_seed(args.seed)
    solver = PSD_PSRO_SOLVER(args, red_model_path_list, blue_model_path_list)
    solver.run()
