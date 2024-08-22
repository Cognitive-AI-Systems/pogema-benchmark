from collections import defaultdict
from copy import deepcopy

import numpy as np
import ray
import torch

from mamba.environments import Env


@ray.remote
class DreamerWorker:

    def __init__(self, idx, env_config, controller_config):
        self.runner_handle = idx
        self.controller_config = controller_config
        self.env = env_config[0].create_env()
        self.controller = controller_config.create_controller()
        self.in_dim = controller_config.IN_DIM
        self.env_type = env_config[0].ENV_TYPE
        self.lifelong = None

        if self.env_type == Env.POGEMA:
            self.on_target = env_config[0].on_target
            self.save_dir = env_config[0].SAVE_DIR
            self.render = env_config[0].RENDER

        self.n_agents = (
            self.env.n_agents
            if self.env_type == Env.STARCRAFT
            else self.env.get_num_agents()
        )

    def _check_handle(self, handle):
        if self.env_type == Env.STARCRAFT:
            return self.done[handle] == 0
        elif self.env_type == Env.POGEMA:
            return self.done[handle] == 0 and self.info[handle]["is_active"]

    def _select_actions(self, state):
        avail_actions = []
        observations = []
        fakes = []
        nn_mask = None

        for handle in range(self.n_agents):
            if self.env_type == Env.STARCRAFT:
                avail_actions.append(
                    torch.tensor(self.env.get_avail_agent_actions(handle))
                )

            def constains(handle, state):
                if isinstance(state, dict):
                    return handle in state
                return handle in range(len(state))

            if self._check_handle(handle) and constains(handle, state):
                fakes.append(torch.zeros(1, 1))
                if self.env_type == Env.POGEMA:
                    observations.append(state[handle].reshape(-1).unsqueeze(0))
                else:
                    observations.append(state[handle].unsqueeze(0))

            elif not self._check_handle(handle):
                fakes.append(torch.ones(1, 1))
                if self.env_type == Env.STARCRAFT:
                    obs = self.get_absorbing_state()
                elif self.env_type == Env.POGEMA:
                    obs = state[handle].reshape(-1).unsqueeze(0)

                observations.append(obs)
            else:
                fakes.append(torch.zeros(1, 1))
                if self.env_type == Env.STARCRAFT:
                    obs = (
                        torch.tensor(self.env.obs_builder._get_internal(handle))
                        .float()
                        .unsqueeze(0)
                    )
                elif self.env_type == Env.POGEMA:
                    obs = state[handle].reshape(-1).unsqueeze(0)

                observations.append(obs)

        observations = torch.cat(observations).unsqueeze(0)
        av_action = (
            torch.stack(avail_actions).unsqueeze(0) if len(avail_actions) > 0 else None
        )
        nn_mask = nn_mask.unsqueeze(0).repeat(8, 1, 1) if nn_mask is not None else None
        actions = self.controller.step(observations, av_action, nn_mask)

        return actions, observations, torch.cat(fakes).unsqueeze(0), av_action

    def _wrap(self, d):
        if isinstance(d, list):
            for key, value in enumerate(d):
                d[key] = torch.tensor(value).float()
        elif isinstance(d, dict):
            for key, value in d.items():
                d[key] = torch.tensor(value).float()
        return d

    def get_absorbing_state(self):
        state = torch.zeros(1, self.in_dim)
        return state

    def augment(self, data, inverse=False):
        aug = []
        if self.env_type == Env.STARCRAFT:
            default = list(data.values())[0].reshape(1, -1)
            it = data.keys()
        elif self.env_type == Env.POGEMA:
            default = data[0].reshape(1, -1)
            it = range(len(data))

        for handle in range(self.n_agents):
            if handle in it:
                aug.append(data[handle].reshape(1, -1))
            else:
                aug.append(
                    torch.ones_like(default) if inverse else torch.zeros_like(default)
                )
        return torch.cat(aug).unsqueeze(0)

    def _check_termination(self, info, steps_done):
        if self.env_type == Env.STARCRAFT:
            return "episode_limit" not in info
        elif self.env_type == Env.POGEMA:
            steps_done < self.env.grid_config.max_episode_steps
        else:
            return steps_done < self.env.max_time_steps

    def act(self, dreamer_params, state):
        self.controller.receive_params(dreamer_params)
        state = self._wrap(state)
        observations = []

        for handle in range(len(state)):
            observations.append(state[handle].reshape(-1).unsqueeze(0))

        observations = torch.cat(observations).unsqueeze(0)
        num_agents = observations.shape[1]
        actions = self.controller.step(observations, None, None)
        return actions.argmax(-1).numpy().reshape(1, num_agents)

    def reset(self, dreamer_params):
        self.controller = self.controller_config.create_controller()
        self.controller.receive_params(dreamer_params)

    def run(self, dreamer_params):
        return
        self.controller.receive_params(dreamer_params)

        if self.env_type == Env.POGEMA:
            state, self.info = self.env.reset()
            state = self._wrap(state)
        else:
            state = self._wrap(self.env.reset())

        steps_done = 0
        self.done = defaultdict(lambda: False)

        while True:
            steps_done += 1
            actions, obs, fakes, av_actions = self._select_actions(state)
            if self.env_type == Env.STARCRAFT:
                next_state, reward, done, info = self.env.step(
                    [action.argmax() for i, action in enumerate(actions)]
                )
            elif self.env_type == Env.POGEMA:
                next_state, reward, terminated, truncated, info = self.env.step(
                    [action.argmax() for i, action in enumerate(actions)]
                )
                done = torch.maximum(torch.Tensor(terminated), torch.Tensor(truncated))
                if actions.ndim < 3:
                    actions = actions[None]
            next_state, reward, done = (
                self._wrap(deepcopy(next_state)),
                self._wrap(deepcopy(reward)),
                self._wrap(deepcopy(done)),
            )
            self.done = done
            self.info = info
            self.controller.update_buffer(
                {
                    "action": self._wrap(actions),
                    "observation": obs,
                    "reward": self.augment(reward),
                    "done": self.augment(done),
                    "fake": fakes,
                    "avail_action": av_actions,
                }
            )

            state = next_state
            if all([done[key] == 1 for key in range(self.n_agents)]):
                if self._check_termination(info, steps_done):
                    obs = torch.cat(
                        [self.get_absorbing_state() for i in range(self.n_agents)]
                    ).unsqueeze(0)
                    actions = torch.zeros(1, self.n_agents, actions.shape[-1])
                    index = torch.randint(
                        0, actions.shape[-1], actions.shape[:-1], device=actions.device
                    )
                    actions.scatter_(2, index.unsqueeze(-1), 1.0)
                    items = {
                        "observation": obs,
                        "action": actions,
                        "reward": torch.zeros(1, self.n_agents, 1),
                        "fake": torch.ones(1, self.n_agents, 1),
                        "done": torch.ones(1, self.n_agents, 1),
                        "avail_action": (
                            torch.ones_like(actions)
                            if self.env_type == Env.STARCRAFT
                            else None
                        ),
                    }
                    self.controller.update_buffer(items)
                    self.controller.update_buffer(items)
                break

        if self.env_type == Env.STARCRAFT:
            reward = 1.0 if "battle_won" in info and info["battle_won"] else 0.0
            reward = {"reward": reward}
        elif self.env_type == Env.POGEMA:
            reward = {
                f"reward_agent_{key}": reward[key].reshape(-1).item()
                for key in range(self.n_agents)
            }
            
            if "metrics" in self.info[0]:
                for k in  self.info[0]["metrics"]:
                    reward.update(
                        {k: self.info[0]["metrics"][k]}
                    )
            if self.render:
                self.env.save_animation(
                    f"{self.save_dir}/episode_animation_{np.random.randint(0, 1000)}.svg"
                )

        return self.controller.dispatch_buffer(), {
            "idx": self.runner_handle,
            **reward,
            "steps_done": steps_done,
        }
