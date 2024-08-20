from gymnasium import Wrapper
from loguru import logger
from pogema_toolbox.create_env import MultiMapWrapper
from pogema.wrappers.metrics import RuntimeMetricWrapper, AgentsDensityWrapper

from pogema import AnimationConfig, AnimationMonitor
from copy import deepcopy
from pogema import pogema_v0
from pogema.generator import generate_new_target, generate_from_possible_targets

class ProvideFutureTargetsWrapper(Wrapper):
    def _get_lifelong_global_targets_xy(self):
        all_goals = []
        cur_goals = self.grid.get_targets_xy()
        generators = deepcopy(self.random_generators)
        for agent_idx in range(self.grid_config.num_agents):
            distance = 0
            cur_goal = cur_goals[agent_idx]
            goals = [cur_goal]
            while distance < self.grid_config.max_episode_steps + 100:
                if self.grid_config.possible_targets_xy is None:
                    new_goal = generate_new_target(generators[agent_idx],
                                                self.grid.point_to_component,
                                                self.grid.component_to_points,
                                                cur_goal)
                else:
                    new_goal = generate_from_possible_targets(generators[agent_idx], self.grid_config.possible_targets_xy, cur_goal)
                    new_goal = (new_goal[0] + self.grid_config.obs_radius, new_goal[1] + self.grid_config.obs_radius)
                distance += abs(cur_goal[0] - new_goal[0]) + abs(cur_goal[1] - new_goal[1])
                cur_goal = new_goal
                goals.append(cur_goal)
            all_goals.append(goals)
        return all_goals
    
    def reset(self, **kwargs):
        observations, infos = self.env.reset(seed=self.env.grid_config.seed)
        observations[0]['after_reset'] = True
        observations[0]['max_episode_steps'] = self.env.grid_config.max_episode_steps
        if self.env.grid_config.on_target == 'restart':
            global_lifelong_targets_xy = self._get_lifelong_global_targets_xy()
            for idx, obs in enumerate(observations):
                obs['global_lifelong_targets_xy'] = global_lifelong_targets_xy[idx]
        return observations, infos

class CollisionsMetricWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._collisions = 0
        self._previous_positions = None

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        for i, obs in enumerate(observations):
            if obs['xy'] == self._previous_positions[i] and actions[i] != 0:
                self._collisions += 1
        self._previous_positions = [obs['xy'] for obs in observations]
        if all(terminated) or all(truncated):
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(collisions=self._collisions)
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        self._collisions = 0
        self._previous_positions = [obs['xy'] for obs in observations]
        return observations, info

def create_env_base(config):
    env = pogema_v0(grid_config=config)
    env = AgentsDensityWrapper(env)
    env = CollisionsMetricWrapper(env)
    env = ProvideFutureTargetsWrapper(env)
    env = MultiMapWrapper(env)
    if config.with_animation:
        logger.debug('Wrapping environment with AnimationMonitor')
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))

    # Adding runtime metrics
    env = RuntimeMetricWrapper(env)

    return env