def one_step(env0, actions, model0, pre_value, input_state, ps, one_episode_perf, message, episodic_buffer0):
    obs, vector, reward, done, _, on_goal, _, _, _, _, _, max_on_goal, num_collide, _, modify_actions = env0.joint_step(
        actions, one_episode_perf['episode_len'], model0, pre_value, input_state, ps, no_reward=False, message=message,
        episodic_buffer=episodic_buffer0)

    one_episode_perf['collide'] += num_collide
    vector[:, :, -1] = modify_actions
    one_episode_perf['episode_len'] += 1
    return reward, obs, vector, done, one_episode_perf, max_on_goal, on_goal
