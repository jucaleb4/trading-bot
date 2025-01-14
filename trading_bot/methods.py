import logging
import numpy as np

from trading_bot.callback import EvalCallback

def train_model(agent, on_policy: bool=True):
    return run_model(agent, 0, train_model=True, on_policy=on_policy, callback=None)

def evaluate_model(agent, iter, callback: EvalCallback=None):
    return run_model(agent, iter, train_model=False, on_policy=True, callback=callback)

def run_model(agent, iter, train_model:bool=False, on_policy: bool=False, callback=None):
    """ Runs the model

    # TODO: How to choose number of steps (hard coded at 1024)

    :params agent: agent with policy and hyperparameters
    :params iter: current ieration of running a model
    :params train_model: whether we want to update parameters (train) or freeze (eval)
    :params on_policy: whether selecting action should be on policy or off policy (randomized)
    :params callback: callback with logging capabilities
    :returns total_reward: total accumulated rewards (undiscounted)
    :returns num_steps: total number of steps taken in the environment
    """
    total_reward = 0
    num_steps = 0

    data_length = 1024 # TODO: Magic num
    env = agent.env
    batch_size = agent.batch_size

    agent.inventory = []
    avg_loss = []

    state, _ = env.reset()
    # TODO: Make this better:- normalize
    transform_state = np.copy(state)
    # transform_state[0] = transform_state[0]/400
    # transform_state[1:11] = (transform_state[1:11]+225)/950
    # transform_state = np.zeros(len(state)-1)
    # for i in range(0, len(state)-2):
    #     transform_state[i] = state[i+1] - state[i+2]

    for _ in range(data_length):
        # select an action
        action = agent.act(transform_state, on_policy=on_policy)

        next_state, reward, term, trunc, _ = env.step(action)
        transform_next_state = np.copy(next_state)
        # transform_next_state[0] = transform_next_state[0]/400
        # transform_next_state[1:11] = (transform_next_state[1:11]+225)/950
        # transform_next_state = np.zeros(len(state)-3, dtype=float)
        # for i in range(0, len(state)-3):
        #     transform_next_state[i] = next_state[i+1] - next_state[i+2]
        # transform_reward = (reward+225)/300
        done = term or trunc

        total_reward += reward
        num_steps += 1

        if callback is not None:
            callback.log((iter, num_steps, next_state[0], next_state[1], action, total_reward))

        if train_model:
            agent.remember(transform_state, action, reward, transform_next_state, done)
            # agent.remember(transform_state, action, lransform_reward, transform_next_state, done)

            # TODO: More principled way to do this?
            # if len(agent.memory) % 4 == 0 and len(agent.memory) > batch_size:
            if len(agent.memory) > batch_size and len(agent.memory) % batch_size == 0:
                logging.debug(f"Train iter {num_steps}")
                loss = agent.train_experience_replay(batch_size)
                avg_loss.append(loss)

        state = next_state
        transform_state = transform_next_state
        if done:
            break

    if callback is not None:
        callback.save_and_clear_cache()

    print(f">>> total_reward={total_reward}")
    return total_reward, num_steps
