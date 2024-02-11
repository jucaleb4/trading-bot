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

    data_length = 128 # TODO: Magic num
    env = agent.env
    batch_size = agent.batch_size

    agent.inventory = []
    avg_loss = []

    state, _ = env.reset()
    # TODO: Make this better:- normalize
    transform_state = np.copy(state)
    transform_state[0] = transform_state[0]/400
    transform_state[1:11] = (transform_state[1:11]+225)/950

    for _ in range(data_length):
        # select an action
        action = agent.act(transform_state, on_policy=on_policy)

        next_state, reward, term, trunc, _ = env.step(action)
        transform_next_state = np.copy(next_state)
        transform_next_state[0] = transform_next_state[0]/400
        transform_next_state[1:11] = (transform_next_state[1:11]+225)/950
        transform_reward = (reward+225)/300
        done = term or trunc

        total_reward += reward
        num_steps += 1

        if callback is not None:
            callback.log((iter, num_steps, state[0], state[1], action, total_reward))

        if train_model:
            agent.remember(transform_state, action, transform_reward, transform_next_state, done)

        if len(agent.memory) > batch_size and len(agent.memory) % batch_size == 0:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state
        transform_state = transform_next_state
        if done:
            break

    if callback is not None:
        callback.save_and_clear_cache()

    soc = np.zeros(data_length+1)

    for t in range(data_length):        
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])
            soc[t+1] = min(400, soc[t] + 50)

            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta
            soc[t+1] = max(0, soc[t] + 50)

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))
            soc[t+1] = soc[t]

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            with open("ql_og.csv", "ab") as fp:
                np.savetxt(fp, soc, delimiter=",")
            return total_profit, history

    with open("ql_og.csv", "ab") as fp:
        np.savetxt(fp, soc, delimiter=",")
    return total_profit, history
