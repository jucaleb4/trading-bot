
def train_model(agent, callback: dict=None):
    total_reward, num_steps = run_model(agent, train_model=True, callback=callback)

def evaluate_model(agent, callback: dict=None):
    total_reward, num_steps = run_model(agent, train_model=False, callback=callback)

def run_model(agent, train_model:bool=False, on_policy: bool=False, callback=None):
    total_reward = 0
    num_steps = 0

    # TODO: More elegant way to do this
    data_length = 1024 # TODO: Magic num
    env = agent.env
    batch_size = agent.batch_size

    agent.inventory = []
    avg_loss = []

    state, _ = env.reset()

    for t in range(data_length):
        # select an action
        action = agent.act(state, on_policy=on_policy)

        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        total_reward += reward
        num_steps += 1

        if callback is not None:
            callback.train_step(((state, action, reward, next_state, done)))

        if train_model:
            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > batch_size:
                loss = agent.train_experience_replay(batch_size)
                avg_loss.append(loss)

        state = next_state
        if done:
            break

    return total_reward, num_steps