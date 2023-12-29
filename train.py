"""
Script for training Stock Trading Bot.

Usage:
  train.py [--on-policy] [--env-mode=<env-mode>] [--strategy=<strategy>] [--id=<id>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --on_policy                       Whether training should be off-policy (epsilon greedy) or on-policy
  --env-mode=<env-mode>             While mode of BatteryEnv to use (see BatteryMode). [default: 0]
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --id=<id>                         Experiment id [default: -1]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 12]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""
import numpy as np

from os import path
import time
import os
import logging
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)
from trading_bot.callback import EvalCallback, create_exp_folder

import gymnasium as gym
import gym_examples

import wandb

from enum import Enum

class BatteryMode(Enum): # borrowed from gym-examples
    DEFAULT = 0
    FINE_CONTROL = 1
    LONG_CHARGE = 2
    PENALIZE = 3
    DELAY = 4

def main(batch_size, 
         ep_count,
         env_mode:BatteryMode=0,
         on_policy=True,
         id=id,
         strategy="t-dqn", 
         model_name="model_debug", 
         pretrained=False,
):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    nhistory = 10
    mode_str = "default"
    if env_mode == BatteryMode.FINE_CONTROL.value:
        mode_str = "fine_control"
    elif env_mode == BatteryMode.LONG_CHARGE.value:
        mode_str = "long_charge"
    elif env_mode == BatteryMode.PENALIZE.value:
        mode_str = "penalize"
    elif env_mode == BatteryMode.DELAY.value:
        mode_str = "delay"

    env = gym.make("gym_examples/BatteryEnv-v0", nhistory=nhistory, data="periodic", mode=mode_str)

    agent = Agent(env, batch_size=batch_size, strategy=strategy, 
                  pretrained=pretrained, model_name=model_name)

    # setup wandb
    config = {
        "data": "periodic",
        "alg": "t-dqn",
        "on_policy": bool(on_policy),
        "battery_env_mode": env_mode,
    }

    # setup local logging
    log_path = "logs"
    exp_name = "periodic_qlearn"
    save_path = create_exp_folder(log_path, exp_name, id)

    # for logging each step of evaluation
    fname_eval = os.path.join(save_path, "eval.csv")
    eval_header = ["iter", "step", "soc", "lmp", "action", "reward"]
    eval_fmt = "%i,%i,%1.2f,%1.4e,%i,%1.2f"
    eval_callback = EvalCallback(fname_eval, eval_header, eval_fmt)

    # for logging overall results from epoch
    fname_ep = os.path.join(save_path, "episode_result.csv")
    ep_header = ["iter", "time (sec)", "tot_ep_rwd", "cum_steps"]
    ep_fmt = "%i,%1.4e,%1.4e,%i"
    ep_callback = EvalCallback(fname_ep, ep_header, ep_fmt)

    # for saving config
    fname_metadata = os.path.join(save_path, "metadata.json")
    import json
    with open(fname_metadata, "w", encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=4)

    wandb_log = None
    wandb_log = wandb.init(
        project="trading-bot",
        name=save_path.replace("/", "-"),
        config=config,
        entity="jucaleb4",
    )

    total_steps = 0
    
    try:
        stime = time.time()
        for i in range(ep_count):
            _, steps = train_model(agent, on_policy=on_policy)
            elap_time = time.time() - stime
            total_steps += steps

            total_reward, _ = evaluate_model(agent, i, callback=eval_callback)
            ep_callback.log((i, elap_time, total_reward, total_steps))
            ep_callback.save_and_clear_cache()
            if wandb_log is not None:
                wandb_log.log({
                    "eval_iter": i, 
                    "elap_time": elap_time, 
                    "tot_ep_rwd": total_reward, 
                    "cum_steps": total_steps
                })

            # show_train_result(train_result, val_result)
    except KeyboardInterrupt:
        print("Terminated early, saving to file...")
        ep_callback.save_and_clear_cache()
        eval_callback.save_and_clear_cache()
        wandb.finish()

if __name__ == "__main__":
    args = docopt(__doc__)

    on_policy = args["--on-policy"]
    env_mode = int(args["--env-mode"])
    strategy = args["--strategy"]
    id = int(args["--id"])
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(batch_size=batch_size,
             ep_count=ep_count, 
             env_mode=env_mode,
             on_policy=on_policy,
             id=id,
             strategy=strategy, 
             model_name=model_name, 
             pretrained=pretrained)
    except KeyboardInterrupt:
        print("Aborted!")
