import os
import time
import logging
import glob

import numpy as np
import wandb

logging.basicConfig(level = logging.INFO)

def write_to_file(fname, np_data, fmt = None):
    with open(fname, "ab") as fp:
        if fmt is None:
            # TODO: Can we just write fmt=None?
            np.savetxt(fp, np_data, delimiter=",")
        else:
            np.savetxt(fp, np_data, fmt=fmt)

def get_latest_run_id(log_path: str, exp_name) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, exp_name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(exp_name) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id

class EvalCallback():
        
    def __init__(
            self, 
            log_path: str,
            exp_name: str,
            config: dict=None,
        ):
        self.stime = time.time()

        exp_name_with_ver = f"{exp_name}_{get_latest_run_id(log_path, exp_name)+1}"
        save_path = os.path.join(log_path, exp_name_with_ver)
        os.makedirs(save_path)

        fname_eval = os.path.join(save_path, "qlearn_eval.csv")
        fname_train = os.path.join(save_path, "qlearn_train.csv")
        fname_results = os.path.join(save_path, "qlearn_res.csv")
        fname_metadata = os.path.join(save_path, "qlearn_metadata.json")

        self.run = None
        if config is not None:
            # TODO: Check project does not exist already (see also `self.run.log_artifact(self.artifact)`)
            self.run = wandb.init(
                project="trading-bot",
                name=exp_name_with_ver,
                config=config,
                entity="jucaleb4",
            )
            # TODO: Run this later
            # self.artifact = wandb.Artifact(name=f"{exp_name}_data", type="dataset")
            # self.artifact.add_dir(local_path=save_path)

        self.fname_eval = fname_eval
        self.fname_train = fname_train
        self.fname_results = fname_results

        logging.info(f"Saving files to {fname_eval}, {fname_train}, {fname_results}")

        # TODO: Can we remove need for explicitly store "step" in csv?
        step_header = ["step", "s[0]", "a[0]"]
        iter_header = ["iter", "rwd", "steps (train)", "steps (eval)", "time"]

        with open(self.fname_eval, "w") as fp:
            fp.write(f"{','.join(step_header)}\n")
        with open(self.fname_train, "w") as fp:
            fp.write(f"{','.join(step_header)}\n")
        with open(self.fname_results, "w") as fp:
            fp.write(f"{','.join(iter_header)}\n")

        # save metadata
        step_metadata = {
            "step": "Number of simulation steps",
            "s[0]": "Current stock price ($)",
            "a[0]": "Action (0: HOLD, 1: BUY, 2: SELL)",
        }

        iter_metadata = {
            "iter": "Number of training iterations",
            "rwd": "Total profit for simulation",
            "steps": "Total number of training steps cumulatively",
            "time": "Cumulative time (sec)"
        }

        metadata = {
            'step': step_metadata,
            'iter': iter_metadata
        }

        import json
        with open(fname_metadata, 'w', encoding='utf-8') as fp:
            json.dump(metadata, fp, ensure_ascii=False, indent=4)

        # TODO: Find a way to automate choosing indices
        self.train_step_arr = np.zeros((1024, 3), dtype=float)
        self.eval_step_arr = np.zeros((1024, 3), dtype=float)
        self.eval_iter_arr = np.zeros((1024, 5), dtype=float)

        self.train_step_ct = 0
        self.eval_step_ct = 0
        self.eval_iter_ct = 0
        self.train_cum_step_ct = 0
        self.eval_cum_step_ct = 0
        self.eval_cum_iter_ct = 0

    # TODO: Can we abstract the three methods below since they are similar
    def write_train_step(self):
        write_to_file(
            self.fname_train, 
            self.train_step_arr[:self.train_step_ct], 
            fmt="%i,%1.4e,%i"
        )

        # reset
        self.train_step_arr[:] = 0
        self.train_step_arr = np.vstack(
            (self.train_step_arr, np.zeros(self.train_step_arr.shape))
        )
        self.train_cum_step_ct += self.train_step_ct
        self.train_step_ct = 0

    def train_step(self, data): 
        (state, action, reward, next_state, done) = data
        output = [
            self.train_cum_step_ct + self.train_step_ct + 1, 
            state[0][-1], 
            action
        ]
        self.train_step_arr[self.train_step_ct] = output

        self.train_step_ct += 1
        if self.train_step_ct >= len(self.train_step_arr):
            self.write_train_step()

    def write_eval_step(self):
        write_to_file(
            self.fname_eval, 
            self.eval_step_arr[:self.eval_step_ct], 
            fmt="%i,%1.4e,%i"
        )

        # reset
        self.eval_step_arr[:] = 0
        self.eval_step_arr = np.vstack(
            (self.eval_step_arr, np.zeros(self.eval_step_arr.shape))
        )
        self.eval_cum_step_ct += self.eval_step_ct
        self.eval_step_ct = 0

    def eval_step(self, data):
        (state, action, reward, next_state, done) = data
        output = [self.eval_cum_step_ct + self.eval_step_ct + 1, state[0][-1], action]

        self.eval_step_arr[self.eval_step_ct] = output
        self.eval_step_ct += 1
        if self.eval_step_ct >= len(self.eval_step_arr):
            self.write_eval_step()

    def write_eval_iter(self):
        write_to_file(
            self.fname_results, 
            self.eval_iter_arr[:self.eval_iter_ct], 
            fmt="%i,%1.4e,%i,%i,%1.2f"
        )

        # reset
        self.eval_iter_arr[:] = 0
        self.eval_iter_arr = np.vstack(
            (self.eval_iter_arr, np.zeros(self.eval_iter_arr.shape))
        )
        self.eval_cum_iter_ct += self.eval_iter_ct
        self.eval_iter_ct = 0

    def eval_iter(self, data):
        (total_reward) = data
        elapsed_time = time.time() - self.stime

        output = [
            self.eval_cum_iter_ct + self.eval_iter_ct + 1, 
            total_reward, 
            self.train_step_ct + self.train_cum_step_ct,
            self.eval_step_ct + self.eval_cum_step_ct,
            elapsed_time
        ]

        self.run.log({
            "train_iter": output[0], 
            "reward": output[1],
            "total_steps (train)": output[2],
            "total_steps (eval)": output[3],
            "time": elapsed_time,
        })

        self.eval_iter_arr[self.eval_iter_ct] = output
        self.eval_iter_ct += 1
        if self.eval_iter_ct >= len(self.eval_iter_arr):
            self.write_eval_iter()

    def finish(self):
        self.write_train_step()
        self.write_eval_step()
        self.write_eval_iter()

        if self.run is not None:
            # self.run.log_artifact(self.artifact)
            wandb.finish()
