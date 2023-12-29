import os
import time
import logging
import glob

import numpy as np

logging.basicConfig(level = logging.INFO)

def get_latest_run_id(log_path: str, exp_name) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, exp_name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(exp_name) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id

def create_exp_folder(log_path: str, exp_name: str, id: int=-1):
    """ If not existant, creates new experimental folder. Returns folder name
    which can be appended:
    
        fname_eval = os.path.join(save_path, "qlearn_eval.csv")
    """
    if id < 0:
        exp_name_with_ver = f"{exp_name}_{get_latest_run_id(log_path, exp_name)+1}"
    else:
        exp_name_with_ver = f"{exp_name}_{id}"

    save_path = os.path.join(log_path, exp_name_with_ver)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path

class EvalCallback():
    """ Simple callback the saves all system outputs """
        
    def __init__(self, fname:str, header: list, fmt: str=None):
        self.stime = time.time()

        # self.run = None
        # if config is not None:
            # TODO: Check project does not exist already (see also `self.run.log_artifact(self.artifact)`)
            # self.run = wandb.init(
            #     project="trading-bot",
            #     name=exp_name_with_ver,
            #     config=config,
            #     entity="jucaleb4",
            # )
            # TODO: Run this later
            # self.artifact = wandb.Artifact(name=f"{exp_name}_data", type="dataset")
            # self.artifact.add_dir(local_path=save_path)

        self.fname = fname
        self.expected_datasize = len(header)
        self.cache = np.zeros((1024, self.expected_datasize), dtype=float)
        self.ct = 0
        # fmt="%i,%1.4e,%i"
        # fmt="%i,%1.4e,%i,%i,%1.2f"
        self.fmt = fmt

        logging.info(f"Saving files to {fname}")
        with open(fname, "w") as fp:
            fp.write(f"{','.join(header)}\n")

    def _write_to_file(self):
        """ Private method that writes current cache to file """
        cache = self.cache[:self.ct]
        with open(self.fname, "ab") as fp:
            if self.fmt is None:
                # TODO: Can we just write fmt=None?
                np.savetxt(fp, cache, delimiter=",")
            else:
                np.savetxt(fp, cache, fmt=self.fmt)

    def save_and_clear_cache(self):
        """ Write current cached/saved data to file.  """
        self._write_to_file()

        # reset
        self.cache[:] = 0
        self.cache = np.vstack((self.cache, np.zeros(self.cache.shape)))
        self.ct = 0

    def log(self, data): 
        """ Log current data (if correct size). Updates cache if too full """
        assert len(data) == self.expected_datasize, f"Expected data of len {self.expected_data_size}, got {len(data)}"

        self.cache[self.ct] = data
        self.ct += 1
        if self.ct >= len(self.cache):
            self.update_file_and_clear_cache()
