import os
import sys
import traceback

sys.path.append("../../")  # nopep8

# 与 accelerate 一致：删除 state_dict 中重复的 shared tensor 以满足 safetensors 要求，但不打 "Removed shared tensor" 的 warning
import collections
import torch
from accelerate.utils import other as _acc_other
from accelerate.utils.modeling import id_tensor_storage

def _clean_state_dict_silent(state_dict):
    ptrs = collections.defaultdict(list)
    for name, tensor in state_dict.items():
        if not isinstance(tensor, str):
            ptrs[id_tensor_storage(tensor)].append(name)
    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
    for names in shared_ptrs.values():
        found_names = [name for name in names if name in state_dict]
        for name in found_names[1:]:
            del state_dict[name]
    state_dict = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    return state_dict

_acc_other.clean_state_dict_for_safetensors = _clean_state_dict_silent

from Core.functions.option_parser import get_classification_options
from Core.classes.dataset_for_classification import Classification_Dataset
from Core.classes.logger import TrainingExperimentLogger
from Core.classes.classification_model import Classification_model
from Core.classes.tokenizer import QA_Tokenizer_T5


def run(opts):
    ### Experiment Initialization ###
    logger = TrainingExperimentLogger(opts)
    logger.start_experiment(opts, start_fresh=True)
    logger.accelerator.print(f"Running experiment '{opts['identifier']}' started!")
    ### Tokenizer ###
    tokenizer_obj = QA_Tokenizer_T5(opts)
    ### Dataset ###
    logger.accelerator.print(f"Loading and processing the dataset...")
    dataset_obj_trainval = Classification_Dataset(opts, tokenizer_obj)
    dataset_obj_trainval.load_dataset(
        "Train", opts["training_data"], opts['input_format'], opts["validation_data"], opts["percentage"]
    )
    dataset_obj_test = Classification_Dataset(opts, tokenizer_obj)
    dataset_obj_test.load_dataset("Test", opts["testing_data"], opts['input_format'], percentage=opts["percentage"])
    ### Model ###
    logger.accelerator.print(f"Load the model...")
    model_obj = Classification_model(opts, tokenizer_obj, dataset_obj_trainval, dataset_obj_test)
    ### Train ###
    logger.accelerator.print(f"Start classification train...")
    model_obj.run(logger, opts)

    ### End ###
    logger.accelerator.print(f"Experiment ended!")
    logger.end_experiment()


if __name__ == "__main__":
    try:
        run(get_classification_options())
    except Exception as e:
        rank = os.environ.get("RANK", "0")
        err_file = os.path.join(os.getcwd(), f"classification_error_rank{rank}.log")
        with open(err_file, "w") as f:
            traceback.print_exc(file=f)
        print(f"[rank {rank}] Exception written to {err_file}", flush=True)
        raise
