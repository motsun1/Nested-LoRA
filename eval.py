import sys
import logging
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np

def eval(args):
    args["seed"]=args["seed"][0]
    device = copy.deepcopy(args["device"])

    checkpt_path = args.get("checkpt_path", None)
    if checkpt_path is None:
        raise ValueError("`checkpt_path` is required for eval runs. Add it to the config or pass --checkpt_path on the CLI.")

    logs_name = "logs/{}/{}".format(args["model_name"],args["backbone_type"])
    logfilename = "logs/{}/{}/eval_{}".format(
        args["model_name"],
        args["backbone_type"],
        args["dataset"],
    )
        
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    model._cur_task = data_manager.nb_tasks-1
    model._network.fc = nn.Linear(768, data_manager.nb_classes)
    model._total_classes = data_manager.nb_classes
    test_dataset = data_manager.get_dataset(np.arange(0, model._total_classes), source="test", mode="test" )
    model.test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=8)
    load_checkpoint(model, checkpt_path, args["device"][0])

    cnn_accy, _ = model.eval_task()
    logging.info("CNN: {}".format(cnn_accy["grouped"]))
    

def load_checkpoint(model, checkpt_path, device):
    state = torch.load(checkpt_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        load_info = model._network.load_state_dict(state["model_state_dict"], strict=False)
        logging.info(f"Loaded checkpoint with missing_keys={load_info.missing_keys}, unexpected_keys={load_info.unexpected_keys}")
    elif isinstance(state, dict):
        load_info = model._network.load_state_dict(state, strict=False)
        logging.info(f"Loaded checkpoint (raw state_dict) with missing_keys={load_info.missing_keys}, unexpected_keys={load_info.unexpected_keys}")
    else:
        raise ValueError(f"Unsupported checkpoint format from {checkpt_path}")

    model._network.to(device)


    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
