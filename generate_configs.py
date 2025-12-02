import json
import os

base_config = {
    "prefix": "nested_lora",
    "dataset": "cifar224",
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": False,
    "shuffle": True,
    "init_cls": 10,
    "increment": 10,
    "model_name": "sema",
    "backbone_type": "pretrained_vit_b16_224_adapter",
    "device": ["0"],
    "seed": [1993],
    
    "batch_size":32,
    "weight_decay":0.0005,
    "min_lr":0,
    "ffn_adapter_type": "nested_lora",
    "nested_lora_rank": 16,
    "nested_lora_lr_fast": 0.01,
    "nested_lora_lr_slow": 0.001,
    "nested_lora_update_interval_slow": 5,
    "nested_lora_use_consolidation": False,
    "nested_lora_consolidation_alpha": 0.1,
    "ffn_num":16,
    "optimizer":"sgd",
    "vpt_type":"shallow",
    "prompt_token_num":5,

    "func_epoch": 5,
    "rd_epoch": 20,
    "init_lr": 0.005,
    "rd_lr": 0.01,
    "rd_dim": 128,
    "buffer_size": 500,
    "detect_batch_size": 128,

    "exp_threshold": 2,
    "adapt_start_layer": 9,
    "adapt_end_layer": 11
}

configs = [
    {"name": "exp_alpha0", "alpha": 0.0, "interval": 1, "use_consolidation": False},
    {"name": "exp_alpha0.05_int1", "alpha": 0.05, "interval": 1, "use_consolidation": True},
    {"name": "exp_alpha0.1_int1", "alpha": 0.1, "interval": 1, "use_consolidation": True},
    {"name": "exp_alpha0.2_int1", "alpha": 0.2, "interval": 1, "use_consolidation": True},
    {"name": "exp_alpha0.05_int5", "alpha": 0.05, "interval": 5, "use_consolidation": True},
    {"name": "exp_alpha0.1_int5", "alpha": 0.1, "interval": 5, "use_consolidation": True},
    {"name": "exp_alpha0.2_int5", "alpha": 0.2, "interval": 5, "use_consolidation": True},
]

for cfg in configs:
    new_config = base_config.copy()
    new_config["prefix"] = cfg["name"]
    new_config["nested_lora_use_consolidation"] = cfg["use_consolidation"]
    new_config["nested_lora_consolidation_alpha"] = cfg["alpha"]
    new_config["nested_lora_consolidation_interval"] = cfg["interval"]
    
    filename = f"exps/{cfg['name']}.json"
    with open(filename, "w") as f:
        json.dump(new_config, f, indent=4)
    print(f"Created {filename}")
