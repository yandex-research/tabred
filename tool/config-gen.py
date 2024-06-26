import lib

import pprint


# Generate Configs for the experiments


NN_NUM_POLICY = {
    "homesite-insurance": "noisy-quantile",
    "weather": "noisy-quantile",
    "sberbank-housing": "noisy-quantile",
    "ecom-offers": "noisy-quantile",
    "homecredit-default": "noisy-quantile",

    # those datasets come pre-normalized (but with N/A values, thus we need to just fill nans)
    "cooking-time": "identity",
    "delivery-eta": "identity",
    "maps-routing": "identity",
}

NN_BATCH_SIZE = {
    "sberbank-housing": 256,
    "homesite-insurance": 1024,
    "weather": 1024,
    "cooking-time": 1024,
    "delivery-eta": 1024,
    "maps-routing": 1024,
    "ecom-offers": 1024,
    "homecredit-default": 1024,
}


def dataset_config(dataset: str, split: str, is_nn: bool):
    config = {
        "seed": 0,
        "path": f":data/{dataset}",
        "cache": True,
        "split": split,
    }

    if (lib.DATA_DIR/dataset/"X_cat.npy").exists():
        config["cat_policy"] = "ordinal"

    if is_nn and (lib.DATA_DIR/dataset/"X_num.npy").exists():
        config["num_policy"] = NN_NUM_POLICY[dataset]

    return config


# >>> model configs
nn_optimizer_tuning_config = {
    "type": "AdamW",
    "lr": ["_tune_", "loguniform", 1e-5, 1e-3],
    "weight_decay": ["_tune_", "loguniform", 1e-6, 1e-3],
}

# This defines the hyperparameter space for each model type
NN_MODEL_FUNCTION = {
    "mlp": "nn_baselines",
    "mlp-plr": "nn_baselines",
    "resnet": "nn_baselines",
    "snn": "nn_baselines",
    "dcn2": "nn_baselines",
    "ft_transformer": "nn_baselines",
    "tabr": "tabr",
    "tabr-causal": "tabr",
    "xgboost_": "xgboost_",
    "lightgbm_": "lightgbm_",
    "catboost_": "catboost_",
}


def nn_model_tuning_config(dataset: str, model: str):
    # Common options
    common = {
        "seed": 0,
        "function": f"bin.{NN_MODEL_FUNCTION[model]}.main",
        
    }

    if model == "xgboost_":
        return common | {
            "n_trials": 200,
            "space": {
                "seed": 0,
                "model": {
                    "booster": "gbtree",
                    "n_estimators": 4000,
                    "early_stopping_rounds": 3999,
                    "n_jobs": 1,
                    "tree_method": "hist",
                    "device": "cuda",
                    "use_label_encoder": False,
                    "colsample_bytree": ["_tune_", "uniform", 0.5, 1.0],
                    "gamma": ["_tune_","?loguniform", 0, 0.001, 100.0],
                    "lambda": ["_tune_", "?loguniform", 0.0, 0.1, 10.0],
                    "learning_rate": [
                        "_tune_",
                        "loguniform",
                        0.001,
                        1.0
                    ],
                    "max_depth": [
                        "_tune_",
                        "int",
                        3,
                        14
                    ],
                    "min_child_weight": [
                        "_tune_",
                        "loguniform",
                        0.0001,
                        100.0
                    ],
                    "subsample": [
                        "_tune_",
                        "uniform",
                        0.5,
                        1.0
                    ]
                },
                "fit": {
                    "verbose": True,
                }
            }
        }
    elif model == "lightgbm_":
        return common | {
            "n_trials": 200,
            "space": {
                "seed": 0,
                "model": {
                    "n_estimators": 4000,
                    "stopping_rounds": 3999,
                    "device_type": "gpu",
                    "verbose": 2,
                    "n_jobs": 4,
                    "feature_fraction": [
                        "_tune_",
                        "uniform",
                        0.5,
                        1.0
                    ],
                    "lambda_l2": [
                        "_tune_",
                        "?loguniform",
                        0.0,
                        0.1,
                        10.0
                    ],
                    "learning_rate": [
                        "_tune_",
                        "loguniform",
                        0.001,
                        1.0
                    ],
                    "num_leaves": [
                        "_tune_",
                        "int",
                        4,
                        768
                    ],
                    "min_sum_hessian_in_leaf": [
                        "_tune_",
                        "loguniform",
                        0.0001,
                        100.0
                    ],
                    "bagging_fraction": [
                        "_tune_",
                        "uniform",
                        0.5,
                        1.0
                    ]
                },
                "fit": {}

            }
        }
    elif model == "catboost_":
        return common | {
            "n_trials": 200,
            "space": {
                "seed": 0,
                "model": {
                    "iterations": 4000,
                    "early_stopping_rounds": 3999,
                    "od_pval": 0.001,
                    "bagging_temperature": [
                        "_tune_",
                        "uniform",
                        0.0,
                        1.0
                    ],
                    "depth": [
                        "_tune_",
                        "int",
                        3,
                        14
                    ],
                    "l2_leaf_reg": [
                        "_tune_",
                        "uniform",
                        0.1,
                        10.0
                    ],
                    "leaf_estimation_iterations": [
                        "_tune_",
                        "int",
                        1,
                        10
                    ],
                    "learning_rate": [
                        "_tune_",
                        "loguniform",
                        0.001,
                        1.0
                    ],
                    "task_type": "GPU",
                    "thread_count": 4
                },
                "fit": {
                    "logging_level": "Verbose"
                }
            }
        }
    elif model == "mlp":
        return common | {
            "n_trials": 100,
            "space": {
                "seed": 0,
                "patience": 16,
                "n_epochs": -1,
                "batch_size": NN_BATCH_SIZE[dataset],
                "model": {
                    "backbone": {
                        "type": "MLP",
                        "n_blocks": ["_tune_", "int", 1, 4],
                        "d_block": ["_tune_", "int-power-of-two", 7, 11], # [2^7 to 2^11]
                        "dropout": ["_tune_", "?uniform", 0.0, 0.0, 0.75],
                    }
                },
                "optimizer": nn_optimizer_tuning_config,
            },
        }
    elif model == "mlp-plr":
        return common | {
            "n_trials": 100,
            "space": {
                "seed": 0,
                "patience": 16,
                "n_epochs": -1,
                "batch_size": NN_BATCH_SIZE[dataset],
                "model": {
                    "num_embeddings": {
                        "type": "PeriodicEmbeddings",
                        "n_frequencies": ["_tune_", "int-power-of-two", 1, 7],
                        "d_embedding": ["_tune_", "int-power-of-two", 1, 5],
                        "frequency_init_scale": ["_tune_", "loguniform", 0.01, 100],
                        "lite": False,
                    },
                    "backbone": {
                        "type": "MLP",
                        "n_blocks": ["_tune_", "int", 1, 4],
                        "d_block": ["_tune_", "int-power-of-two", 7, 11], # [2^7 to 2^11]
                        "dropout": ["_tune_", "?uniform", 0.0, 0.0, 0.75],
                    },
                },
                "optimizer": nn_optimizer_tuning_config,
            }
        }
    elif model == "resnet":
        return common | {
            "n_trials": 100,
            "space": {
                "seed": 0,
                "patience": 16,
                "n_epochs": -1,
                "batch_size": NN_BATCH_SIZE[dataset],
                "model": {
                    "backbone": {
                        "type": "ResNet",
                        "n_blocks": ["_tune_", "int", 1, 4],
                        "d_block": ["_tune_", "int-power-of-two", 7, 11], # [2^7 to 2^11]
                        "d_hidden_multiplier": 2,
                        "dropout1": ["_tune_", "uniform", 0.0, 0.5],
                        "dropout2": ["_tune_", "?uniform", 0.0, 0.0, 0.5],
                    }
                },
                "optimizer": nn_optimizer_tuning_config,
            }

        }
    elif model == "snn":
        return common | {
            "n_trials": 100,
            "space": {
                "seed": 0,
                "patience": 16,
                "n_epochs": -1,
                "batch_size": NN_BATCH_SIZE[dataset],
                "model": {
                    "backbone": {
                        "type": "SNN",
                        "n_blocks": ["_tune_", "int", 1, 16],
                        "d_block": ["_tune_", "int-power-of-two", 7, 11], # [2^7 to 2^11]
                        "dropout": ["_tune_", "?uniform", 0.0, 0.0, 0.75],
                    }
                },
                "optimizer": nn_optimizer_tuning_config,
            }

        }
    elif model == "dcn2":
        return common | {
            "n_trials": 100,
            "space": {
                "seed": 0,
                "patience": 16,
                "n_epochs": -1,
                "batch_size": NN_BATCH_SIZE[dataset],
                "model": {
                    "backbone": {
                        "type": "DCNv2",
                        "d_cat_embedding": ["_tune_", "int-power-of-two", 1, 5],
                        "d_deep": ["_tune_", "int-power-of-two", 7, 11], # [2^7 to 2^11]
                        "n_cross_layers": ["_tune_", "int", 1, 4],
                        "n_deep_layers": ["_tune_", "int", 1, 4],
                        "dropout_p": ["_tune_", "?uniform", 0.0, 0.0, 0.5],
                        "nonlin_cross": False
                    }
                },
                "optimizer": nn_optimizer_tuning_config,
            }
        }
    elif model == "ft_transformer":
        return common | {
            "n_trials": 25,
            "space": {
                "seed": 0,
                "patience": 16,
                "n_epochs": -1,
                "batch_size": NN_BATCH_SIZE[dataset],
                "model": {
                    "backbone": {
                        "type": "FTTransformerBackbone",
                        "attention_n_heads": 8,
                        "ffn_d_hidden_multiplier": 2,
                        "ffn_activation": "ReLU",
                        "residual_dropout": 0,
                        "n_blocks": ["_tune_", "int", 1, 4],
                        "d_block": ["_tune_", "int-power-of-two", 5, 8],
                        "attention_dropout": ["_tune_", "uniform", 0.0, 0.5],
                        "ffn_dropout": ["_tune_", "uniform", 0.0, 0.5]
                    },
                    "num_embeddings": {
                        "type": "LinearEmbeddings"
                    }
                },
                "optimizer": nn_optimizer_tuning_config,
            }
        }
    elif model == "tabr":
        return common | {
            "n_trials": 100 if dataset == "homesite-insurance" else 25,
            "space": {
                "seed": 0,
                "patience": 16,
                "n_epochs": -1,
                "causal": False,
                "batch_size": NN_BATCH_SIZE[dataset],
                "context_size": 96,
                "model": {
                    "d_main": ["_tune_", "int-power-of-two", 7, 10], # [2^7 to 2^11]
                    "context_dropout": ["_tune_", "?uniform", 0.0, 0.0, 0.6],
                    "d_multiplier": 2.0,
                    "encoder_n_blocks": 0,
                    "predictor_n_blocks": 1,
                    "mixer_normalization": "auto",
                    "dropout0": ["_tune_", "?uniform", 0.0, 0.0, 0.6],
                    "dropout1": 0.0,
                    "normalization": "LayerNorm",
                    "activation": "ReLU"
                },
                "optimizer": nn_optimizer_tuning_config,
            }

        }
    elif model == "tabr-causal":
        return common | {
            "n_trials": 100 if dataset == "homesite-insurance" else 25,
            "space": {
                "seed": 0,
                "patience": 16,
                "n_epochs": -1,
                "causal": True,
                "batch_size": NN_BATCH_SIZE[dataset],
                "context_size": 96,
                "model": {
                    "d_main": ["_tune_", "int-power-of-two", 7, 10], # [2^7 to 2^11]
                    "context_dropout": ["_tune_", "?uniform", 0.0, 0.0, 0.6],
                    "d_multiplier": 2.0,
                    "encoder_n_blocks": 0,
                    "predictor_n_blocks": 1,
                    "mixer_normalization": "auto",
                    "dropout0": ["_tune_", "?uniform", 0.0, 0.0, 0.6],
                    "dropout1": 0.0,
                    "normalization": "LayerNorm",
                    "activation": "ReLU"
                },
                "optimizer": nn_optimizer_tuning_config,
            }
        }
    else:
        raise NotImplemented("No config space for such model")
    

if __name__ == "__main__":
    # for model in [
    #     "xgboost_",
    #     "catboost_",
    #     "lightgbm_",
    #     "mlp",
    #     "mlp-plr",
    #     "resnet",
    #     "snn",
    #     "dcn2",
    #     "ft_transformer",
    #     "tabr",
    #     "tabr-causal",
    # ]:
    #     for dataset in [
    #         "sberbank-housing",
    #         "homecredit-default",
    #         "ecom-offers",
    #         # "homesite-insurance",
    #         # "weather",
    #         # "cooking-time",
    #         # "delivery-eta",
    #         # "maps-routing",
    #     ]:
    #         config = nn_model_tuning_config(dataset, model)
    #         config["space"]["data"] = dataset_config(dataset, split="default", is_nn=model not in ["xgboost_", "catboost_", "lightgbm_"])
    #         print('\n' + '=' * 80 + '\n')
    #         pprint.pprint(config)
    #         out = lib.EXP_DIR/model/dataset
    #         out.mkdir(exist_ok=True, parents=True)
    #         lib.dump_config(out/'tuning.toml', config, force=True)


    for model in [
        "xgboost_",
        "mlp",
        "mlp-plr",
        "tabr",
    ]:
        for dataset in [
            "sberbank-housing",
            "homecredit-default",
            "ecom-offers",
            "homesite-insurance",
            "weather",
            "cooking-time",
            "delivery-eta",
            "maps-routing",
        ]:
            for split in [
                'sliding-window-0',
                'sliding-window-1',
                'sliding-window-2',
                'random-0',
                'random-1',
                'random-2',
            ]:
                config = nn_model_tuning_config(dataset, model)
                config["space"]["data"] = dataset_config(dataset, split=split, is_nn=model not in ["xgboost_", "catboost_", "lightgbm_"])

                print('\n' + '=' * 80 + '\n')
                pprint.pprint(config)

                out = lib.EXP_DIR/'temporal-shift-analysis'/model/f'{dataset}-{split}'
                out.mkdir(exist_ok=True, parents=True)
                lib.dump_config(out/'tuning.toml', config, force=True)
            
            

            
