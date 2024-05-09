import pathlib
import sys

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
from hydra_slayer import Registry
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.append("../../../")

import src.datasets
import src.models
import src.utils


def create_results(model, data_loader, raw_data):
    predictions = src.utils.predict(model, data_loader)
    mapping = raw_data.get_all_labels()
    mapping["preds"] = predictions.detach().cpu().numpy()
    return mapping


def measure_metrics(result: pd.DataFrame):
    metrics = {}
    result = result.copy()

    non_agg_accuracy = accuracy_score(result["target"], np.where(result["preds"] > 0.5, 1, 0))
    non_agg_roc_auc = roc_auc_score(result["target"], result["preds"])

    result["slide"] = result["large_image"].str.rsplit("/").str[-2]

    # Inverse sigmoid
    result["preds"] = -np.log((1 / result["preds"]) - 1)
    result = result.groupby("slide")[["preds", "target"]].mean()
    # Apply sigmoid back
    result["preds"] = 1 / (1 + np.exp(-result["preds"]))

    agg_accuracy = accuracy_score(result["target"], np.where(result["preds"] > 0.5, 1, 0))
    agg_roc_auc = roc_auc_score(result["target"], result["preds"])

    metrics["non_agg_accuracy"] = non_agg_accuracy
    metrics["non_agg_roc_auc"] = non_agg_roc_auc
    metrics["agg_accuracy"] = agg_accuracy
    metrics["agg_roc_auc"] = agg_roc_auc

    return metrics


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.data.load_dir)

    num_workers = cfg.eval_params.num_workers
    batch_size = cfg.eval_params.batch_size

    device = src.utils.get_default_device()
    logger.info(f"Current device is {device}")

    cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    registry = Registry()
    registry.add_from_module(src.datasets, prefix="src.datasets.")
    registry.add_from_module(src.models, prefix="src.models.")

    valid_cfg = cfg_dct["dataset"].copy()
    valid_cfg["annotation_file"] = load_dir / "valid.csv"
    valid_data = registry.get_from_params(**valid_cfg)

    test_cfg = cfg_dct["dataset"].copy()
    test_cfg["annotation_file"] = load_dir / "test.csv"
    test_data = registry.get_from_params(**test_cfg)

    valid_dataloader = DataLoader(valid_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    valid_dataloader = src.utils.DeviceDataLoader(valid_dataloader, device)
    test_dataloader = src.utils.DeviceDataLoader(test_dataloader, device)

    model = registry.get_from_params(**cfg_dct["model"])
    model.load_state_dict(torch.load(cfg.data.saved_model_path, map_location=torch.device("cpu")))
    src.utils.to_device(model, device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"The total number of trainable parameters is {params}")

    with torch.no_grad():
        model.eval()

        valid_result = create_results(model, valid_dataloader, valid_data)
        test_result = create_results(model, test_dataloader, test_data)

        valid_result.to_csv("valid_result.csv", index=False)
        test_result.to_csv("test_result.csv", index=False)

    logger.info(f"Valid metrics: {measure_metrics(valid_result)}")
    logger.info(f"Test metrics: {measure_metrics(test_result)}")


if __name__ == "__main__":
    main()
