from typing import Any, Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import datasets
from fewie.data.datasets.utils import get_label_to_id
from fewie.evaluation.scenarios.few_shot_linear_readout import (
    eval_few_shot_linear_readout,
)
from fewie.evaluation.utils import seed_everything


def evaluate_config(cfg: DictConfig) -> Dict[str, Any]:
    """Evaluates with the given hydra configuration.

    Args:
        cfg: Hydra-format configuration given in a dict.

    Returns:
        Evaluation result given in a dict, containing f1-micro and f1-macro scores. For each
        f1 score type, "mean", "margin_of_error" and "confidence" are reported.

    Raises:
        ValueError if the scenario provided by `cfg` is not yet supported.
    """
    seed_everything(cfg.seed)

    # Device configuration 
    device = torch.device("cpu")
    if hasattr(cfg, "device"):
        if cfg.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif cfg.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
    
    dataset = instantiate(cfg.dataset)

    # numerize labels (NER tags in this case) to {0, 1, ..., #classes}
    if isinstance(dataset, datasets.DatasetDict):
        label_to_id = get_label_to_id(dataset["train"], cfg.label_column_name)
    else:
        label_to_id = get_label_to_id(dataset, cfg.label_column_name)

    # instantiate the tokenizer
    dataset_processor = instantiate(
        cfg.dataset_processor,
        label_to_id=label_to_id,
        text_column_name=cfg.text_column_name,
        label_column_name=cfg.label_column_name,
    )

    encoder = instantiate(cfg.encoder)
    encoder = encoder.to(device)

    classifier = instantiate(cfg.evaluation.classifier)

    # tokenize text in data using tokenizers provided by transformers
    processed_dataset = dataset_processor(dataset)

    if cfg.scenario.name == "few_shot_linear_readout":
        few_shot_dataset = instantiate(
            cfg.evaluation.dataset,
            dataset=processed_dataset,
            columns=dataset_processor.feature_columns,
            label_column_name=cfg.label_column_name,
        )

        evaluation_results = eval_few_shot_linear_readout(
            classifier=classifier,
            encoder=encoder,
            dataset=processed_dataset,
            few_shot_dataset=few_shot_dataset,
            device=device,
            batch_size=cfg.batch_size,
            metrics=cfg.scenario.metrics,
            f1_include_O=cfg.f1_include_O,
        )
    else:
        raise ValueError("Unknown evaluation scenario {}".format(cfg.scenario))

    return evaluation_results
