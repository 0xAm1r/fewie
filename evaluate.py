import json
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from fewie.eval import evaluate_config
from fewie.utils import resolve_relative_path


logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="config", version_base="1.1")
def evaluate(cfg):
    """
    Conducts evaluation given the configuration.

    Args:
        cfg: Hydra-format configuration given in a dict.
    """
    resolve_relative_path(cfg=cfg, start_path=os.path.abspath(__file__))
    print(OmegaConf.to_yaml(cfg))

    evaluation_results = evaluate_config(cfg)

    logger.info("Evaluation results:\n%s" % evaluation_results)

    with open("./evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f)


if __name__ == "__main__":
    evaluate()
