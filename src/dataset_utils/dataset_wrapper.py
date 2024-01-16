from dataset_utils.bias_in_bios import BiasBiosGender, BiasBiosOccupation
from dataset_utils.bigbench import BigBench
from dataset_utils.counterfact import CounterFact
from dataset_utils.fever import FEVER
from dataset_utils.hotpot import Hotpot
from dataset_utils.truthfulqa import TruthfulQA


class DatasetUtil:

    datasets = {
        "counterfact": CounterFact,
        "bias_gender": BiasBiosGender,
        "bias_prof": BiasBiosOccupation,
        "fever": FEVER,
        "hotpot": Hotpot,
        "BigBench": BigBench,
        "truthfulqa": TruthfulQA
    }

    def __init__(self):
        pass

    @staticmethod
    def get_dataset(dataset_name, logger, **kwargs):

        for it_dataset_name, it_dataset_constructor in DatasetUtil.datasets.items():
            if it_dataset_name == dataset_name:
                return it_dataset_constructor(logger, **kwargs)

        raise AssertionError(f"Dataset {dataset_name} not supported.")
