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

    def __init__(self, args, logger):
        self.args = args
        self.dataset_name = args.dataset_name
        self.logger = logger

    def get_dataset(self):

        for it_dataset_name, it_dataset_constructor in DatasetUtil.datasets.items():
            if it_dataset_name == self.dataset_name:
                dataset, choices = it_dataset_constructor().get_dataset(self.logger, self.args)

                if type(dataset) == list:
                    # Convert to validation and test set
                    pass

        raise AssertionError(f"Dataset {self.dataset_name} not supported.")
