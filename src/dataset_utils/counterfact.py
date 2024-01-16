import os
import json
import time
import pickle
import urllib.request

from dataset_utils.abstract_dataset import AbstractDataset


class CounterFact(AbstractDataset):

    def __init__(self, dataset_file="./data/counterfact"):
        super(AbstractDataset, self).__init__()
        self.dataset_file = dataset_file

    def download_counterfact_dataset(self, logger):

        time_s = time.time()
        rome_path = "https://rome.baulab.info/data/dsets/counterfact.json"
        logger.log(f"Fetching Counterfact data from {rome_path}")

        with urllib.request.urlopen(rome_path) as url:
            orig_dataset = json.load(url)

        logger.log(f"Dataset fetched in {time.time() - time_s:.3f} seconds.")

        logger.log(f"The original dataset has {len(orig_dataset)} many datapoints.")
        dataset = []
        for dp in orig_dataset:
            question = dp["requested_rewrite"]["prompt"].format(dp["requested_rewrite"]["subject"])
            paraphrases = dp["paraphrase_prompts"]
            assert len(paraphrases) == 2, f"Expected 2 paraphrases per questions but instead found {len(paraphrases)}."
            answer = dp["requested_rewrite"]["target_true"]["str"]

            dataset.append(
                {"question": question,
                 "gold-answer": " " + answer
                 })

            for paraphrase in paraphrases:
                dataset.append(
                    {"question": paraphrase,
                     "gold-answer": " " + answer
                     })

        logger.log(f"After processing, the new dataset has {len(dataset)} many datapoints.")

        with open(self.dataset_file, "wb") as f:
            pickle.dump(dataset, f)

        return dataset

    def get_dataset(self, logger, args):

        if not os.path.exists(self.dataset_file):
            original_data = self.download_counterfact_dataset(logger)
        else:
            with open(self.dataset_file, "rb") as f:
                original_data = pickle.load(f)

        num_dp = len(original_data)
        dataset = []

        for i in range(num_dp):
            question = original_data[i]["question"]
            answer = original_data[i]["gold-answer"]
            assert answer.startswith(" "), f"Found answer that doesn't start with space ${answer}$"
            dataset.append((question, answer))

        logger.log(f"Processed CounterFact dataset of size {num_dp}")

        return dataset
