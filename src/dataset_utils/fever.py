from datasets import load_dataset
from dataset_utils.abstract_dataset import AbstractDataset


class FEVER(AbstractDataset):

    def __init__(self):
        super(AbstractDataset, self).__init__()

    @staticmethod
    def _get_consistent_unique(dataset_split):

        dp_claim_dict = dict()
        for dp in dataset_split:

            claim = dp["claim"]
            label = dp["label"]

            if claim in dp_claim_dict:
                dp_claim_dict[claim].add(label)
            else:
                dp_claim_dict[claim] = {label}

        consistent = []
        for claim, labels in dp_claim_dict.items():
            if len(labels) == 1:
                consistent.append(
                    {
                        "question": claim,
                        "answer": list(labels)[0]
                     })

        return consistent

    def get_dataset(self, logger):

        dataset = load_dataset("EleutherAI/fever",'v1.0')

        paper_dev = dataset["paper_dev"]
        paper_test = dataset["paper_test"]

        # See if claims are unique
        claims_dev = [dp["claim"] for dp in paper_dev]
        claims_test = [dp["claim"] for dp in paper_test]

        logger.log(f"Raw paper_dev set is {len(claims_dev)} and paper_test set is {len(claims_test)}.")

        assert len(set(claims_dev).intersection(set(claims_test))) == 0, "dev and test set cannot share claims"
        logger.log("Paper_dev and paper_test splits dont have a common context/claim.")

        # Remove inconsistent and duplicate pairs
        dataset_dev = self._get_consistent_unique(paper_dev)
        dataset_test = self._get_consistent_unique(paper_test)

        logger.log(f"After filtering paper_dev set is {len(dataset_dev)} and paper_test set is {len(dataset_test)}.")

        # d = dataset['dev']
        # label_dict = {0: 'false', 1: 'true'}
        # i = 0
        # question = "Is the following statement true of false: " +d[i]['claim'] + " This is"
        # answer = label_dict[d[i]['label']]

        dataset = dataset_dev + dataset_test

        logger.log(f"Read dataset of size {len(dataset)} of which the first {len(dataset_dev)} examples are from the "
                   f"validation set and the remaining {len(dataset_test)} from the test split.")

        return dataset
