import pickle


from dataset_utils.abstract_dataset import AbstractDataset


class CounterFact(AbstractDataset):

    def __init__(self, dataset_file="/mnt/data/counterfact"):
        super(AbstractDataset, self).__init__()
        self.dataset_file = dataset_file

    def get_dataset(self, logger):

        with open(self.dataset_file, "rb") as f:
            data = pickle.load(f)

        num_dp = len(data)
        dataset = []

        for i in range(num_dp):
            question = data[i]["question"]
            answer = data[i]["gold-answer"]
            assert answer.startswith(" "), f"Found answer that doesn't start with space ${answer}$"
            dataset.append((question, answer))

        logger.log(f"Read dataset of size {num_dp}")

        return dataset
