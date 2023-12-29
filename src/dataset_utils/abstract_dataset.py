class AbstractDataset:

    def __init__(self):
        pass

    def get_dataset(self, logger):
        raise NotImplementedError()

    def create_generation_prompt(self, question):
        raise NotImplementedError()

    def update_dataset_metric(self):
        raise NotImplementedError()
