class AbstractDataset:

    def __init__(self):
        pass

    def get_dataset(self, args, logger):
        """
        :param args: command line arguments
        :param logger: logger for logging
        :return: returns two things:
        - a dataset which is either a dictionary of validation and test or a list
        - a choice list which is a list of possible choices for multi-choice answering questions and None for others
        """
        raise NotImplementedError()
