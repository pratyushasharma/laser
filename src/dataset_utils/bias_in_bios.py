from datasets import load_dataset
from dataset_utils.abstract_dataset import AbstractDataset


class BiasBiosGender(AbstractDataset):

    def __init__(self):
        super(AbstractDataset, self).__init__()

    def get_dataset(self, logger):

        # train_dataset = load_dataset("LabHC/bias_in_bios", split='train')
        # test_dataset = load_dataset("LabHC/bias_in_bios", split='test')
        dev_dataset = load_dataset("LabHC/bias_in_bios", split="dev")
        # gender_dict = {0: "male", 1: "female"}

        # Gender is a integer which is either 0 or 1
        dataset = [{"hard_text": dp["hard_text"], "answer": dp["gender"]} for dp in dev_dataset]

        # As male and female are single-token, we keep the entire dev split

        # question = dev_dataset[i]['hard_text'] + " Is the person male or female? The person is"
        # answer_gender = gender_dict[str(dev_dataset[i]['gender'])]

        return dataset


class BiasBiosOccupation(AbstractDataset):

    occupations = ['journalist', 'poet', 'composer', 'model', 'teacher', 'architect', 'painter', 'professor']

    def __init__(self):
        super(AbstractDataset, self).__init__()

    def get_dataset(self, logger):

        # train_dataset = load_dataset("LabHC/bias_in_bios", split='train')
        # test_dataset = load_dataset("LabHC/bias_in_bios", split='test')
        dev_dataset = load_dataset("LabHC/bias_in_bios", split="dev")

        dataset = [{"hard_text": dp["hard_text"], "answer": dp["profession"]} for dp in dev_dataset]

        # We filter out tokens that are more than 1 token
        occupation_dict = {0: 'accountant',
                           1: 'architect',
                           2: 'attorney',
                           3: 'chiropractor',
                           4: 'comedian',
                           5: 'composer',
                           6: 'dentist',
                           7: 'dietitian',
                           8: 'dj',
                           9: 'filmmaker',
                           10: 'interior designer',
                           11: 'journalist',
                           12: 'model',
                           13: 'nurse',
                           14: 'painter',
                           15: 'paralegal',
                           16: 'pastor',
                           17: 'personal trainer',
                           18: 'photographer',
                           19: 'physician',
                           20: 'poet',
                           21: 'professor',
                           22: 'psychologist',
                           23: 'rapper',
                           24: 'software engineer',
                           25: 'surgeon',
                           26: 'teacher',
                           27: 'yoga teacher'}

        # We choose occupations that result in single token answer

        filtered_dataset = [
            {
                "hard_text": dp["hard_text"],
                "answer": BiasBiosOccupation.occupations.index(occupation_dict[dp["answer"]])
            }
            for dp in dataset if occupation_dict[dp["answer"]] in BiasBiosOccupation.occupations
        ]

        # question = dev_dataset[i]['hard_text'] + " What is the person's profession? They are "
        # answer_profession = occupation_dict[str(dev_dataset[i]['profession'])]
        # if answer_profession[0] in ['a', 'e', 'i', 'o', 'u']:
        #     question += 'an'
        # else:
        #     question += 'a'

        logger.log(f"Out of a dataset of size {len(dataset)}, we create a filtered dataset of size "
                   f"{len(filtered_dataset)} that only contains occupations in {BiasBiosOccupation.occupations}.")

        return filtered_dataset
