from tqdm import tqdm
from datasets import load_dataset
from dataset_utils.abstract_dataset import AbstractDataset


class TruthfulQA(AbstractDataset):

    def __init__(self, args, logger):
        super(AbstractDataset, self).__init__(args, logger)

    def get_dataset(self):

        dataset = load_dataset("truthful_qa", "multiple_choice")
        dataset = dataset['validation']
        num_dp = len(dataset)
        self.logger.log(f"Read dataset of size {num_dp}")

        pointwise_dataset = []
        for i in tqdm(range(0, num_dp)):
            question = dataset[i]["question"]
            answers = dataset[i]["mc2_targets"]["choices"]
            labels_ans = dataset[i]["mc2_targets"]["labels"]
            num_choices = len(answers)
            correct_answers = [answer_ix for answer_ix in range(0, num_choices) if labels_ans[answer_ix] == 1]
            assert len(answers) == len(labels_ans) and len(correct_answers) > 0

            for j in range(num_choices):
                # TODO check if prompting makes sense
                prompt = question + " " + answers[j]
                if not prompt.endswith("."):
                    prompt += "."
                prompt += "Is this statement true or false? This statement is"
                pointwise_dataset.append((prompt, labels_ans[j]))

        self.logger.log(f"Created modified dataset of size {len(pointwise_dataset)}.")
        choices = ["true", "false"]

        return pointwise_dataset, choices
