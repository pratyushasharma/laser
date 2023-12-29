from tqdm import tqdm
from datasets import load_dataset


def get_truthfulqa_pointwise_data(logger):

    dataset = load_dataset("truthful_qa", "multiple_choice")
    dataset = dataset['validation']
    num_dp = len(dataset)
    logger.log(f"Read dataset of size {num_dp}")

    pointwise_dataset = []
    #####
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

    logger.log(f"Created modified dataset of size {len(pointwise_dataset)}.")

    return pointwise_dataset
