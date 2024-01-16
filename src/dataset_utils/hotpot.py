from datasets import load_dataset
from transformers import LlamaTokenizerFast
from dataset_utils.abstract_dataset import AbstractDataset


class Hotpot(AbstractDataset):

    def __init__(self, args, logger):
        super(Hotpot, self).__init__(args, logger)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(args.llama_tokenizer_path)

    def get_dataset(self):

        full_dataset = load_dataset("hotpot_qa", "fullwiki")
        num_val = len(full_dataset["validation"])

        # As the hotpot QA does not have answers for test set, we use the train set
        train = []
        ctr = 0
        for dp in full_dataset["train"]:
            if ctr >= num_val:
                break
            ctr += 1
            question = dp["question"].strip()
            answer = dp["answer"].strip()
            num_tokens = len(self.tokenizer(answer).input_ids)
            if num_tokens <= 15:
                train.append({"question": question,
                              "answer": answer})

        validation = []
        for dp in full_dataset["validation"]:
            question = dp["question"].strip()
            answer = dp["answer"].strip()
            num_tokens = len(self.tokenizer(answer).input_ids)
            if num_tokens <= 15:
                validation.append({"question": question,
                                   "answer": answer})

        dataset = train + validation
        num_dp = len(dataset)
        self.logger.log(f"Read dataset of size {num_dp} of which the first {len(train)} examples are from the "
                        f"train set and the remaining {len(validation)} from the validation split.")

        # Since this is an open-ended QA task, there is no fixed choice set
        choices = None

        return dataset, choices
