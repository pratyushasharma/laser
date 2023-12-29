from datasets import load_dataset


def get_bb_dataset(split):

    if split == "causal_judgement":

        raw_dataset = load_dataset("tasksource/bigbench", "causal_judgment")
        choices = ["Yes", "No"]

        dataset = []
        for split_ in ["validation", "train"]:
            for dp in raw_dataset[split_]:
                targets = dp["targets"]
                assert len(targets) == 1
                assert targets[0] in choices
                dataset.append((dp["inputs"], targets[0]))

    elif split == "web_of_lies":

        raw_dataset = load_dataset("lighteval/big_bench_hard", "web_of_lies")
        choices = ["Yes", "No"]

        dataset = []
        for dp in raw_dataset["train"]:
            target = dp["target"]
            assert target in choices
            dataset.append((dp["input"], target))

    elif split == "epistemic_reasoning":

        raw_dataset = load_dataset("tasksource/bigbench", "epistemic_reasoning")
        choices = ["entailment", "non-entailment"]

        dataset = []
        for split_ in ["validation", "train"]:
            for dp in raw_dataset[split_]:
                targets = dp["targets"]
                assert len(targets) == 1
                assert targets[0] in choices
                dataset.append((dp["inputs"], targets[0]))

    elif split == "epistemic_reasoning_y":

        raw_dataset = load_dataset("tasksource/bigbench", "epistemic_reasoning")
        choices = ["True", "False"]

        dataset = []
        for split_ in ["validation", "train"]:
            for dp in raw_dataset[split_]:
                targets = dp["targets"]
                assert len(targets) == 1
                assert targets[0] in choices

                assert dp["inputs"].endswith("Relation:")
                text = dp["inputs"][:-len("Relation:")] + \
                       "Does the premise entails the hypothesis, True or False? Answer is"
                dataset.append((text, targets[0]))

    elif split == "qa_wikidata":

        raw_dataset = load_dataset("tasksource/bigbench", "qa_wikidata")

        dataset = []
        choices = None

        for split_ in ["validation", "train"]:
            for dp in raw_dataset[split_]:
                targets = dp["targets"]
                if len(targets) == 1:
                    dataset.append((dp["inputs"], targets[0]))

    else:
        raise AssertionError(f"Unhandled split {split}.")

    if choices is not None:
        assert len(set(choices)) == len(choices), f"Found duplicates in {choices}"

    return dataset, choices
