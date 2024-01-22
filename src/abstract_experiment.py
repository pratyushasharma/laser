import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from experiment_header import ExperimentHeader
from intervention.intervention_wrapper import InterventionWrapper
from intervention.parse_interventions import InterventionParser
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress


class Results:

    def __init__(self, val_acc, val_logloss, test_acc, test_logloss):
        self.val_acc = val_acc
        self.val_logloss = val_logloss
        self.test_acc = test_acc
        self.test_logloss = test_logloss

    def to_str(self, only_test=False):
        if only_test:
            return f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"
        else:
            return f"Validation acc {self.val_acc:.3f}, Validation logloss {self.val_logloss:.3f}, " \
                   f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"


class AbstractExperiment:

    def __init__(self, save_dir, logger):

        header_util = ExperimentHeader()
        self.setup = header_util.generate_header()

        self.save_dir = save_dir
        self.logger = self.setup.logger
        self.args = self.setup.args

        self.llm = None
        self.tokenizer = None
        self.dataset = None

        # Parse the list of interventions to perform
        self.interventions = InterventionParser(setup=self.setup).create_grid_search_interventions()

        # Object to measure progress (as in time taken and time left to complete)
        self.progress = Progress(logger=logger)

        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics(logger=logger)

        # Device for the experiment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):

        self.logger.log(f"Starting a new experiment. LLM Name: {self.args.llm_name}, Dataset {self.args.dataset}.")

        # Make the LLM and dataset
        self.llm, self.tokenizer = self.setup.llm_util.get_llm_and_tokenizer()

        # Make the dataset
        self.logger.log("Creating the dataset.")
        dataset, choices = self.setup.dataset_util.get_dataset()
        self.logger.log("Dataset created.")

        # Evaluate the model. We make two decisions
        # - either to do grid search or evaluate all provided intervention(s) at once
        # - evaluate on the entire dataset or do selection based on validation
        #   (and evaluate on test only if validation performance improves)

        for intervention in self.interventions:

            # Perform intervention
            time_edit_start = time.time()
            model_edit = InterventionWrapper.get_edited_model(model=model,
                                                              intervention=intervention,
                                                              logger=self.logger)

            model_edit.to(self.device)
            self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

            # Evaluate the model
            for split_name, split_datapoints in dataset.items():

                predictions = self.evaluate_model(model=model_edit,
                                                  dataset=split_datapoints,
                                                  choices=choices)

                # Calculate accuracy

            # Save results and terminate
            self.terminate_and_save(predictions)

    def evaluate_model(self, model, dataset, choices):

        assert type(dataset) == list, f"Dataset must be a list, found {type(dataset)}."
        dataset_size = len(dataset)

        predictions = []

        choice_token_ids = self.get_choice_tokens(choices, self.tokenizer)
        if choice_token_ids is None:
            single_token_choices = False
            self.logger.log(f"Set of choices {choices} is a multi-token set.")
        else:
            single_token_choices = True
            self.logger.log(f"Set of choices {choices} is a single token set with token ids {choice_token_ids}.")

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        for i in tqdm(range(0, dataset_size)):

            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            prompt = dataset[i][0]
            label = dataset[i][1]

            with torch.no_grad():

                if single_token_choices:
                    is_correct, log_prob_results = self.single_token_eval(prompt=prompt,
                                                                          label=label,
                                                                          model_edit=model,
                                                                          choices=choices,
                                                                          choice_token_ids=choice_token_ids)
                else:
                    is_correct, log_prob_results = self.multi_token_eval(prompt=prompt,
                                                                         label=label,
                                                                         model_edit=model,
                                                                         choices=choices)

            # We compute 0-1 match, f1, precision, and recall score in addition to log-prob of the answer tokens
            # correct_log_prob_results = [all_log_prob_results[answer_ix] for answer_ix in correct_answers]
            self.dataset_metric.accept(is_correct=is_correct,
                                       f1pr_score=None,
                                       log_prob_results=log_prob_results)

            predictions_ = {
                "ix": i,
                "question": prompt,
                "gold-answer": label,
                "generation": "N/A",
                "correct": is_correct,
                "f1_score": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "case-sensitive": self.case_sensitive,  # We ignore case when checking answer
                "white-space-strip": self.strip,  # We ignore white space when checking answer
                "total_logprob": log_prob_results.total_log_prob,
                "answer_logprob": log_prob_results.answer_log_prob,
                "answer_length": log_prob_results.answer_len
            }
            predictions.append(predictions_)

        return predictions

    def get_choice_tokens(self, choices, tokenizer):

        choice_token_ids = []
        for choice in choices:
            assert not choice.startswith(" "), f"Expecting choice token {choice} to not start with space"
            assert not choice.endswith(" "), f"Expecting choice token {choice} to not end with space"

            token_ids = tokenizer(f"{choice}")
            if not (len(token_ids["input_ids"]) == 2 and token_ids["input_ids"][0] == 1):
                # This is a multi-token target and so must be evaluated differently
                return None
            else:
                token_id = int(token_ids["input_ids"][1])
                choice_token_ids.append(token_id)

        return choice_token_ids

    def single_token_eval(self, prompt, label, model_edit, choices, choice_token_ids):

        input_and_answer = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate from the model
        # Compute log probability of question + answer
        results = model_edit(input_and_answer.input_ids)
        logits = results.logits[0]  # question + answer length x vocab
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)  # question + answer length x vocab

        choice_logprobs = [log_prob[-1, choice_token_id].item() for choice_token_id in choice_token_ids]

        prediction_label_id = int(np.argmax(choice_logprobs))
        label_id = choices.index(label)

        is_correct = label_id == prediction_label_id

        answer_log_prob = choice_logprobs[label_id]
        log_prob_results = ContextAnswerLogProb(total_log_prob=answer_log_prob,
                                                answer_log_prob=answer_log_prob,
                                                answer_len=1)

        return is_correct, log_prob_results

    def multi_token_eval(self, prompt, label, model_edit, choices):

        all_log_prob_results = []

        for choice in choices:

            input_and_answer = self.tokenizer(prompt + " " + choice, return_tensors="pt").to(self.device)

            # Generate from the model
            # Compute log probability of question + answer
            results = model_edit(input_and_answer.input_ids)
            logits = results.logits[0]  # question + answer length x vocab
            log_prob = torch.nn.functional.log_softmax(logits, dim=1)  # question + answer length x vocab

            log_prob_results = self.metrics.answer_log_prob(log_prob=log_prob,
                                                            question_answer_token_ids=input_and_answer.input_ids[0],
                                                            answer=choice,
                                                            llm_tokenizer=tokenizer)
            all_log_prob_results.append(log_prob_results)

        choice_logprobs = [log_prob_results.answer_log_prob for log_prob_results in all_log_prob_results]
        prediction_label_id = int(np.argmax(choice_logprobs))
        label_id = choices.index(label)

        is_correct = label_id == prediction_label_id
        log_prob_results = all_log_prob_results[label_id]

        return is_correct, log_prob_results

    def terminate_and_save(self, predictions):

        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        # Save predictions
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{self.args.lnum}-{self.args.lname}-{self.args.rate}.p"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        # Save the summary
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.lnum}-{args.lname}-{args.rate}.pkl"

        results = self.dataset_metric.agg_to_dict()
        for k, v in self.args.__dict__.items():
            results["args/%s" % k] = v

        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        # Print final numbers and return
        self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")

    @staticmethod
    def get_acc_log_loss(predictions):

        acc = np.mean([1.0 if prediction["correct"] else 0.0 for prediction in predictions]) * 100.0
        log_loss = np.mean([-prediction["answer_logprob"]/float(prediction["answer_length"])
                            for prediction in predictions])

        return acc, log_loss

    @staticmethod
    def validate(predictions, split=0.2):

        val_size = int(split * len(predictions))
        validation_predictions = predictions[:val_size]
        test_predictions = predictions[val_size:]

        val_acc, val_logloss = AbstractExperiment.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = AbstractExperiment.get_acc_log_loss(test_predictions)

        return Results(val_acc=val_acc,
                       val_logloss=val_logloss,
                       test_acc=test_acc,
                       test_logloss=test_logloss)


if __name__ == '__main__':

    # Step 6: Run intervention
    base_results = None
    best_results = None
    best_lnum = None
    best_lname = None
    best_rate = None

    # for lnum in [-1, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
    for lnum in [-1, 31, 30, 29, 28, 27]:

        if lnum == -1:
            lnames = ["dont"]
            rates = [9.9]
        else:
            lnames = ["fc_in", "fc_out"]
            rates = [8.0, 9.0, 9.5, 9.9, 9.95]
            # rates = [1.0, 2.0, 4.0, 6.0, 8.0, 9.0, 9.5, 9.9, 9.95]

        for lname in lnames:
            for rate in reversed(rates):

                args.lnum = lnum
                args.lname = lname
                args.rate = rate
                model = deepcopy(base_model)
                predictions = experiment.intervene(model=model,
                                                   tokenizer=tokenizer,
                                                   dataset=dataset,
                                                   args=args,
                                                   llm_name=llm_name,
                                                   choices=choices)

                results = experiment.validate(predictions, split=0.2)

                if lname == "dont":
                    base_results = results
                    logger.log(f"Base Llama2 => {results.to_str()}")
                else:
                    logger.log(f"Llama2 => Layer number: {lnum}, Layer name {lname}, Rate {rate} => "
                               f"{results.to_str()}")
                    if best_results is None or \
                            (results.val_acc > best_results.val_acc) or \
                            (results.val_acc == best_results.val_acc and results.val_logloss < best_results.val_logloss):

                        best_results = results
                        best_lnum = lnum
                        best_lname = lname
                        best_rate = rate

                    logger.log(f"Base model results {base_results.to_str()}. "
                               f"Best results {best_results.to_str()} at "
                               f"layer: {best_lnum}, lname: {best_lnum}, rate: {best_rate}")
                    logger.log("=============")

    logger.log("Experimented Completed.")
