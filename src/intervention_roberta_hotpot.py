import os
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer

from dataset_utils.hotpot import Hotpot
from study_utils.log_utils import Logger
from transformers import RobertaForMaskedLM
from laser.LaserWrapper import LaserWrapper
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, beautify, Progress


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


class RobertaExperiment:

    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger

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

    def get_accuracy(self, batch, model_edit, tokenizer):

        prompts = []

        for dp in batch:
            question, answer = dp[0], dp[1]
            prompted_question = f"{question} <mask> <mask> <mask> <mask> <mask>"
            prompts.append(prompted_question)

        batch_token_ids_and_mask = tokenizer(prompts, return_tensors="pt", padding="longest").to(self.device)
        mask_token_id = tokenizer.mask_token_id

        # Generate log probabilities
        with torch.no_grad():
            logits = model_edit(**batch_token_ids_and_mask).logits  # batch x max_length x vocab
            argmax_tokens = logits.argmax(dim=2)  # batch x max_length
            max_len = argmax_tokens.shape[1]

        scores = []
        for i, dp in enumerate(batch):

            answer = dp[1]

            # Find argmax tokens that correspond to mask token id
            token_ids = []
            for j in range(0, max_len):
                if int(batch_token_ids_and_mask.input_ids[i, j]) == mask_token_id:
                    token_ids.append(argmax_tokens[i, j].item())

            generation = tokenizer.decode(token_ids)

            # We compute 0-1 match, f1, precision, and recall score in addition to log-prob of the answer tokens
            is_correct = self.metrics.generation_match(generation=generation, answer=answer)
            f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)

            scores.append((is_correct, f1pr_score, generation))

        return scores

    def get_choice_accuracy(self, batch, model_edit, choices, tokenizer):

        choice_log_probs = [[] for _ in batch]

        for choice in choices:

            choice_batch = [(dp[0], choice) for dp in batch]
            choice_batch_log_prob_results = self.get_log_prob(choice_batch, model_edit, tokenizer)

            for i, results in enumerate(choice_batch_log_prob_results):
                choice_log_probs[i].append(results)

        scores = []
        batch_log_prob_results = []

        for i, (question, answer) in enumerate(batch):
            assert answer in choices
            assert len(choice_log_probs[i]) == len(choices)

            gold_answer_ix = choices.index(answer)

            answer_log_probs = [log_prob_results_.answer_log_prob for log_prob_results_, _ in choice_log_probs[i]]
            predicted_answer_ix = np.argmax(answer_log_probs)

            is_correct = gold_answer_ix == predicted_answer_ix
            scores.append((is_correct, None, None))

            # Use log-results of the correct answer for computing log-prob of the answer
            batch_log_prob_results.append(choice_log_probs[i][gold_answer_ix])

        return scores, batch_log_prob_results

    def _to_mask(self, batch_token_ids_and_mask, batch, tokenizer):

        masked_token_ids = deepcopy(batch_token_ids_and_mask)

        for i, (question, answer) in enumerate(batch):
            # Find the answer tokens and mask them
            prompt_len = batch_token_ids_and_mask.attention_mask[i].sum()            # max_length
            answer_len = self.metrics.find_answer_len(batch_token_ids_and_mask.input_ids[i][:prompt_len], answer, tokenizer)
            masked_token_ids.input_ids[i][:prompt_len][-answer_len:] = tokenizer.mask_token_id

        return masked_token_ids

    def get_log_prob(self, batch, model_edit, tokenizer):

        claims = []

        for dp in batch:
            question, answer = dp[0], dp[1]
            claim = f"{question} {answer}"
            claims.append(claim)

        batch_token_ids_and_mask = tokenizer(claims,
                                             return_tensors="pt",
                                             padding="longest",
                                             add_special_tokens=False).to(self.device)

        # Replace the answers with mask_token_id
        masked_batch_token_ids_and_mask = self._to_mask(batch_token_ids_and_mask, batch, tokenizer)

        # Generate log probabilities
        with torch.no_grad():
            logits = model_edit(**masked_batch_token_ids_and_mask).logits  # batch x max_length x vocab
            log_prob = torch.log_softmax(logits, dim=2)                    # batch x max_length x vocab

        batch_log_prob_results = []
        for i in range(len(batch)):

            prompt_len = batch_token_ids_and_mask.attention_mask[i].sum()            # max_length

            # Compute logprob
            log_prob_results = self.metrics.masked_answer_log_prob(
                log_prob=log_prob[i, :prompt_len],
                question_answer_token_ids=batch_token_ids_and_mask.input_ids[i, :prompt_len],
                masked_question_answer_token_ids=masked_batch_token_ids_and_mask.input_ids[i, :prompt_len],
                tokenizer=tokenizer)

            batch_log_prob_results.append((log_prob_results, prompt_len))

        return batch_log_prob_results

    def intervene(self, model, tokenizer, dataset, args, llm_name, choices):

        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")

        time_edit_start = time.time()
        model_edit = LaserWrapper.get_edited_model(model=model,
                                                   lname=args.lname,
                                                   lnum=args.lnum,
                                                   rate=args.rate,
                                                   intervention=args.intervention,
                                                   logger=self.logger,
                                                   in_place=True)

        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

        predictions = []

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        for i in tqdm(range(0, dataset_size, args.batch_size)):

            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            # Prepare questions
            my_batch_size = min(args.batch_size, dataset_size - i)
            batch = dataset[i: i + my_batch_size]

            # Get accuracy
            if choices is None:
                # Do generation and compute generation accuracy. Works for open-ended domains
                batch_scores = self.get_accuracy(batch, model_edit, tokenizer)

                # Do log-prob of the answer
                batch_log_prob_results = self.get_log_prob(batch, model_edit, tokenizer)
            else:
                # Compute accuracy in classification mode
                batch_scores, batch_log_prob_results = self.get_choice_accuracy(batch, model_edit, choices, tokenizer)

                # Do log-prob of the selected answer

            for j in range(my_batch_size):

                question, answer = batch[j][0], batch[j][1]

                is_correct, f1pr_score, generation = batch_scores[j]
                self.dataset_metric.accept(is_correct=is_correct,
                                           f1pr_score=f1pr_score,
                                           log_prob_results=batch_log_prob_results[j][0],
                                           )

                if (i + j) % 1000 == 0:
                    print(f"Question: {question} and gold answer {answer}. Generation {generation}.")

                predictions_ = {
                    "ix": i + j,
                    "question": question,
                    "gold-answer": answer,
                    "generation": generation,
                    "correct": is_correct,
                    "f1_score": None if f1pr_score is None else f1pr_score.f1,
                    "precision": None if f1pr_score is None else f1pr_score.precision,
                    "recall": None if f1pr_score is None else f1pr_score.recall,
                    "case-sensitive": False,  # We ignore case when checking answer
                    "white-space-strip": True,  # We ignore white space when checking answer
                    "answer_logprob": batch_log_prob_results[j][0].answer_log_prob,
                    "answer_length": batch_log_prob_results[j][0].answer_len,
                    "question_answer_length": batch_log_prob_results[j][1]
                }
                predictions.append(predictions_)

        self.terminate_and_save(predictions)

        return predictions

    def terminate_and_save(self, predictions):

        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        # Save predictions
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        # Save the summary
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
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

        val_acc, val_logloss = RobertaExperiment.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = RobertaExperiment.get_acc_log_loss(test_predictions)

        return Results(val_acc=val_acc,
                       val_logloss=val_logloss,
                       test_acc=test_acc,
                       test_logloss=test_logloss)


if __name__ == '__main__':

    # Step 1: Command line argument
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with Roberta LLM on CounterFact')

    parser.add_argument('--st', type=int, default=0, help='0,14 27# samples per instruction')
    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="dropout",
                        choices=['dropout', 'rank-reduction'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None', 'dont'],
                        help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=12, help='Layers to edit', choices=list(range(0, 13)))
    parser.add_argument('--model_path',
                        type=str,
                        default="/home/dimisra/llama/Llama-2-7b-hf",
                        help="Place where model weights are stored")
    parser.add_argument('--home_dir', type=str,
                        default="./iclr2024/hotpot-fixed/",
                        help='Directory where the data is')

    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    base_model = RobertaForMaskedLM.from_pretrained(llm_name)

    # Step 3: Create save directory and logger
    home_dir = args.home_dir

    save_dir = f"{home_dir}/{llm_name}/{args.intervention}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log.txt")

    # Step 4: Create an experiment
    experiment = RobertaExperiment(save_dir=save_dir, logger=logger)

    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
    dataset_util = Hotpot(llama_tokenizer_path=args.model_path)      # We use the LLAMA tokenizer for consistency
    dataset = dataset_util.get_dataset(logger)

    filtered_dataset = []
    for dp in dataset:
        question, answer = dp["question"], dp["answer"]
        if not question.endswith("?") and not question.endswith("."):
            prompted_question = f"{question}? The answer is"
        else:
            prompted_question = f"{question} The answer is"
        filtered_dataset.append((prompted_question, answer))
    choices = None

    # Step 6: Run intervention
    base_results = None
    best_results = None
    best_lnum = None
    best_lname = None
    best_rate = None

    for lnum in [-1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:

        if lnum == -1:
            lnames = ["dont"]
            rates = [9.9]
        else:
            lnames = ["fc_in", "fc_out"]
            rates = [1.0, 2.0, 4.0, 6.0, 8.0, 9.0, 9.5, 9.9, 9.95]

        for lname in lnames:
            for rate in reversed(rates):

                args.lnum = lnum
                args.lname = lname
                args.rate = rate
                model = deepcopy(base_model)
                predictions = experiment.intervene(model=model,
                                                   tokenizer=tokenizer,
                                                   dataset=filtered_dataset,
                                                   args=args,
                                                   llm_name=llm_name,
                                                   choices=choices)

                results = experiment.validate(predictions, split=0.2)

                if lname == "dont":
                    base_results = results
                    logger.log(f"Base Roberta => {results.to_str()}")
                else:
                    logger.log(f"Roberta => Layer number: {lnum}, Layer name {lname}, Rate {rate} => "
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
