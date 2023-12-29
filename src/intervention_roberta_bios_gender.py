import os
import time
import torch
import pickle
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer

from study_utils.log_utils import Logger
from transformers import RobertaForMaskedLM
from laser.LaserWrapper import LaserWrapper
from dataset_utils.bias_in_bios import BiasBiosGender
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, beautify, Progress


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

    def intervene(self, model, tokenizer, dataset, args, logger):

        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")

        time_edit_start = time.time()
        model_edit = LaserWrapper.get_edited_model(model=model,
                                                   lname=args.lname,
                                                   lnum=args.lnum,
                                                   rate=args.rate,
                                                   intervention=args.intervention,
                                                   logger=logger,
                                                   in_place=True)

        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

        predictions = []

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        male_token_id = tokenizer(" male")["input_ids"]      # TODO check for space
        assert len(male_token_id) == 3 and male_token_id[0] == 0 and male_token_id[2] == 2
        male_token_id = male_token_id[1]

        female_token_id = tokenizer(" female")["input_ids"]    # TODO check for space
        assert len(female_token_id) == 3 and female_token_id[0] == 0 and female_token_id[2] == 2
        female_token_id = female_token_id[1]

        for i in tqdm(range(0, dataset_size, args.batch_size)):

            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            # Prepare questions
            my_batch_size = min(args.batch_size, dataset_size - i)
            batch = dataset[i: i + my_batch_size]
            batch_token_ids_and_mask = tokenizer([question for question, _ in batch],
                                                 return_tensors="pt", padding="longest").to(self.device)

            # Find position of the masked_token_id
            mask_token_flag = \
                (batch_token_ids_and_mask["input_ids"] == tokenizer.mask_token_id).float()         # batch x max_length
            assert (mask_token_flag.sum(1) == 1.0).all().item()
            mask_token_ids = mask_token_flag.argmax(dim=1)                                         # batch

            # Prepare gold answers
            gold_answers = [gold_answer if gold_answer.startswith(" ") else f" {gold_answer}" for _, gold_answer in batch]

            # batch_gold_answer_token_ids = []
            # for gold_answer in gold_answers:
            #     gold_answer_token_ids = tokenizer(gold_answer)["input_ids"]
            #     if not (len(gold_answer_token_ids) == 3 and
            #             gold_answer_token_ids[0] == 0 and
            #             gold_answer_token_ids[2] == 2):
            #         raise AssertionError(f"Gold answer {gold_answer} has tokens {gold_answer_token_ids}")
            #     batch_gold_answer_token_ids.append(gold_answer_token_ids[1])
            #
            # batch_gold_answer_token_ids = torch.LongTensor(batch_gold_answer_token_ids).unsqueeze(1).to(self.device)  # batch x 1

            # if torch.cuda.is_available():
            #     batch_token_ids_and_mask = {k: v.cuda() for k, v in batch_token_ids_and_mask.items()}
            #     batch_gold_answer_token_ids = batch_gold_answer_token_ids.cuda()

            # Generate log probabilities over masked tokens, 1 per data point
            with torch.no_grad():
                logits = model_edit(**batch_token_ids_and_mask).logits       # batch x max_length x vocab
                logprob = torch.log_softmax(logits, dim=2)                   # batch x max_length x vocab

            vocab_size = logprob.shape[2]
            mask_token_ids = mask_token_ids.view(my_batch_size, 1, 1)
            mask_token_ids = mask_token_ids.expand([my_batch_size, 1, vocab_size])

            predicted_logprob = torch.gather(logprob, index=mask_token_ids, dim=1)     # batch size x 1 x vocab_size
            predicted_logprob = predicted_logprob[:, 0, :]                             # batch x vocab_size

            # Generate top-k tokens
            sorted_logprob, sorted_indices = torch.sort(predicted_logprob, descending=True)    # both are batch x vocab_size
            sorted_logprob = sorted_logprob[:, :args.k].detach().cpu().numpy()                    # batch x k
            sorted_indices = sorted_indices[:, :args.k].detach().cpu().numpy()                    # batch x k

            # Compute top-k accuracy
            batch_top_10_tokens = [
                [tokenizer.decode(sorted_indices[j, l]).lower().strip() for l in range(10)]
                for j in range(my_batch_size)
            ]

            batch_top_1_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:1]
                                    for j in range(my_batch_size)]
            batch_top_5_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:5]
                                    for j in range(my_batch_size)]
            batch_top_10_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:10]
                                     for j in range(my_batch_size)]

            # Compute log_prob using the probability of gold tokens
            # gold_log_prob = torch.gather(predicted_logprob, index=batch_gold_answer_token_ids, dim=1)[:, 0]   # batch

            # Compute log_prob using the probability of gold tokens
            batch_male_token_ids = torch.LongTensor([male_token_id] * my_batch_size).unsqueeze(1).to(self.device)
            male_log_prob = torch.gather(predicted_logprob, index=batch_male_token_ids, dim=1)[:, 0]   # batch

            # Compute log_prob using the probability of gold tokens
            batch_female_token_ids = torch.LongTensor([female_token_id] * my_batch_size).unsqueeze(1).to(self.device)
            female_log_prob = torch.gather(predicted_logprob, index=batch_female_token_ids, dim=1)[:, 0]   # batch

            # Compute perplexity
            for j in range(my_batch_size):

                if batch[j][1] == "male":
                    is_correct = male_log_prob[j].item() > female_log_prob[j].item()
                    answer_log_prob = male_log_prob[j].item()
                    answer_token_id = int(male_token_id)
                else:
                    assert batch[j][1] == "female", f"must be male or female. Found '{batch[j][1]}'"
                    is_correct = female_log_prob[j].item() > male_log_prob[j].item()
                    answer_log_prob = female_log_prob[j].item()
                    answer_token_id = int(female_token_id)

                # Update the accuracy metric
                answer_len = 1
                logprob_results = ContextAnswerLogProb(total_log_prob=None,
                                                       answer_log_prob=answer_log_prob,
                                                       answer_len=answer_len)

                self.dataset_metric.accept(is_correct=is_correct,
                                           f1pr_score=None,
                                           log_prob_results=logprob_results,
                                           top_k_acc={1: batch_top_1_accuracy[j],
                                                      5: batch_top_5_accuracy[j],
                                                      10: batch_top_10_accuracy[j]})

                if (i + j) % 1000 == 0:
                    print(f"Question: {batch[j][0]} and gold answer {batch[j][1]}. "
                          f"Predicted top-10 tokens {batch_top_10_tokens[j]}.")

                predictions_ = {
                    "ix": i + j,
                    "question": batch[j][0],
                    "gold-answer": batch[j][1],
                    "answer_token_id": answer_token_id,
                    "correct": is_correct,
                    "case-sensitive": False,        # We ignore case when checking answer
                    "white-space-strip": True,      # We ignore white space when checking answer
                    "predicted-topk-logprob": sorted_logprob[j],
                    "predicted-topk-token-id": sorted_indices[j],
                    "predicted-topk-tokens": batch_top_10_tokens[j],
                    "male_log_prob": male_log_prob[j].item(),
                    "female_log_prob": female_log_prob[j].item(),
                    "answer_logprob": answer_log_prob,
                    "answer_length": answer_len
                }
                predictions.append(predictions_)

        self.terminate_and_save(predictions)

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
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/bios_gender/roberta_results",
                        help='Directory where the data is')
    parser.add_argument('--dataset_file', type=str,
                        default="./counterfact",
                        help='Directory where the data is')

    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = RobertaForMaskedLM.from_pretrained(llm_name)

    # Step 3: Create save directory and logger
    home_dir = args.home_dir
    dataset_loc = args.dataset_file

    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    # Step 4: Create an experiment
    experiment = RobertaExperiment(save_dir=save_dir, logger=logger)

    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Create dataset
    dataset_util = BiasBiosGender()
    dataset = dataset_util.get_dataset(logger)

    processed_data = []
    for dp in dataset:
        question = dp["hard_text"]

        # Answer is either 0 (male) or 1 (female)
        answer_ix = dp["answer"]
        assert answer_ix in [0, 1]
        if answer_ix == 0:
            answer = "male"
        else:
            answer = "female"

        max_len = 50
        question_token_ids = tokenizer(question, add_special_tokens=False)["input_ids"][-max_len:]
        assert len(question_token_ids) <= max_len
        question = tokenizer.decode(question_token_ids, skip_special_tokens=True)

        # Given that we do 1-token look up we do the following:
        # - Compute log-prob of the gold token
        # - Compute top-1, top-5 and top-10 accuracies
        if question.strip().endswith(".") or question.strip().endswith("?"):
            prompted_question = "Consider the following text: " + question.strip() + \
                                " Is the person in this text a male or female? This person is <mask>."
        else:
            prompted_question = "Consider the following text: " + question.strip() + \
                                ". Is the person in this text a male or female? This person is <mask>."

        processed_data.append((prompted_question, answer))

    # Run intervention
    logger.log("Starting experiments!")
    experiment.intervene(model=model,
                         tokenizer=tokenizer,
                         dataset=processed_data,
                         args=args,
                         logger=logger)
