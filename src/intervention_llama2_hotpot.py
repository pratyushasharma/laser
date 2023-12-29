import os
import time
import torch
import pickle
import argparse

from tqdm import tqdm
from datasets import load_dataset
from transformers import LlamaTokenizerFast
from transformers import LlamaForCausalLM
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics
from study_utils.time_utils import elapsed_from_str, Progress


class LLAMA2Experiment:

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

    def intervene(self, model, tokenizer, dataset, args, llm_name):

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

        for i in tqdm(range(0, dataset_size)):

            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            question = dataset[i]["question"]

            if not question.endswith("?") and not question.endswith("."):
                prompted_question = f"{question}? The answer is"
            else:
                prompted_question = f"{question} The answer is"

            answer = dataset[i]["answer"]
            inputs = tokenizer(prompted_question, return_tensors="pt").to(self.device)
            input_and_answer = tokenizer(prompted_question + " " + answer, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Generate from the model
                generate_ids = model_edit.generate(inputs.input_ids,
                                                   max_new_tokens=args.max_len,
                                                   min_new_tokens=1)

                generation = tokenizer.batch_decode(generate_ids,
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)[0]

                # Compute log probability of question + answer
                results = model_edit(input_and_answer.input_ids)
                logits = results.logits[0]                                      # question + answer length x vocab
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)       # question + answer length x vocab

                log_prob_results = self.metrics.answer_log_prob(log_prob=log_prob,
                                                                question_answer_token_ids=input_and_answer.input_ids[0],
                                                                answer=answer,
                                                                llm_tokenizer=tokenizer)

            # We compute 0-1 match, f1, precision, and recall score in addition to log-prob of the answer tokens
            is_correct = self.metrics.generation_match(generation=generation, answer=answer)
            f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)

            self.dataset_metric.accept(is_correct=is_correct,
                                       f1pr_score=f1pr_score,
                                       log_prob_results=log_prob_results)

            if i % 10 == 0:
                print(f"Question: {prompted_question} and gold answer {answer}")
                print(f"{llm_name} generated {generation}")

            predictions_ = {
                "ix": i,
                "question": question,
                "prompted_question": prompted_question,
                "gold-answer": answer,
                "generation": generation,
                "correct": is_correct,
                "f1_score": f1pr_score.f1,
                "precision": f1pr_score.precision,
                "recall": f1pr_score.recall,
                "case-sensitive": self.case_sensitive,        # We ignore case when checking answer
                "white-space-strip": self.strip,              # We ignore white space when checking answer
                "total_logprob": log_prob_results.total_log_prob,
                "answer_logprob": log_prob_results.answer_log_prob,
                "answer_length": log_prob_results.answer_len,
                "question_answer_length": input_and_answer.input_ids.shape[1]
            }
            predictions.append(predictions_)

        # Save results and terminate
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
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with LLAMA 2 LLM on Hotpot')

    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=15, help='maximum length for generation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out',
                                 'None', 'dont', 'all', 'mlp', 'attn'],
                        help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=28, help='Layers to edit', choices=list(range(-1, 33)))
    parser.add_argument('--model_path',
                        type=str,
                        default="/mnt/data/Llama2/Llama-2-7b-hf",
                        help="Place where model weights are stored")
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/hotpot/llama2_results",
                        help='Directory where the data is')
    parser.add_argument('--dataset_file', type=str,
                        default="None",
                        help='Directory where the data is')

    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = "Llama2-7G"
    llm_path = args.model_path
    tokenizer = LlamaTokenizerFast.from_pretrained(llm_path)
    model = LlamaForCausalLM.from_pretrained(llm_path)

    # Step 3: Create save directory and logger
    home_dir = args.home_dir
    dataset_loc = args.dataset_file

    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    # Step 4: Create an experiment
    experiment = LLAMA2Experiment(save_dir=save_dir, logger=logger)

    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
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
        num_tokens = len(tokenizer(answer).input_ids)
        if num_tokens <= 15:
            train.append({"question": question,
                          "answer": answer})

    validation = []
    for dp in full_dataset["validation"]:
        question = dp["question"].strip()
        answer = dp["answer"].strip()
        num_tokens = len(tokenizer(answer).input_ids)
        if num_tokens <= 15:
            validation.append({"question": question,
                               "answer": answer})

    dataset = train + validation
    num_dp = len(dataset)
    logger.log(f"Read dataset of size {num_dp} of which the first {len(train)} examples are from the "
               f"train set and the remaining {len(validation)} from the validation split.")

    # Step 6: Run intervention
    experiment.intervene(model=model,
                         tokenizer=tokenizer,
                         dataset=dataset,
                         args=args,
                         llm_name=llm_name)

    logger.log("Experimented Completed.")
