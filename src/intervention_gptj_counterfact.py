import os
import time
import torch
import pickle
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress


class GPTJExperiment:

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

            question, answer = dataset[i]
            # Given that we do 1-token look up we do the following:
            # - Compute log-prob of the gold token
            # - Compute top-1, top-5 and top-10 accuracies

            inputs = tokenizer(question, return_tensors="pt").to(self.device)
            gold_answer_token_ids = tokenizer(answer)["input_ids"]
            answer_len = len(gold_answer_token_ids)
            assert answer_len == 1, f"For GPTJ+CounterFact special case, we assume the answer " \
                                    f"has 1 token. Found {gold_answer_token_ids}."
            gold_answer_token_id = int(gold_answer_token_ids[0])

            with torch.no_grad():
                # Compute log probability of question
                results = model_edit(inputs.input_ids)
                logits = results.logits[0]                                      # question length x vocab
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)       # question length x vocab

                last_token_logprob = log_prob[-1]                               # vocab
                answer_log_prob = last_token_logprob[gold_answer_token_id].item()

                sorted_logprob, sorted_indices = torch.sort(last_token_logprob, descending=True)

                top_k_logprob = sorted_logprob[:10].detach().cpu().numpy()
                top_k_indices = sorted_indices[:10].detach()

                decoded_tokens = tokenizer.batch_decode(top_k_indices)
                top_k_tokens = [token for token in decoded_tokens]
                assert len(top_k_tokens) == 10

                is_correct = answer.lower().strip() == top_k_tokens[0].lower().strip()
                top_1_acc = float(is_correct)
                top_5_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:5]])
                top_10_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:10]])

                # Compute log-prob of question and answer
                selected_log_prob = log_prob[:-1, :]  # question - 1 x vocab
                indices = inputs.input_ids[0, 1:].unsqueeze(1)  # question - 1 x 1

                selected_log_prob = torch.gather(selected_log_prob,
                                                 index=indices,
                                                 dim=1)  # question - 1 x 1
                question_log_prob = selected_log_prob.sum().item()
                total_log_prob = question_log_prob + answer_log_prob

                logprob_results = ContextAnswerLogProb(total_log_prob=total_log_prob,
                                                       answer_log_prob=answer_log_prob,
                                                       answer_len=answer_len)

            self.dataset_metric.accept(is_correct=is_correct,
                                       f1pr_score=None,
                                       log_prob_results=logprob_results,
                                       top_k_acc={1: top_1_acc, 5: top_5_acc, 10: top_10_acc})

            if i % 10 == 0:
                print(f"Question: {question} and gold answer {answer}")

            predictions_ = {
                "ix": i,
                "question": question,
                "gold-answer": answer,
                "generation": top_k_tokens[0],      # We can view the top token as the 1-step generation
                "correct": is_correct,
                "top_1_acc": top_1_acc,
                "top_5_acc": top_5_acc,
                "top_10_acc": top_10_acc,
                "top_10_logprob": top_k_logprob,
                "top_10_tokens": top_k_tokens,
                "f1_score": None,
                "precision": None,
                "recall": None,
                "case-sensitive": self.case_sensitive,        # We ignore case when checking answer
                "white-space-strip": self.strip,              # We ignore white space when checking answer
                "total_logprob": total_log_prob,
                "question_logprob": question_log_prob,
                "answer_logprob": answer_log_prob,
                "answer_length": answer_len,
                "question_answer_length": inputs.input_ids.shape[1] + answer_len
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
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with GPTJ LLM on CounterFact')

    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=1, help='maximum length for generation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction', 'zero'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None', 'dont',
                                 "all", "mlp", "attn"],
                        help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=24, help='Layers to edit', choices=list(range(-1, 28)))
    parser.add_argument('--model_path',
                        type=str,
                        default="/mnt/data/Llama2/Llama-2-7b-hf",
                        help="Place where model weights are stored")
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/counterfact/gptj_results",
                        help='Directory where the data is')
    parser.add_argument('--dataset_file', type=str,
                        default="/mnt/data/counterfact",
                        help='Directory where the data is')

    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = "GPTJ"
    llm_path = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = GPTJForCausalLM.from_pretrained(
        llm_path,
        revision="float16",
        torch_dtype=torch.float16
    )

    # Step 3: Create save directory and logger
    home_dir = args.home_dir
    dataset_loc = args.dataset_file

    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    # Step 4: Create an experiment
    experiment = GPTJExperiment(save_dir=save_dir, logger=logger)

    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
    with open(args.dataset_file, "rb") as f:
        data = pickle.load(f)

    num_dp = len(data)
    dataset = []

    for i in range(num_dp):
        question = data[i]["question"]
        answer = data[i]["gold-answer"]
        assert answer.startswith(" "), f"Found answer that doesn't start with space ${answer}$"
        dataset.append((question, answer))
    logger.log(f"Read dataset of size {num_dp}")

    # Step 6: Run intervention
    experiment.intervene(model=model,
                         tokenizer=tokenizer,
                         dataset=dataset,
                         args=args,
                         llm_name=llm_name)

    logger.log("Experimented Completed.")