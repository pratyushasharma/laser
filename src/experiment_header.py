import os
import argparse

from dataset_utils.dataset_wrapper import DatasetUtil
from llm_wrapper import LLMWrapper
from study_utils.log_utils import Logger


class ExperimentSetup:

    def __init__(self, args, save_path, base_llm, logger):

        self.args = args
        self.save_path = save_path
        self.base_llm = base_llm
        self.logger = logger


class ExperimentHeader:

    def __init__(self):
        pass

    def generate_header(self):

        # Parse the command line arguments
        args = self.generate_parser_args()

        # Create save paths
        home_dir = args.home_dir
        split = args.split

        save_dir = f"{home_dir}/{split}/{args.llm_name}/{args.intervention}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create logger
        logger = Logger(save_dir=save_dir, fname=f"{args.llm_name}-log.txt")

        logger.log(f"Created a logger. Save director {save_dir} and log name {args.llm_name}-log.txt")
        logger.log("=" * 50)
        for k, v in args.__dict__.items():
            logger.log(f">>>> Command line argument {k} => {v}")
        logger.log("=" * 50)

        # Create base LLMs
        base_llm = LLMWrapper().get_llm_and_tokenizer(args.llm_name)
        logger.log(f"Created LLM {args.llm_name} and put it on device {args.device}")

        # Create the evaluation dataset
        dataset = DatasetUtil().get_dataset(dataset_name=args.dataset,
                                            logger=logger,
                                            split=args.split)

        logger.log(f"Created Dataset {args.dataset} (split: {args.split})")

        return ExperimentSetup(args=args,
                               save_path=save_dir,
                               base_llm=base_llm,
                               dataset=dataset,
                               logger=logger)

    @staticmethod
    def generate_parser_args():

        parser = argparse.ArgumentParser(description="Arguments for experiments on evaluating LLMs with intervention.")

        # LLM hyperparameters
        parser.add_argument("--llm", type=str, default="llama2-7", help="name of the LLM")
        parser.add_argument("--model_path",
                            type=str,
                            default="/home/dimisra/llama/Llama-2-7b-hf",
                            help="Place where model weights are stored")

        # Intervention hyperparameters
        parser.add_argument("--intervention", type=str, default="LASER",
                            choices=["zero", "pruning", "LASER"],
                            help="what type of intervention to perform. Zero just zero out the matrix. "
                                 "Pruning zero's out values of the chosen weight matrix(ces) with the "
                                 "k-smallest absolute values. LASER performs replaces chosen weight matrix(ces) with "
                                 "their low-rank approximation.")

        # For LASER hyperparameters
        parser.add_argument("--lname", type=str, default="None",
                            choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None',
                                     'dont'],
                            help="provided which type of parameters to effect")
        parser.add_argument("--lnum", type=int, default=27, help='Layers to edit', choices=list(range(-1, 28)))
        parser.add_argument("--rho", type=float, default=1, help='rates for intervention')
        parser.add_argument("--in_place", action="store_true", help="if true, then interventions "
                                                                    "are directly done on the model")

        # Hyperparameters related to memory and speed
        parser.add_argument("--compress", action="store_true", help="if true, then compress matrices")

        # Dataset hyperparameters
        parser.add_argument("--dataset", type=str, default="causal_judgement", help="dataset name to run on")

        # TODO Make it None to mean all splits or something
        parser.add_argument("--split", type=str, default="causal_judgement", help="split of the dataset")

        # Evaluation hyperparameters
        parser.add_argument('--sample_size', type=int, default=817, help='# samples per instruction')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation')
        parser.add_argument('--max_len', type=int, default=10, help='maximum length for generation')
        parser.add_argument('--k', type=int, default=10, help='top k for evaluation')

        # Logging hyperparameters
        parser.add_argument('--home_dir', type=str,
                            default="./iclr2024/big_bench/",
                            help='Directory where the data is')

        args = parser.parse_args()

        return args
