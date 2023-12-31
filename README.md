# LASER: Layer Selective Rank Reduction

This repository contains code for the LASER paper _"The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction"_, Pratyusha Sharma, Jordan T. Ash, and Dipendra Misra, [arXiv 2023](https://arxiv.org/pdf/2312.13558.pdf). 

**Website:** [https://pratyushasharma.github.io/laser/](https://pratyushasharma.github.io/laser/)

**This is an early development release. We will do a major refactoring in the first half of Jan 2024, to make the code more easy to use, and extend to other settings.** 

We welcome issues and pull requests. If you report a new result using LASER on a given LLM and NLP task, then please feel free to issue a pull request and we will add it to the leaderboard on the website and acknowledge the result to you.

## What is LASER?

LASER stands for **LA**yer **SE**lective **R**ank-Reduction, and is an intervention where we replace a selected weight matrix in the transformer architecture of an LLM with its low-rank approximation. A single LASER transformation consists of 3 hyperparameters: the layer number to modify (&ell;) such as 16th layer, the parameter type (&tau;) such as the first MLP layer, and the fraction of the maximum rank to retain (&rho;) such as 0.01 fraction of the rank. We can write this transformation as (&ell;, &tau;, &rho;) and we can stack these transformations and apply them in parallel. The low-rank approximation is performed using SVD. Figure below from our paper shows an illustration.

![LASER illustration](https://pratyushasharma.github.io/laser/main.png)


LASER modifies the weight matrix, thereby, allowing us to see the changes as a result and build understanding of what these matrices contain about the problem. We originally wanted to prod the transformer architecture this way, but to our surprise found significant gains in accuracy on various LLM tasks, when these reductions were performed. The paper above presents various results related to evaluating LASER on 3 different LLMs and several LLM benchmark. This repository contains the code to reproduce these results.

## How to run a sample code

We first discuss installing the code and then discuss how to run an experiment.

To install the experiment, please install the pip file. We chiefly just need pytorch and the datasets and transformers package from huggingface. It might be a good idea to create a conda environment.

> pip3 install -r requirements.txt

### Installation

At the moment, each setup is its own file. To run an experiment that performs a single LASER transformer to GPTJ on the Counterfact dataset, you can run:
 
> python3 intervention_gptj_counterfact.py --lname fc_in --rate 9.9 --lnum 26

here _lnum_ is &ell;, _lname_ is &tau;, and _rate_ is related to &rho; by &rho; = 1 - 0.1 * rate. The rate is a value between [0, 10.0] and measures how much rank to retain. The use of rate is for legacy reasons and we will refactor the code to directly use &rho; in the future. 

Note that the above experiments will save accuracies and log-losses for each datapoint. In some files, one has to take the validation set (first 20% examples) and do hyperparameter selection separately, and then compute the accuracy on the test set (remaining 80% examples) with the chose hyperparameters. In the future, we will refactor the code to make this very easy to do.

## Code Organization

Code is inside the `src` folder. The main experiment files are top-level inside the `src`. The filename convention is `intervention_<llm-name>_<dataset-name>.py` where `<llm-name>` is the name of the LLM and `<dataset-name>` is the name of the dataset. For BigBench, the dataset split is often specified with an additional flag --split. Please see the codebase for details of command line arguments. We will provide a comprehensive tutorial later.

The code for performing laser is inside the `laser` package. We use PyTorch to do SVD and compute low-rank approximation. The code for low-rank approximation happens [here](https://github.com/pratyushasharma/laser/blob/main/src/laser/matrix_utils.py#L39). The code for reading and processing dataset is inside `dataset_util`. Finally, metrics and logging are done using the `study_utils`.  

## Citation

If you find this codebase useful, then please cite the following paper. Additionally, feel free to send a PR or an email and we will cite your result/paper on the leaderboard.

> @article{sharma2023truth,
> 
>  title={The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction},
>
>  author={Sharma, Pratyusha and Ash, Jordan T and Misra, Dipendra},
>
> journal={arXiv preprint arXiv:2312.13558},
>
>   year={2023}
> }
