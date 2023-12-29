import nltk
import math
import torch


class DatasetMetrics:

    CORRECTNESS = "0-1 correctness"
    AvgF1Score = "avg f1 score"
    MeanLogProb = "mean log prob"
    PERPLEXITY = "perplexity"
    DatasetSize = "dataset_size"
    LogProbExamples = "num log probs"
    TotalAnswerTokens = "total answer tokens"
    TOPK = "top-"

    def __init__(self, logger):

        self.logger = logger

        self.num_examples = 0
        self.num_logprob_examples = 0
        self.num_correct = 0.0
        self.sum_f1_score = 0.0
        self.sum_total_log_prob = 0.0
        self.sum_mean_log_prob = 0.0
        self.total_answer_words = 0.0
        self.total_top_k_acc = dict()
        self._terminate = False

    def reset(self):

        self.num_examples = 0
        self.num_logprob_examples = 0
        self.num_correct = 0.0
        self.sum_f1_score = 0.0
        self.sum_total_log_prob = 0.0
        self.sum_mean_log_prob = 0.0
        self.total_answer_words = 0.0
        self.total_top_k_acc = dict()
        self._terminate = False

    def accept(self, is_correct, f1pr_score, log_prob_results, top_k_acc=None):
        """
        :param is_correct: True if exact match was correct and False otherwise
        :param f1pr_score: An F1PR score object
        :param log_prob_results:  A ContextAnswerLogProb object
        :return:
        """

        if self._terminate:
            raise AssertionError("Cannot add entries to a terminated dataset metric. "
                                 "This means this metric is finalized.")

        self.num_examples += 1
        self.num_correct += (1.0 if is_correct else 0.0)

        if f1pr_score is None:
            self.sum_f1_score = None
        else:
            self.sum_f1_score += f1pr_score.f1

        if log_prob_results is not None:
            if type(log_prob_results) == list:
                self.num_logprob_examples += len(log_prob_results)
                for log_prob_results_ in log_prob_results:
                    self.sum_total_log_prob += log_prob_results_.answer_log_prob
                    self.sum_mean_log_prob += (
                                log_prob_results_.answer_log_prob / float(max(1, log_prob_results_.answer_len)))
                    self.total_answer_words += log_prob_results_.answer_len
            else:
                self.num_logprob_examples += 1
                self.sum_total_log_prob += log_prob_results.answer_log_prob
                self.sum_mean_log_prob += (
                        log_prob_results.answer_log_prob / float(max(1, log_prob_results.answer_len)))
                self.total_answer_words += log_prob_results.answer_len

        if top_k_acc is not None:

            if len(self.total_top_k_acc) != 0:
                # Check that keys of self.total_top_acc and top_k_acc are the same
                assert self.total_top_k_acc.keys() == top_k_acc.keys(), \
                    f"Top k accuracy key set must be the same across runs. The total k accuracy so far has k values" \
                    f" {self.total_top_k_acc.keys()} but found the following k values " \
                    f"in the recent run {top_k_acc.keys()}."

            for k, v in top_k_acc.items():
                if k not in self.total_top_k_acc:
                    self.total_top_k_acc[k] = 0.0

                self.total_top_k_acc[k] += float(v)

    def terminate(self):
        self._terminate = True

    def print(self):
        results = self.agg_to_dict()

        prefix = f"Final Performance: Dataset size {self.num_examples}" \
            if self._terminate else f"After {self.num_examples}"

        if len(self.total_top_k_acc) == 0:
            top_k_results = "."
        else:
            top_k_results = ", ".join([f"{k} is {v}" for k, v in sorted(results.items())
                                       if k.startswith(DatasetMetrics.TOPK)])
            top_k_results = ", " + top_k_results + "."

        self.logger.log(f"{prefix} 0-1 Correctness is {results[DatasetMetrics.CORRECTNESS]} percentage, "
                        f"Mean F1 score is {results[DatasetMetrics.AvgF1Score]}, "
                        f"Mean Log Prob is {results[DatasetMetrics.MeanLogProb]}{top_k_results}")

    def agg_to_dict(self):

        # Compute aggregate dataset scores
        accuracy = (self.num_correct * 100.0) / float(max(1, self.num_examples))
        if self.sum_f1_score is None:
            avg_f1_score = None
        else:
            avg_f1_score = self.sum_f1_score / float(max(1, self.num_examples))
        mean_log_prob = self.sum_mean_log_prob / float(max(1, self.num_logprob_examples))
        perplexity = math.exp(- self.sum_total_log_prob / float(max(1, self.total_answer_words)))

        results = {
            DatasetMetrics.CORRECTNESS: accuracy,
            DatasetMetrics.AvgF1Score: avg_f1_score,
            DatasetMetrics.MeanLogProb: mean_log_prob,
            DatasetMetrics.PERPLEXITY: perplexity,
            DatasetMetrics.DatasetSize: self.num_examples,
            DatasetMetrics.TotalAnswerTokens: self.total_answer_words,
            DatasetMetrics.LogProbExamples: self.num_logprob_examples
        }

        for k, v in self.total_top_k_acc.items():
            top_k = (v * 100.0) / float(max(1, self.num_examples))
            results[DatasetMetrics.TOPK + f"{k} accuracy"] = top_k

        return results


class Metrics:

    def __init__(self, case_sensitive=False, strip=True):
        self.case_sensitive = case_sensitive
        self.strip = strip
        self.tokenizer = nltk.word_tokenize
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def _prepare(self, generation, answer):

        if self.strip:
            generation = generation.strip()
            answer = answer.strip()

        if not self.case_sensitive:
            generation = generation.lower()
            answer = answer.lower()

        return generation, answer

    def _to_bow(self, generation, answer):

        generation_tokens = self.tokenizer(generation)
        answer_tokens = self.tokenizer(answer)

        generation_token_set = set(generation_tokens)
        answer_token_set = set(answer_tokens)

        return generation_token_set, answer_token_set

    def exact_match(self, generation, answer):

        generation, answer = self._prepare(generation, answer)
        return answer == generation

    def generation_match(self, generation, answer):

        generation, answer = self._prepare(generation, answer)
        return answer in generation

    def precision(self, generation, answer):
        """
            :param generation:  A generated string containing only the newly generated tokens
            :param answer:  Answer string
            :return: fraction of unique tokens in generation that also in the answer
        """
        generation, answer = self._prepare(generation, answer)
        generation_token_set, answer_token_set = self._to_bow(generation, answer)

        return self._precision(generation_token_set, answer_token_set)

    def recall(self, generation, answer):
        """
            :param generation:  A generated string containing only the newly generated tokens
            :param answer:  Answer string
            :return: fraction of unique tokens in answer that also in the generation
        """
        generation, answer = self._prepare(generation, answer)
        generation_token_set, answer_token_set = self._to_bow(generation, answer)

        return self._recall(generation_token_set, answer_token_set)

    @staticmethod
    def _precision(generation_token_set, answer_token_set):
        # Compute what percentage of generation tokens are in answer tokens
        intersect = generation_token_set.intersection(answer_token_set)
        return float(len(intersect)) / float(max(1, len(generation_token_set)))

    @staticmethod
    def _recall(generation_token_set, answer_token_set):
        intersect = generation_token_set.intersection(answer_token_set)
        return float(len(intersect)) / float(max(1, len(answer_token_set)))

    def f1pr_scores(self, generation, answer):
        """
        :param generation:  A generated string containing only the newly generated tokens
        :param answer:  Answer string
        :param return_all: If True, also return precision and recall, otherwise, only return F1 score
        :return: return an object containing F1 score, precision and recall
        """

        generation, answer = self._prepare(generation, answer)
        generation_token_set, answer_token_set = self._to_bow(generation, answer)

        precision_score = self._precision(generation_token_set, answer_token_set)
        recall_score = self._recall(generation_token_set, answer_token_set)

        f1_score = (2 * precision_score * recall_score) / float(max(1, precision_score + recall_score))

        return F1PR(f1=f1_score,
                    precision=precision_score,
                    recall=recall_score)

    def f1_match(self, generation, answer):
        """
        :param generation:  A generated string containing only the newly generated tokens
        :param answer:  Answer string
        :return: F1 score of their match
        """

        f1pr_scores = self.f1pr_scores(generation, answer)
        return f1pr_scores.f1

    @staticmethod
    def find_answer_len(question_answer_token_ids, answer, llm_tokenizer):

        answer_stripped = answer.strip()

        length = question_answer_token_ids.shape[0]
        for i in range(length - 1, -1, -1):
            pad = llm_tokenizer.decode(question_answer_token_ids[i:], clean_up_tokenization_spaces=False)
            if pad.strip() == answer_stripped:
                return length - i
        raise AssertionError(f" Did not find {answer} in {question_answer_token_ids}")

    def answer_log_prob(self, log_prob, question_answer_token_ids, answer, llm_tokenizer):
        """
        :param log_prob: Log prob of question+answer of size question_answer_length x vocab
        :param question_answer_token_ids: indices of size question_answer_length
        :param answer: Actual answer
        :param llm_tokenizer: LLM tokenizer which is quite likely different from the self.tokenizer
        :return:
        """

        answer_len = self.find_answer_len(question_answer_token_ids, answer, llm_tokenizer)

        selected_log_prob = log_prob[:-1, :]                        # question + answer length - 1 x vocab
        indices = question_answer_token_ids[1:].unsqueeze(1)        # question + ans length - 1 x 1

        selected_log_prob = torch.gather(selected_log_prob, index=indices, dim=1)  # question + ans length - 1 x 1
        total_log_prob = selected_log_prob.sum().item()

        answer_log_prob = selected_log_prob[-answer_len:, 0]  # answer length
        answer_log_prob = answer_log_prob.sum().item()

        return ContextAnswerLogProb(total_log_prob=total_log_prob,
                                    answer_log_prob=answer_log_prob,
                                    answer_len=answer_len)

    def masked_answer_log_prob(self, log_prob, question_answer_token_ids, masked_question_answer_token_ids, tokenizer):
        """
        :param log_prob: Log-prob of question+answer
        :param question_answer_token_ids: token ids of question+answer
        :param masked_question_answer_token_ids: token ids of question+answer with some tokens masked
        :param tokenizer: tokenizer
        :return: ContextAnswerLogProb object
        """

        assert log_prob.ndim == 2        # Max length x vocab
        assert question_answer_token_ids.ndim == 1
        assert masked_question_answer_token_ids.ndim == 1
        assert question_answer_token_ids.shape[0] == masked_question_answer_token_ids.shape[0]
        assert question_answer_token_ids.shape[0] == log_prob.shape[0]

        answer_log_prob = 0.0
        answer_len = 1.0

        for i in range(masked_question_answer_token_ids.shape[0]):
            if int(masked_question_answer_token_ids[i]) == tokenizer.mask_token_id:
                token_id = int(question_answer_token_ids[i])
                answer_log_prob += log_prob[i, token_id]
                answer_len += 1

        answer_log_prob = float(answer_log_prob)
        answer_len = int(answer_len)

        return ContextAnswerLogProb(total_log_prob=None,
                                    answer_log_prob=answer_log_prob,
                                    answer_len=answer_len)


class ContextAnswerLogProb:

    def __init__(self, total_log_prob, answer_log_prob, answer_len):
        self.total_log_prob = total_log_prob
        self.answer_log_prob = answer_log_prob
        self.answer_len = answer_len


class F1PR:
    """
        A class object that wraps f1 score, precision and recall
    """

    def __init__(self, f1, precision, recall):

        self.f1 = f1
        self.precision = precision
        self.recall = recall
