from transformers import LlamaTokenizerFast, LlamaForCausalLM


class LLMWrapper:

    def __init__(self, llm_name):
        self.llm_name = llm_name

    def get_llm_and_tokenizer(self):

        if self.llm_name == "Roberta":
            pass

        elif self.llm_name == "GPTJ":
            pass

        elif self.llm_name == "Llama2-7B":
            llm_name = "Llama2-7G"
            llm_path = args.model_path
            tokenizer = LlamaTokenizerFast.from_pretrained(llm_path)
            base_model = LlamaForCausalLM.from_pretrained(llm_path)

        else:
            raise AssertionError("Unhandled LLM name")
