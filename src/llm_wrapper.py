class LLMWrapper:

    def __init__(self):
        pass

    def get_llm_and_tokenizer(self, llm_name):

        if llm_name == "Roberta":
            pass

        elif llm_name == "GPTJ":
            pass

        elif llm_name == "Llama2-7B":
            llm_name = "Llama2-7G"
            llm_path = args.model_path
            tokenizer = LlamaTokenizerFast.from_pretrained(llm_path)
            base_model = LlamaForCausalLM.from_pretrained(llm_path)


        else:
            raise AssertionError("Unhandled LLM name")
