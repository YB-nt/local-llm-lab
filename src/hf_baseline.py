from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "Hello, how are you?"


def default_model_test():
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    model_inputs = tokenizer([DEFAULT_PROMPT], return_tensors="pt").to(model.device)

    model_outputs = model.generate(**model_inputs, max_length=30)

    print(tokenizer.batch_decode(model_outputs)[0])


if __name__ == "__main__":
    default_model_test()
