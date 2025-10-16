import argparse
import json
from pathlib import Path

from datasets import load_dataset


def load_model_config(model_name: str, config_dir: str = "../../config"):
    safe_name = model_name.replace("/", "-")
    config_path = Path(config_dir) / f"{safe_name}.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"No config file found for model '{model_name}' in {config_dir}/"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tinyllama_input(config, user_content, assistant_content=None):
    chat_input = {}

    if "messages" in config:
        chat_input["messages"] = json.loads(json.dumps(config["messages"]))
    else:
        # Fallback if config has only schema, not example messages
        chat_input["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ""}
        ]

    for msg in chat_input["messages"]:
        if msg["role"] == "user":
            msg["content"] = user_content
        elif msg["role"] == "assistant" and assistant_content is not None:
            msg["content"] = assistant_content

    for key, value in config.get("properties", {}).items():
        if "default" in value:
            chat_input[key] = value["default"]

    return chat_input

def build_llama3_input(config, user_content, assistant_content=None):
    B_SYS, E_SYS = "<|start_header_id|>system<|end_header_id|>\n", "<|eot_id|>\n"
    B_USER, E_USER = "<|start_header_id|>user<|end_header_id|>\n", "<|eot_id|>\n"
    B_ASSISTANT, E_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>\n", "<|eot_id|>\n"

    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    message = f"<|begin_of_text|>\n{B_SYS}{system_prompt}{E_SYS}"

    message += f"{B_USER}{user_content}{E_USER}"
    if assistant_content is not None:
        message += f"{B_ASSISTANT}{assistant_content}{E_ASSISTANT}"

    return {"prompt": message}

def build_qwen_input(config, user_content, assistant_content=None):
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    messages = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>",
        f"<|im_start|>user\n{user_content}<|im_end|>"
    ]

    if assistant_content is not None:
        messages.append(f"<|im_start|>assistant\n{assistant_content}<|im_end|>")

    return {"prompt": "\n".join(messages)}


def build_chat_input_for_model(model_name, config, user_content, assistant_content=None):
    if "Llama-3" in model_name or "llama3" in model_name.lower():
        return build_llama3_input(config, user_content, assistant_content)
    elif "qwen" in model_name.lower():
        return build_qwen_input(config, user_content, assistant_content)
    elif "tinyllama" in model_name.lower():
        return build_tinyllama_input(config, user_content, assistant_content)
    else:
        #some fallback option here, probably
        return


def preprocess_for_chat_model(dataset, config, text_column="text", max_examples=5):
    # Generic fallback
    processed = []

    for i, sample in enumerate(dataset):
        content = sample.get(text_column) or json.dumps(sample)
        chat_input = build_chat_input_for_model(config, user_content=content)
        processed.append(chat_input)
        if i + 1 >= max_examples:
            break

    return processed


def load_and_preprocess_dolly(config, model, split="train", max_examples=5):
    dataset = load_dataset("databricks/databricks-dolly-15k", split=split)
    processed = []

    for i, sample in enumerate(dataset):
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        response = sample.get("response", "")
        context = sample.get("context", "")

        user_parts = [instruction]
        if input_text and input_text.strip():
            user_parts.append(input_text)
        if context and context.strip():
            user_parts.append(f"\nContext:\n{context}")

        user_prompt = "\n".join(part.strip() for part in user_parts if part.strip())

        chat_input = build_chat_input_for_model(
            model, 
            config,
            user_content=user_prompt,
            assistant_content=response
        )

        chat_input["ground_truth"] = response

        processed.append(chat_input)
        if i + 1 >= max_examples:
            break

    return processed

def load_and_preprocess_squad_v2(config, model, split="train", max_examples=5):
    dataset = load_dataset("squad_v2", split=split)
    processed = []

    for i, sample in enumerate(dataset):
        question = sample.get("question", "").strip()
        context = sample.get("context", "").strip()
        answers = sample.get("answers", {})
        answer_texts = answers.get("text", [])

        # SQuAD 2.0 has questions with no answers
        is_unanswerable = len(answer_texts) == 0
        ground_truth = answer_texts[0].strip() if not is_unanswerable else "I'm sorry, but this question is unanswerable based on the given context."

        user_prompt = f"Answer the following question based on the context.\n\nContext:\n{context}\n\nQuestion:\n{question}"

        chat_input = build_chat_input_for_model(
            model, 
            config,
            user_content=user_prompt,
            assistant_content=ground_truth
        )

        chat_input["ground_truth"] = ground_truth
        chat_input["unanswerable"] = is_unanswerable

        processed.append(chat_input)

        if i + 1 >= max_examples:
            break

    return processed


def load_and_preprocess_gsm8k(config, model, cfg="main", max_examples=5):
    dataset = load_dataset("openai/gsm8k", cfg)
    train_dataset = dataset["train"]
    #test_dataset = dataset["test"]
    processed = []

    for i, sample in enumerate(train_dataset):
        print(f"Sample {i}: {sample}")
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        user_prompt = question.strip()
        assistant_response = answer.strip()

        chat_input = build_chat_input_for_model(
            model, 
            config,
            user_content=user_prompt,
            assistant_content=assistant_response
        )

        chat_input["ground_truth"] = answer.strip()

        processed.append(chat_input)
        if i + 1 >= max_examples:
            break

    return processed

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess dataset for a given LLM.")
    parser.add_argument("--model", required=True, help="Model name (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. imdb or databricks/databricks-dolly-15k)")
    parser.add_argument("--max_examples", type=int, default=5, help="Limit examples for preview or small runs")
    args = parser.parse_args()

    print(f"Loading config for model: {args.model}")
    config = load_model_config(args.model)

    if args.dataset == "databricks/databricks-dolly-15k":
        processed_data = load_and_preprocess_dolly(config, args.model, max_examples=args.max_examples)
    if args.dataset == "openai/gsm8k":
        processed_data = load_and_preprocess_gsm8k(config, args.model, max_examples=args.max_examples)
    if args.dataset == "rajpurkar/squad_v2":
        processed_data = load_and_preprocess_squad_v2(config, args.model, max_examples=args.max_examples)
    # if needed, add function to handle other datsets HERE
    else:
        print(f"Downloading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset)
        print("Preprocessing dataset...")
        processed_data = preprocess_for_chat_model(dataset, config, args.model, max_examples=args.max_examples)

    output_path = Path("../../data") / f"{args.dataset.replace('/', '-')}_{args.model.replace('/', '-')}.jsonl"

    print(f"Saving processed data to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")


    print("Example processed entry:")
    print(json.dumps(processed_data[0], indent=2))

    return processed_data


if __name__ == "__main__":
    main()
