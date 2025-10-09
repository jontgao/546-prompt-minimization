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


def build_chat_input_from_config(config, user_content, assistant_content=None):
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


def preprocess_for_chat_model(dataset, config, text_column="text", max_examples=5):
    # Generic fallback
    processed = []

    for i, sample in enumerate(dataset):
        content = sample.get(text_column) or json.dumps(sample)
        chat_input = build_chat_input_from_config(config, user_content=content)
        processed.append(chat_input)
        if i + 1 >= max_examples:
            break

    return processed


def load_and_preprocess_dolly(config, split="train", max_examples=5):
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

        chat_input = build_chat_input_from_config(
            config,
            user_content=user_prompt,
            assistant_content=response
        )

        chat_input["ground_truth"] = response

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
        processed_data = load_and_preprocess_dolly(config, max_examples=args.max_examples)
    # if needed, add function to handle other datsets HERE
    else:
        print(f"Downloading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset)
        print("Preprocessing dataset...")
        processed_data = preprocess_for_chat_model(dataset, config, max_examples=args.max_examples)

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
