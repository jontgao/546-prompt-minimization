import torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

import json
import hashlib
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer

# REQUIRES mistral (or other llm) running locally on a ollama server, https://ollama.com/download
# TODO: maybe include output in system prompt, maybe genereate multiple candidate prompts with .batch()

from metrics import BERTScoreScorer, CompressionLengthScorer

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

class PromptMinimizerLangChain:
    def __init__(self, config):
        self.config = config
        self.bert_scorer  = BERTScoreScorer()
        self.compression_scorer = CompressionLengthScorer()
        
        if getattr(self.config, "ollama", False):
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(model="mistral", temperature=self.config.temperature, num_predict=self.config.max_token_length)

            self.ollama_minimizer_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a expert prompt engineer. Your job is to rewrite and minimize prompts to be as short and concise as possible "
                    "The user will provide you with an original prompt."
                    "Your job is to generate a new, shorter prompt that will produce the same output"
                    "ONLY adapt the original prompt to be shorter while preserving its meaning."
                    "DO NOT use the original output in your response"
                    "Only answer with the new shorter prompt and nothing else"
                )),
                ("user", (
                    "Original prompt: \n\n{prompt}\n\n "
                    #"Original prompt: \n\n{output}\n\n " # Spent too much time trying to get this to work, if I include it the LLM copies it... 
                    "Short prompt:"
                )),
            ])
        else:
            from langchain_community.llms import VLLM
            self.llm = VLLM(model=self.config.model, temperature=self.config.temperature, max_new_tokens=self.config.max_token_length,
                            tensor_parallel_size=torch.cuda.device_count(),
                            gpu_memory_utilization=0.85,
                            enable_prefix_caching=True
                            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)        

        # Define the chains
        self.minimize_chain = RunnableLambda(self.minimize)
        self.test_chain = RunnableLambda(self.test_minimized_prompt)
        self.evaluator_chain = RunnableLambda(self.evaluate)
        self.full_chain = self.minimize_chain | self.test_chain | self.evaluator_chain

        if self.config.save_run:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                session_hash = stable_hash(str(config.__dict__))
                self.save_dir = Path("runs") / f"session-{timestamp}-{session_hash[:8]}"
                self.save_dir.mkdir(parents=True, exist_ok=True)
                self.save_meta()

                # for saving run files
                self.run_idx = 0

    def __call__(self, original_prompt: str, original_output: str = None):

        if self.config.generate_output:
            if self.config.verbose: 
                print("generating origial output.. ")
            if self.config.ollama:
                output = self.llm.invoke(self.ollama_minimizer_prompt.format_messages(prompt=original_prompt))
                original_output = output.content if hasattr(output, "content") else output
            else:
                output = self.llm.invoke(self.generator_prompt(original_prompt))
                original_output = output.content if hasattr(output, "content") else output
        else:
            if original_output is None:
                raise ValueError("original_output must be provided if generate_output is False")
            if self.config.verbose:
                print("Using provided output")
            
        self.original_prompt = original_prompt
        self.original_output = original_output

        if self.config.verbose:
            print("Original prompt:", original_prompt)
            print("Original output:", original_output)

        history = self.iterative_minimization()

        if self.config.save_run:
            self.run_idx += 1
            timestamp = datetime.now().strftime('%H%M%S')
            run_hash = stable_hash(str(self.config.__dict__) + timestamp)
            self.run_file = f"run-{(self.run_idx):03d}-{run_hash[:6]}.json"

            self.save_history(history)
        
        return history

    def generator_prompt(self, prompt: str):
        message = [
            {"role": "system", "content": "You are a helpful AI chat assistant."},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    def minimizer_prompt(self, inputs: dict):

        system_prompt = """
                You are a expert prompt engineer. Your job is to rewrite and minimize prompts to be as short and concise as possible 
                The user will provide you with an original prompt.
                Your job is to generate a new, shorter prompt that will produce the same output.
                You should adapt the original prompt to be shorter while preserving its meaning.

                Your output shoud look like this:
                (short prompt)

                Only answer with the short prompt and nothing else.
                The short prompt MUST be shorther than the original prompt.
                DO NOT just copy the original prompt.
                """
        user_prompt = f"""
                Original prompt: {inputs["prompt"]}
                Short prompt:
                """
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # Minimize the prompt
    def minimize(self, inputs: dict):
        if self.config.verbose:
            print("Current prompt: ", inputs["prompt"])

        if self.config.ollama:
            msg = self.ollama_minimizer_prompt.format_messages(prompt=inputs["prompt"])
        else:
            msg = self.minimizer_prompt(inputs)
          
        response = self.llm.invoke(msg)
        content = response.content if hasattr(response, "content") else response
        return {**inputs, "new_prompt": content}

    # Test the minimized prompt
    def test_minimized_prompt(self, inputs: dict):
        if self.config.verbose:
            print("Proposed prompt:", inputs["new_prompt"])

        if self.config.ollama:
            output = self.llm.invoke(inputs["new_prompt"])
        else:
            output = self.llm.invoke(self.generator_prompt(inputs["new_prompt"]))
          
        content = output.content if hasattr(output, "content") else output

        if self.config.verbose:
            print("Generated output:", content)
          
        return {**inputs, "new_output": content}

    # Evaluate new prompt and output.
    def evaluate(self, inputs: dict):
        comp_score = self.compression_scorer.compute_score(inputs["new_prompt"], self.original_prompt)
        bert_score = self.bert_scorer.compute_score([inputs["new_output"]], [self.original_output])[0]
        total_score = (1-bert_score) * self.config.bert_score_weight + comp_score * self.config.compression_weight

        return {**inputs, "score": total_score, "bert_score": bert_score, "compression_score": comp_score}

    def iterative_minimization(self):
        inputs = {
            "prompt": self.original_prompt,
            "output": self.original_output,
        }

        history = []

        for i in range(self.config.num_iterations):
            
            print(f"\n--- Step {i+1} ---")
            inputs["iteration"] = i + 1

            result = self.full_chain.invoke(inputs)
            history.append(result)

            print(f"Compression: {result['compression_score']:.4f} | "
                  f"BERT: {result['bert_score']:.4f} | "
                  f"Total: {result['score']:.4f}")

            inputs["prompt"] = result["new_prompt"]
            #inputs["output"] = result["new_output"]

        return history

    def save_meta(self):
        """Save configuration and initial data (meta.json)"""
        meta = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "num_iterations": self.config.num_iterations,
            "bert_score_weight": self.config.bert_score_weight,
            "compression_weight": self.config.compression_weight,
            "max_token_length": self.config.max_token_length,
            "ollama": getattr(self.config, "ollama", False),
            "generate_output": self.config.generate_output,
        }
        with open(self.save_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

    def save_history(self, history):
        # Save everything to a single file at the end
        run_data = {
            "meta": {
                "original_prompt": self.original_prompt,
                "original_output": self.original_output,
                "generate_output": self.config.generate_output,
            },
            "history": history,
        }

        with open(self.save_dir / self.run_file, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=4)

        if self.config.verbose:
            print(f"Run saved to {self.save_dir / self.run_file}")



if __name__ == "__main__":

    class Config:
        ollama = False
        model = 'Qwen/Qwen2.5-32B-Instruct-AWQ'
        temperature = 0.0
        bert_score_weight = 0.5
        compression_weight = 0.5
        num_iterations = 10
        max_token_length = 32000
        save_run = True
        generate_output = True
        verbose = False

    temp = PromptMinimizerLangChain(Config())

    # Test input
    #original_prompt = "Explain the process of photosynthesis in simple terms for a 10th grade science class."
    #original_output = "Photosynthesis is the process plants use to convert sunlight into energy. They take in carbon dioxide and water, and with the help of sunlight, they produce glucose and oxygen."

    #history = temp(original_prompt, original_output)
    history_list = []
    # prompt_list = [
    #     "Explain the process of photosynthesis in simple terms for a 10th grade science class.",
    #     "Explain the process of planetary motion in simple terms for highschool children",
    #     "Who won the football world cup in 2014, and who scored the final goal"
    # ]

    with open('data/long_prompts.json', 'r') as f:
        prompt_list = json.load(f)

    for prompt in prompt_list:
        out = temp(prompt, None)
        history_list.append(out)

    # plotting results 

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 5))

    num_iters = len(history_list[0])  # assumes all histories same length

    # Collect data
    all_scores = np.array([[step['score'] for step in h] for h in history_list])
    all_bert_scores = np.array([[1 - step['bert_score'] for step in h] for h in history_list])
    all_compression_scores = np.array([[step['compression_score'] for step in h] for h in history_list])

    # Compute mean and std
    mean_scores = all_scores.mean(axis=0)
    std_scores = all_scores.std(axis=0)

    mean_bert = all_bert_scores.mean(axis=0)
    std_bert = all_bert_scores.std(axis=0)

    mean_compression = all_compression_scores.mean(axis=0)
    std_compression = all_compression_scores.std(axis=0)

    x = np.arange(num_iters)

    # Plot mean lines
    plt.plot(x, mean_scores, color='tab:blue', label='Overall (mean)', linewidth=2.5, marker='o')
    plt.plot(x, mean_bert, color='tab:orange', label='BERT (mean)', linewidth=2.5, marker='s')
    plt.plot(x, mean_compression, color='tab:green', label='Compression (mean)', linewidth=2.5, marker='^')

    # Plot ±1 std shading
    plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, color='tab:blue', alpha=0.2)
    plt.fill_between(x, mean_bert - std_bert, mean_bert + std_bert, color='tab:orange', alpha=0.2)
    plt.fill_between(x, mean_compression - std_compression, mean_compression + std_compression, color='tab:green', alpha=0.2)

    # Labels & formatting
    plt.xlabel('Iteration')
    plt.xticks(x, range(1, num_iters + 1))
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('Average Prompt Minimization Scores (±1 Std) Over Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{temp.save_dir}/prompt_minimization_scores.png")






