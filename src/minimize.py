import heapq
import random
from typing import Tuple, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_LLM = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

from metrics import BERTScoreScorer, CompressionLengthScorer


class MultiStageOptimization:
    def __init__(self, config):
        self.llm = LLM(model=config.model, seed=config.seed)

        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)

        self.bert_scorer = BERTScoreScorer()
        self.bert_score_weight = config.bert_score_weight
        self.compression_scorer = CompressionLengthScorer()
        self.compression_weight = config.compression_weight
        self.top_n = config.top_n

        self.num_iterations = config.num_iterations

        self.batch_size = config.batch_size

    def __call__(self, initial_prompt: str, initial_prompt_output: str):
        prompts: List[Tuple[str, float]] = []


        initial_prompt_output_encoded = self.llm.encode(initial_prompt_output)

        max_output_length = min(len(initial_prompt_output_encoded), self.config.max_token_length)

        self.sampling_params = SamplingParams(temperature=self.config.temperature, top_p=self.config.top_p, max_tokens=max_output_length)

        print("Using max token length", max_output_length)


        current_best_prompts: List[Tuple[float, Tuple[str, str]]] = [
            (self.compression_weight, (initial_prompt, initial_prompt_output))]

        for _ in range(self.num_iterations):
            print(current_best_prompts)

            prompt_outputs = self.stage_one(current_best_prompts, initial_prompt, initial_prompt_output)

            print("Prompt outputs:", prompt_outputs)

            scores = self.score(prompt_outputs, initial_prompt, initial_prompt_output)

            print("Scored outputs:", scores)

            for p in scores:
                heapq.heappush(prompts, p)

            # Currently only top 1s
            current_best_prompts = heapq.nsmallest(self.top_n, prompts)

            best = current_best_prompts[0]
            print(f"Using the following prompts as top, score:{best[0]}, prompt:{best[1][0]}")

    def stage_one(self, prompts: List[Tuple[float, Tuple[str, str]]], initial_prompt: str,
                  initial_prompt_output: str) -> List[
        Tuple[str, str]]:

        system_prompt = (f"You are trying to generate minimal prompt that would generate the exact same result as the "
                         f"the following prompt. You are trying to minimize compression ratio length of the new "
                         f"prompt and the original prompt and the BERT Score which is the semantic "
                         f"similarity"
                         f"\nPrompt: {initial_prompt}"
                         f"\nAnswer: {initial_prompt_output}"
                         f"\nThe user will be providing the seed output and the score associated with that prompt and "
                         f"you want to ensure that the prompts that you provide are better than the previous ones. Do "
                         f"not add extra text except the output prompt only. ONLY output the prompt")
        return self.generate(prompts, system_prompt)

    def generate(self, prompts: List[Tuple[float, Tuple[str, str]]], system_prompt: str) -> List[Tuple[str, str]]:
        messages = []

        for score, (prompt, prompt_output) in prompts:
            text = [
                {
                    'role': 'system', 'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': f'Prompt to optimize: {prompt} which generated "{prompt_output}" and a score of {score}.\n\nNew prompt: ',
                }
            ]

            text = self.tokenizer.apply_chat_template(
                text,
                tokenize=False,
                add_generation_prompt=True
            )

            messages.append(text)

        if len(messages) < self.batch_size:
            outputs = random.choices(messages, k=self.batch_size - len(messages))

            messages.extend(outputs)

        outputs = self.llm.generate(messages, self.sampling_params)

        new_prompts = [output.outputs[0].text for output in outputs]

        new_prompt_outputs = self.llm.generate(new_prompts, self.sampling_params)

        return [(new_prompt, new_outut.outputs[0].text) for new_prompt, new_outut in
                zip(new_prompts, new_prompt_outputs)]

    def score(self, prompts: List[Tuple[str, str]], initial_prompt: str, initial_prompt_output: str) -> List[
        Tuple[float, Tuple[str, str]]]:
        # TODO: vectorize
        scoring: List[Tuple[float, Tuple[str, str]]] = []

        for prompt, prompt_output in prompts:
            compression = self.compression_scorer.compute_score(prompt, initial_prompt)

            bert_score = self.bert_scorer.compute_score([prompt_output], [initial_prompt_output])[0]

            total_score = (1 - bert_score) * self.bert_score_weight + compression * self.compression_weight

            scoring.append((total_score, (prompt, prompt_output)))

        return scoring


if __name__ == '__main__':
    class Config:
        model = DEFAULT_LLM
        temperature = 0.9
        top_p = 0.95
        seed = 0
        bert_score_weight = 10.0
        compression_weight = 1.0
        num_iterations = 10
        top_n = 1
        batch_size = 100
        max_token_length = 10000

    temp = MultiStageOptimization(Config())

    prompt = 'Who was Kyle Van Zyl playing against when he scored 36 of hisa teams 61 points?'

    output = '''Van Zyl joined the Eastern Province Kings Academy, where he played for the Eastern Province U19 side in the 2010 Under-19 Provincial Championship. He was a key player for the Eastern Province U21 side in the 2012 Under-21 Provincial Championship, scoring 71 points in eight appearances. Van Zyl was under the Top SARU Performers, scoring the most tries at 6 in the 2012 Provincial Under 21 in the Rugby Junior Provincials.

This included a record and a remarkable personal haul in their opening match, when he scored 36 of his team's points in a 61–3 victory over Boland U21, consisting of four tries and eight conversions and was awarded Man of the Match.'''

    temp(prompt, output)
