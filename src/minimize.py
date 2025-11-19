import gc
import hashlib
import heapq
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from metrics import BERTScoreScorer, CompressionLengthScorer

DEFAULT_LLM = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


class MultiStageOptimization:
    def __init__(self, config):
        self.llm = LLM(model=config.model, seed=config.seed,
                       tensor_parallel_size=torch.cuda.device_count(),
                       gpu_memory_utilization=0.85,
                       enable_prefix_caching=True)

        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)

        self.bert_scorer = BERTScoreScorer()
        self.bert_score_weight = config.bert_score_weight
        self.compression_scorer = CompressionLengthScorer()
        self.compression_weight = config.compression_weight
        self.top_n = config.top_n

        self.num_iterations = config.num_iterations

        self.batch_size = config.batch_size

        self.base_folder = Path(config.run_folder)

    def save_meta(self, save_dir, initial_prompt: str, initial_prompt_output, system_prompt):
        with open(save_dir / 'meta.json', 'w') as f:
            json.dump({
                'initial_prompt': initial_prompt,
                'initial_output': initial_prompt_output,
                'system_prompt': system_prompt,
                'tokenizer_model': self.config.model,
                'use_initial_output': self.config.use_initial_output,
                'bert_score_weight': self.bert_score_weight,
                'compression_weight': self.compression_weight,
                'top_n': self.top_n,
                'num_iterations': self.num_iterations,
                'batch_size': self.batch_size,
                'base_folder': str(self.base_folder.absolute())
            }, f, indent=4)

    def save_events(self, save_dir, events: List[Dict[str, Union[str, float]]]):
        with open(save_dir / 'events.json', 'w') as f:
            json.dump(events, f, indent=4)

    def __call__(self, initial_prompt: str, initial_prompt_output: Optional[str] = None):
        prompts: List[Tuple[str, float]] = []

        if self.config.use_initial_output:
            assert initial_prompt_output is not None, "You need to supply an initial prompt output"
        else:
            sampling_params = SamplingParams(temperature=self.config.temperature, top_p=self.config.top_p,
                                             max_tokens=self.config.max_token_length)

            new_prompt_outputs = self.llm.generate(initial_prompt, sampling_params)[0]

            initial_prompt_output = new_prompt_outputs.outputs[0].text.strip()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_hash = str(stable_hash(initial_prompt))
        run_name = f'run-{timestamp}-{run_hash}'

        save_folder = self.base_folder / config.model.split('/')[-1] / run_name

        save_folder.mkdir(parents=True, exist_ok=True)

        system_prompt = self.generate_system_prompt(initial_prompt, initial_prompt_output)

        self.save_meta(save_folder, initial_prompt, initial_prompt_output, system_prompt)

        initial_prompt_output_encoded = self.tokenizer.encode(initial_prompt_output)

        max_output_length = min(len(initial_prompt_output_encoded), self.config.max_token_length)

        self.sampling_params = SamplingParams(temperature=self.config.temperature, top_p=self.config.top_p,
                                              max_tokens=max_output_length)

        print("Using max token length", max_output_length)

        current_best_prompts: List[Tuple[float, Tuple[str, str]]] = [
            (self.compression_weight, (initial_prompt, initial_prompt_output))]

        events = [
            {
                'iteration': 0,
                'prompt': initial_prompt,
                'output': initial_prompt_output,
                'score': self.compression_weight,
                'origin': None,
                'scores': {
                    'bert': 0,
                    'compression': self.compression_weight,
                    'total': self.compression_weight,
                }
            }

        ]

        self.save_events(save_folder, events)

        for iteration in range(self.num_iterations):
            print(current_best_prompts)

            prompt_outputs, seeds = self.stage_one(current_best_prompts, system_prompt)

            # print("Prompt outputs:", prompt_outputs)

            scores, score_decomp = self.score(prompt_outputs, initial_prompt, initial_prompt_output)

            # print("Scored outputs:", scores)

            for (score_, (prompt_, prompt_output_)), score_decomp_, seed in zip(scores, score_decomp, seeds):
                events.append({
                    'iteration': iteration + 1,
                    'prompt': prompt_,
                    'output': prompt_output_,
                    'score': score_,
                    'origin': seed,
                    'scores': score_decomp_
                })
            self.save_events(save_folder, events)

            for p in scores:
                heapq.heappush(prompts, p)

            # Currently only top 1s
            current_best_prompts = heapq.nsmallest(self.top_n, prompts)

            best = current_best_prompts[0]
            print(f"Using the following prompts as top, score:{best[0]}, prompt:{best[1][0]}")

    def generate_system_prompt(self, initial_prompt: str,
                               initial_prompt_output: str) -> str:
        system_prompt = fr"""You are a PRECISION prompt-compression specialist. Your job: produce a new,
significantly shorter prompt that, when given to the same LLM under the
same generation settings, will produce the EXACT SAME textual output as the
original prompt did. Be concise, deterministic, and conservative: you may
remove only words or structure that provably do not change the output.

CONSTRAINTS (must follow all):
1) ONLY output the NEW PROMPT TEXT and NOTHING ELSE — no explanation, no quotes,
no extra whitespace, no metadata, no JSON, no commentary. If you cannot compress
without changing the output, output the ORIGINAL PROMPT verbatim.
2) Preserve all critical facts: named entities, numbers, punctuation that affect
parsing (dates, scores, code tokens). If the original answer depends on a specific
format (e.g., list, table, JSON), preserve the format instruction.
3) Minimize tokens: prefer shorter synonyms, remove filler, eliminate redundancy,
and collapse multi-sentence instructions into a single concise instruction.
4) Do not introduce new assumptions, unspecified defaults, or inventions.
5) Do not use placeholders (e.g., [DETAILS]) unless they are provably inert for the output.

SCORING HEURISTICS (informational — you do not compute these):
- Fidelity: preserve semantics; any change that could alter the generated output is forbidden.
- Brevity: shorter prompts are better when fidelity is preserved.
- Clarity: keep the prompt unambiguous for the LLM.

HOW TO COMPRESS (concrete techniques):
- Remove polite framing ('please', 'kindly') and meta commentary ('I want you to...').
- Convert multi-step prose into compact imperative instructions (e.g., 'Write a 3-sentence summary' -> '3-sentence summary').
- Replace verbose qualifiers with compact explicit constraints ('in no more than 50 words' -> '≤50 words').
- Merge context into one short clause; move examples only if essential.

EXACT TASK CONTEXT:
ORIGINAL PROMPT: {initial_prompt}
ORIGINAL PROMPT OUTPUT (for fidelity reference): {initial_prompt_output}\
ORIGINAL SCORE {self.compression_weight}\

You need to be MINIMIZING the score, you should never generate a score that is larger. 

OUTPUT RULE (critical):
- If you are absolutely certain the new prompt will produce the same output, output that new prompt only.
- If you are uncertain or cannot guarantee exact parity, output the ORIGINAL PROMPT exactly as given.

ONLY OUTPUT THE PROMPT — DO NOT ADD ANYTHING ELSE.

FEW-SHOT EXAMPLES (for conditioning — do not print these examples in final output):

[Example 1]
ORIGINAL PROMPT: Please write a very short (2–3 sentence) summary of the plot of "Romeo and Juliet", focusing on the main events and the motivations of the principal characters. Be concise and do not include quotations from the play.
ORIGINAL PROMPT OUTPUT: Romeo and Juliet is a tragedy about two young lovers from feuding families in Verona. Their secret marriage and attempts to reconcile their families lead to misunderstandings and a sequence of events that ends in both their deaths—driven by love, impulsiveness, and family honor.
COMPRESSED PROMPT (valid): 2–3 sentence summary of Romeo & Juliet focusing on main events and principal motivations; no quotations.

[Example 2]
ORIGINAL PROMPT: I want a JSON object listing the teams, the final score, and the winner from the match where Team A beat Team B 3-1 on 2021-05-06. The JSON keys should be "teams", "score", and "winner" in that exact order.
ORIGINAL PROMPT OUTPUT: {{"teams": ["Team A", "Team B"], "score": "3-1", "winner": "Team A"}}
COMPRESSED PROMPT (valid): Return JSON with keys ["teams","score","winner"] in that order for the 2021-05-06 Team A vs Team B match (3-1).

[Example 3 — do not compress]
ORIGINAL PROMPT: Translate the following legal clause into plain English but do not change its legal meaning or remove any conditions: "If the Lessee fails to pay rent within thirty (30) days of written notice, the Lessor may, at its option, terminate this lease." Use formal legal phrasing but simpler language.
ORIGINAL PROMPT OUTPUT: If the tenant doesn't pay rent within thirty (30) days after written notice, the landlord can choose to end the lease. (Retains legal force and conditions.)
COMPRESSED PROMPT (invalid — must return original): Paraphrase clause: tenant nonpayment after 30 days → landlord may end lease.

[Example 4]
ORIGINAL PROMPT: Who was Kyle Van Zyl playing against when he scored 36 of his team's 61 points?
ORIGINAL PROMPT OUTPUT: He was playing against Boland U21.
COMPRESSED PROMPT (valid conservative): Opponent when Kyle Van Zyl scored 36 of his team's 61 points?"""

        return system_prompt

    def stage_one(self, prompts: List[Tuple[float, Tuple[str, str]]], system_prompt: str) -> Tuple[
        List[Tuple[str, str]], List[str]]:

        return self.generate(prompts, system_prompt)

    def generate(self, prompts: List[Tuple[float, Tuple[str, str]]], system_prompt: str) -> Tuple[
        List[Tuple[str, str]], List[str]]:
        messages = []

        weights = [1 / (s + 1) for s, _ in prompts]

        prompts_to_use = random.choices(prompts, k=self.batch_size, weights=weights)

        for score, (prompt, prompt_output) in prompts_to_use:
            text = [
                {
                    'role': 'system', 'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': f'''For context on previous iterations, the following seed prompt was used
SEED PROMPT: {prompt} 
SEED PROMPT OUTPUT: {prompt_output}
SEED PROMPT SCORE: {score} 
''',
                }
            ]

            text = self.tokenizer.apply_chat_template(
                text,
                tokenize=False,
                add_generation_prompt=True
            )

            messages.append(text)

        prompt_seed = [seed for _, (seed, _) in prompts_to_use]

        outputs = self.llm.generate(messages, self.sampling_params)

        new_prompts = [output.outputs[0].text.strip() for output in outputs]

        new_prompt_outputs = self.llm.generate(new_prompts, self.sampling_params)

        return [(new_prompt, new_output.outputs[0].text.strip()) for new_prompt, new_output in
                zip(new_prompts, new_prompt_outputs)], prompt_seed

    def score(self, prompts: List[Tuple[str, str]], initial_prompt: str, initial_prompt_output: str) -> Tuple[List[
        Tuple[float, Tuple[str, str]]], List[Dict[str, float]]]:
        prompt_inputs, prompt_outputs = zip(*prompts)
        prompt_inputs, prompt_outputs = list(prompt_inputs), list(prompt_outputs)

        compression = np.array(self.compression_scorer.compute_score(prompt_inputs, initial_prompt))
        bert_score = 1.0 - np.array(self.bert_scorer.compute_score(prompt_outputs, initial_prompt_output))

        for i in range(len(prompt_outputs)):
            prompt_output: str = prompt_outputs[i]

            if len(prompt_output.strip()) == 0:
                bert_score[i] = 1.0

        total_score = ((bert_score * self.bert_score_weight) + (compression * self.compression_weight)).tolist()

        scoring = list(zip(total_score, prompts))

        score_decomps = []

        for c, b, t in zip(compression, bert_score, total_score):
            score_decomp = {
                'bert': b,
                'compression': c,
                'total': t,
            }

            score_decomps.append(score_decomp)

        # scoring: List[Tuple[float, Tuple[str, str]]]
        # assert len(scoring) == len(prompts)
        # assert all(isinstance(s[0], float) for s in scoring)
        # assert all(isinstance(s[1], tuple) and len(s[1]) == 2 for s in scoring)
        # assert all(isinstance(s[1][0], str) and isinstance(s[1][1], str) for s in scoring)

        return scoring, score_decomps


if __name__ == '__main__':

    class Config:
        model = 'Qwen/Qwen2.5-32B-Instruct-AWQ'
        # model = DEFAULT_LLM
        temperature = 0.9
        top_p = 0.95
        seed = 0
        bert_score_weight = 10.0
        compression_weight = 1.0
        num_iterations = 15
        top_n = 10
        batch_size = 100
        max_token_length = 30000
        run_folder = 'runs'
        use_initial_output = False


    models = ['meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen2.5-32B-Instruct-AWQ']

    config = Config()

    with open('data/long_prompts.json', 'r') as f:
        prompts = json.load(f)

    for model in models:
        config.model = model

        temp = MultiStageOptimization(config)

        for prompt in prompts:
            for i in range(1):
                temp(prompt)

        del temp
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
