import json
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
# really really not a good idea but makes AWQ work?
try:
    import transformers.activations
    if not hasattr(transformers.activations, "PytorchGELUTanh"):
        transformers.activations.PytorchGELUTanh = transformers.activations.NewGELUActivation
except ImportError:
    pass

from peft import LoraConfig, get_peft_model, TaskType

from metrics import BERTScoreScorer, CompressionLengthScorer


@dataclass
class PPOConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # Will default to ["q_proj", "v_proj"]

    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 4
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    max_new_tokens: int = 30000
    temperature: float = 0.8
    top_p: float = 0.9

    num_iterations: int = 15
    warmup_steps: int = 100

    semantic_weight: float = 0.5
    compression_weight: float = 0.5

    output_dir: str = "../runs_li"
    save_every: int = 5
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class ExperienceBuffer:
    def __init__(self):
        self.prompts = []
        self.responses = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.feedback = []
    
    def add(self, prompt, response, log_prob, value, reward, feedback="", done=False):
        self.prompts.append(prompt)
        self.responses.append(response)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.feedback.append(feedback)
        self.dones.append(done)
    
    def clear(self):
        self.prompts.clear()
        self.responses.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.feedback.clear()
        self.dones.clear()
    
    def get(self):
        return {
            'prompts': self.prompts,
            'responses': self.responses,
            'log_probs': torch.stack(self.log_probs),
            'values': torch.stack(self.values),
            'rewards': torch.tensor(self.rewards),
            'feedback': self.feedback,
            'dones': torch.tensor(self.dones)
        }


class ValueHead(torch.nn.Module):    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, hidden_states):
        return self.value_head(hidden_states).squeeze(-1)


class PromptCompressionPPO:    
    def __init__(
        self, 
        config: PPOConfig,
        initial_prompt: str
    ):
        self.config = config
        self.initial_prompt = initial_prompt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.semantic_scorer = BERTScoreScorer()
        self.compression_scorer = CompressionLengthScorer()

        self._init_model()

        self.run_dir = self._create_run_dir()

        self.milestones = []
        self.iteration = 0
        self.last_feedback = None
        
    def _init_model(self):
        print(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("Using bfloat16 for stability")
            dtype = torch.bfloat16

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto"
        )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        hidden_size = self.model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.device).to(base_model.dtype)

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            lr=self.config.learning_rate
        )

        total_steps = self.config.num_iterations * self.config.ppo_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
    
    def _create_run_dir(self) -> Path:
        """Create directory for this run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model_name.replace("/", "_")
        prompt_hash = hash(self.initial_prompt) % 10000
        
        run_dir = Path(self.config.output_dir) / f"{model_name}_prompt{prompt_hash}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        with open(run_dir / "initial_prompt.txt", "w") as f:
            f.write(self.initial_prompt)
        
        return run_dir
    
    def generate_response(
        self, 
        prompt: str, 
        return_log_probs: bool = False,
        deterministic: bool = False
    ) -> Tuple[str, Optional[torch.Tensor], Optional[torch.Tensor]]:
        feedback_context = ""
        aggressive_instruction = "Remove filler words. Use imperative mood."
        
        if self.last_feedback and self.iteration > 0:
            feedback_context = f"\nFEEDBACK FROM LAST ATTEMPT:\n{self.last_feedback}"

            if "minimal compression" in self.last_feedback.lower():
                aggressive_instruction = "Your last attempt was too long. You MUST cut at least 30% of the words. Use symbols (&, ->) and remove all articles (the, a)."
            elif "semantic preservation" in self.last_feedback.lower() and "poor" in self.last_feedback.lower():
                aggressive_instruction = "Your last attempt lost too much meaning. Keep specific constraints and entities, but shorten the grammar."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Prompt Compression Specialist. Your goal is to shorten prompts "
                    "while retaining their semantic instruction. "
                    "1. Remove politeness and filler words. "
                    "2. Use symbols (=, >, &) to replace words. "
                    "3. Merge instructions. "
                    "4. Output ONLY the compressed text."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original Prompt:\n{prompt}\n"
                    f"{feedback_context}\n\n"
                    f"Task: Rewrite the prompt above to be shorter but functionally equivalent.\n"
                    f"Constraint: {aggressive_instruction}\n"
                    f"Compressed Prompt:"
                )
            }
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            text_input, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(self.device)

        if deterministic:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip(), None, None
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=return_log_probs,
                return_dict_in_generate=return_log_probs
            )

        if return_log_probs:
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            logits = torch.stack(outputs.scores, dim=0)
            log_probs = F.log_softmax(logits[:, 0, :], dim=-1)
            token_log_probs = log_probs[range(len(generated_ids)), generated_ids]
            avg_log_prob = token_log_probs.mean()

            with torch.no_grad():
                full_outputs = self.model(outputs.sequences, output_hidden_states=True)
                last_hidden = full_outputs.hidden_states[-1][0, -1, :]
                value = self.value_head(last_hidden)
            
            return response.strip(), avg_log_prob, value
        else:
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip(), None, None
    
    def compute_log_probs_and_values(self, prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.last_feedback and self.iteration > 0:
            formatted_prompt = f"""Task: Compress the following prompt while preserving its meaning.

Original Prompt: {prompt}

Previous Attempt Feedback:
{self.last_feedback}

Taking this feedback into account, provide an improved compressed version:

Compressed Prompt:"""
        else:
            formatted_prompt = f"""Task: Compress the following prompt while preserving its meaning.

Original Prompt: {prompt}

Compressed Prompt:"""

        full_text = formatted_prompt + " " + response
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        outputs = self.model(**full_inputs, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        prompt_len = inputs.input_ids.shape[1]

        response_logits = logits[0, prompt_len-1:prompt_len-1+len(response_tokens), :]
        log_probs = F.log_softmax(response_logits, dim=-1)

        token_log_probs = []
        for i, token_id in enumerate(response_tokens):
            if i < log_probs.shape[0]:
                token_log_probs.append(log_probs[i, token_id])
        
        if token_log_probs:
            avg_log_prob = torch.stack(token_log_probs).mean()
        else:
            avg_log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)

        last_hidden = hidden_states[0, -1, :]
        value = self.value_head(last_hidden)
        
        return avg_log_prob, value

    def compute_reward(self, compressed_prompt: str) -> Tuple[Dict[str, float], str]:
        semantic_score = self.semantic_scorer.compute_score(
            compressed_prompt, 
            self.initial_prompt
        )

        compression_ratio = self.compression_scorer.compute_score(
            compressed_prompt,
            self.initial_prompt
        )

        compression_reward = max(0, 1.0 - compression_ratio)

        total_reward = (
            self.config.semantic_weight * semantic_score +
            self.config.compression_weight * compression_reward
        )

        feedback = self._generate_feedback(
            semantic_score, 
            compression_ratio, 
            compressed_prompt
        )
        
        reward_dict = {
            'total': float(total_reward),
            'semantic': float(semantic_score),
            'compression': float(compression_ratio),
            'compression_reward': float(compression_reward)
        }
        
        return reward_dict, feedback
    
    def _generate_feedback(
        self, 
        semantic_score: float, 
        compression_ratio: float,
        compressed_prompt: str
    ) -> str:
        original_len = len(self.initial_prompt)
        compressed_len = len(compressed_prompt)
        
        feedback_parts = []

        if semantic_score >= 0.9:
            feedback_parts.append(f" Excellent semantic preservation (score: {semantic_score:.3f}). The meaning is well maintained.")
        elif semantic_score >= 0.75:
            feedback_parts.append(f" Good semantic preservation (score: {semantic_score:.3f}), but some meaning may be lost.")
        elif semantic_score >= 0.6:
            feedback_parts.append(f" Moderate semantic preservation (score: {semantic_score:.3f}). Important details are being lost.")
        else:
            feedback_parts.append(f" Poor semantic preservation (score: {semantic_score:.3f}). Too much meaning has been lost.")

        compression_pct = (1 - compression_ratio) * 100
        if compression_ratio < 0.5:
            feedback_parts.append(f" Excellent compression ({compression_pct:.1f}% reduction, {compressed_len}/{original_len} chars).")
        elif compression_ratio < 0.7:
            feedback_parts.append(f" Good compression ({compression_pct:.1f}% reduction, {compressed_len}/{original_len} chars).")
        elif compression_ratio < 0.9:
            feedback_parts.append(f" Modest compression ({compression_pct:.1f}% reduction, {compressed_len}/{original_len} chars). Can compress more.")
        else:
            feedback_parts.append(f" Minimal compression ({compression_pct:.1f}% reduction, {compressed_len}/{original_len} chars). Compress more aggressively.")

        if semantic_score >= 0.8 and compression_ratio < 0.6:
            feedback_parts.append("→ Excellent balance! This is a high-quality compression.")
        elif semantic_score < 0.7:
            feedback_parts.append("→ Focus on preserving more of the original meaning and key details.")
        elif compression_ratio > 0.8:
            feedback_parts.append("→ Focus on being more concise. Remove redundant words and use shorter phrases.")
        else:
            feedback_parts.append("→ Try to improve both compression and semantic preservation.")
        
        return " ".join(feedback_parts)
    
    def compute_advantages(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        dones_float = dones.float()
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones_float[t]) - values[t]
            advantages[t] = last_advantage = delta + self.config.gamma * self.config.gae_lambda * (1 - dones_float[t]) * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_update(self, buffer: ExperienceBuffer):
        data = buffer.get()
        
        old_log_probs = data['log_probs'].to(self.device).detach()
        old_values = data['values'].to(self.device).detach()
        rewards = data['rewards'].to(self.device)
        dones = data['dones'].to(self.device)

        advantages, returns = self.compute_advantages(rewards, old_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config.ppo_epochs):
            new_log_probs = []
            new_values = []
            
            for prompt, response in zip(data['prompts'], data['responses']):
                log_prob, value = self.compute_log_probs_and_values(prompt, response)
                new_log_probs.append(log_prob)
                new_values.append(value)
            
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)

            if advantages.dtype != new_log_probs.dtype:
                advantages = advantages.to(new_log_probs.dtype)
            if returns.dtype != new_values.dtype:
                returns = returns.to(new_values.dtype)
            if old_log_probs.dtype != new_log_probs.dtype:
                old_log_probs = old_log_probs.to(new_log_probs.dtype)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_values, returns)

            entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()

            loss = (
                policy_loss + 
                self.config.value_loss_coef * value_loss - 
                self.config.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value_head.parameters()),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }
    
    def train_iteration(self):
        buffer = ExperienceBuffer()

        iteration_feedbacks = []
        for _ in range(self.config.batch_size):
            compressed, log_prob, value = self.generate_response(
                self.initial_prompt, 
                return_log_probs=True
            )
            
            reward_dict, feedback = self.compute_reward(compressed)
            iteration_feedbacks.append(feedback)
            
            buffer.add(
                prompt=self.initial_prompt,
                response=compressed,
                log_prob=log_prob,
                value=value,
                reward=reward_dict['total'],
                feedback=feedback,
                done=False
            )

        loss_dict = self.ppo_update(buffer)

        best_idx = torch.argmax(torch.tensor(buffer.rewards)).item()
        best_compressed = buffer.responses[best_idx]
        best_feedback = iteration_feedbacks[best_idx]
        best_reward_dict, _ = self.compute_reward(best_compressed)

        self.last_feedback = best_feedback
        
        milestone = {
            'iteration': self.iteration,
            'compressed_prompt': best_compressed,
            'score': 1.0 - best_reward_dict['total'],
            'scores': {
                'semantic': float(best_reward_dict['semantic']),
                'compression': float(best_reward_dict['compression']),
            },
            'reward': best_reward_dict['total'],
            'feedback': best_feedback,
            'loss': loss_dict
        }
        
        self.milestones.append(milestone)
        self._save_milestone(milestone)
        
        print(f"Iteration {self.iteration}: Score={milestone['score']:.4f}, "
              f"Compression={best_reward_dict['compression']:.4f}, "
              f"Semantic={best_reward_dict['semantic']:.4f}")
        print(f"Feedback: {best_feedback}")
        
        self.iteration += 1
        buffer.clear()
    
    def _save_milestone(self, milestone: Dict):
        log_file = self.run_dir / "milestones.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(milestone) + "\n")
    
    def save_checkpoint(self):
        checkpoint_dir = self.run_dir / f"checkpoint_{self.iteration}"
        checkpoint_dir.mkdir(exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)

        torch.save(
            self.value_head.state_dict(),
            checkpoint_dir / "value_head.pt"
        )
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def train(self):
        print(f"Starting training for {self.config.num_iterations} iterations")
        print(f"Output directory: {self.run_dir}")
        
        for iteration in range(self.config.num_iterations):
            self.train_iteration()
            
            if (iteration + 1) % self.config.save_every == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        print(f"Training complete! Results saved to {self.run_dir}")

        if self.milestones:
            best_milestone = max(self.milestones, key=lambda x: x['reward'])
            
            print("\n" + "="*50)
            print("TRAINING COMPLETE - BEST RESULT")
            print("="*50)
            print(f"Iteration: {best_milestone['iteration']}")
            print(f"Score: {best_milestone['score']:.4f} (Lower is better)")
            print(f"Reward: {best_milestone['reward']:.4f}")
            print(f"Compression Score: {best_milestone['scores']['compression']:.4f}")
            print(f"Semantic Score: {best_milestone['scores']['semantic']:.4f}")
            print("-"*20)
            print(f"Original Prompt:\n{self.initial_prompt}")
            print("-"*20)
            print(f"Best Compressed Prompt:\n{best_milestone['compressed_prompt']}")
            print("="*50 + "\n")

            best_result_path = self.run_dir / "best_result.json"
            with open(best_result_path, "w") as f:
                json.dump(best_milestone, f, indent=2)
            print(f"Best result details saved to {best_result_path}")

if __name__ == "__main__":
    config = PPOConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        num_iterations=15,
        batch_size=4,
        output_dir="../runs_li"
    )
    
    with open('../data/long_prompts.json', 'r') as f:
        prompts = json.load(f)

    for prompt in prompts:
        print(f"Compressing: {prompt}")
        trainer = PromptCompressionPPO(
            config=config,
            initial_prompt=prompt
        )
        
        trainer.train()

    print(f"Training complete! Check {trainer.run_dir} for results")