import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_peft_config,
)
from transformers import PretrainedConfig
from types import SimpleNamespace


@dataclass
class PromptMinimizationArgs:    
    compression_ratio_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for compression ratio reward (higher = prioritize compression more)"}
    )
    quality_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for semantic similarity reward (higher = prioritize quality more)"}
    )
    num_prompts: int = field(
        default=100,
        metadata={"help": "Number of prompts to use for training"}
    )
    target_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to target model for evaluating compressed prompts"}
    )
    quality_eval_frequency: int = field(
        default=10,
        metadata={"help": "Evaluate quality every N steps (expensive operation)"}
    )
    max_new_tokens: int = field(
        default=30,
        metadata={"help": "Number of new tokens to generate"}
    )


def create_sample_dataset(num_prompts=100):
    # dummy dataset
    base_prompts = [
        "Please provide a detailed explanation of what machine learning is, including its key concepts and applications.",
        "I would like to know the capital city of France. Can you tell me what it is?",
        "Can you explain in detail how the process of photosynthesis works in plants?",
        "I'm interested in understanding the water cycle. Could you describe the various stages involved?",
        "What are the main programming paradigms used in software development? Please list and briefly explain each.",
        "Could you give me a comprehensive definition of artificial intelligence and its subfields?",
        "I'd like to understand Einstein's theory of relativity. Can you explain it in simple terms?",
        "Please describe how modern computers work, including the role of the CPU and memory.",
        "Can you explain the fundamental concept of gravity and how it affects objects?",
        "What is quantum mechanics? Please provide an overview suitable for beginners.",
        "I want to understand how the internet works. Can you explain the basic architecture and protocols?",
        "Could you describe what climate change is and what causes it?",
        "Please tell me about our solar system, including the planets and their characteristics.",
        "What are the three laws of thermodynamics? Can you explain each one?",
        "How does DNA store genetic information? Please explain the structure and function.",
        "I'm curious about the stock market. Can you explain how it works and what affects stock prices?",
        "Could you provide an explanation of blockchain technology and how it's used in cryptocurrencies?",
        "What is Darwin's theory of natural selection? Please explain how evolution works.",
        "How do vaccines work to protect against diseases? Please explain the mechanism.",
        "Can you describe the scientific method and why it's important for research?",
    ]
    
    prompts = (base_prompts * ((num_prompts // len(base_prompts)) + 1))[:num_prompts]
    
    return Dataset.from_dict({"prompt": prompts})


class DummyBackbone(torch.nn.Module):
    # Dummy backbone that mimics a transformer model for TRL compatibility
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.dummy = torch.nn.Linear(1, 1)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)

        return SimpleNamespace(
            last_hidden_state=hidden_states,
            hidden_states=(hidden_states,)
        )


class PromptCompressionRewardModel(torch.nn.Module):
    #should eventually import from metrics/ folder
    def __init__(self, target_model, tokenizer, compression_ratio_weight=1.0, 
                 quality_weight=1.0, quality_eval_frequency=10):
        super().__init__()
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.compression_ratio_weight = compression_ratio_weight
        self.quality_weight = quality_weight
        self.quality_eval_frequency = quality_eval_frequency

        self.original_prompts = {}
        self.step_counter = 0

        self.config = PretrainedConfig()
        self.config.hidden_size = 768
        self.base_model_prefix = 'model'
        self.model = DummyBackbone(hidden_size=768)
        self.score = torch.nn.Linear(self.config.hidden_size, 1, bias=False)
    
    def set_original_prompts(self, batch_idx, prompts):
        for i, prompt in enumerate(prompts):
            self.original_prompts[f"{batch_idx}_{i}"] = prompt
    
    def clean_compressed_output(self, text):
        text = text.split('\n')[0].strip()

        if '→' in text:
            text = text.split('→')[0].strip()

        text = text.strip('"\'')

        prefixes = ['Compressed:', 'Output:', 'Result:']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text
    
    def compute_semantic_similarity(self, compressed_text, original_text):
        try:
            with torch.no_grad():
                orig_inputs = self.tokenizer(
                    original_text, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.target_model.device)
                
                orig_outputs = self.target_model.generate(
                    **orig_inputs,
                    max_new_tokens=30, #could increase this
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                comp_inputs = self.tokenizer(
                    compressed_text, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.target_model.device)
                
                comp_outputs = self.target_model.generate(
                    **comp_inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                orig_tokens = set(orig_outputs[0].cpu().tolist())
                comp_tokens = set(comp_outputs[0].cpu().tolist())
                
                if len(orig_tokens) == 0 and len(comp_tokens) == 0:
                    return 1.0
                
                intersection = len(orig_tokens.intersection(comp_tokens))
                union = len(orig_tokens.union(comp_tokens))
                
                similarity = intersection / union if union > 0 else 0.0
                return similarity
                
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def forward(self, input_ids, attention_mask=None, original_prompts=None, **kwargs):
        batch_size = input_ids.shape[0]
        rewards = []
        self.step_counter += 1

        compute_quality = (self.step_counter % self.quality_eval_frequency == 0)
        
        for i in range(batch_size):
            if attention_mask is not None:
                output_length = attention_mask[i].sum().item()
            else:
                output_length = input_ids.shape[1]

            compressed_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            compressed_text = self.clean_compressed_output(compressed_text)

            output_length = len(self.tokenizer.encode(compressed_text, add_special_tokens=False))

            if original_prompts and i < len(original_prompts):
                original_text = original_prompts[i]
                original_length = len(self.tokenizer.encode(original_text, add_special_tokens=False))
            else:
                #fallback
                original_length = 100

            compression_ratio = max(0.0, (original_length - output_length) / original_length)
            compression_reward = compression_ratio * self.compression_ratio_weight

            if compute_quality and original_prompts and i < len(original_prompts):
                quality_score = self.compute_semantic_similarity(compressed_text, original_prompts[i])
                quality_reward = quality_score * self.quality_weight
                
                if i == 0:
                    print(f"\n[Quality Check] Original: '{original_prompts[i][:50]}...'")
                    print(f"[Quality Check] Compressed: '{compressed_text[:50]}...'")
                    print(f"[Quality Check] Similarity: {quality_score:.3f}, Compression: {compression_ratio:.3f}")
            else:
                # fallback
                quality_reward = 0.5 * self.quality_weight
            
            total_reward = compression_reward + quality_reward
            rewards.append([total_reward])

        logits = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
        
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(logits=logits)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig, PromptMinimizationArgs))
    script_args, training_args, model_args, prompt_args = parser.parse_args_into_dataclasses()

    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    model_name = model_args.model_name_or_path or "EleutherAI/pythia-6.9b"

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    is_chat_model = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    print(f"Chat model detected: {is_chat_model}")

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype,
        trust_remote_code=model_args.trust_remote_code,
        device_map="auto",
    )

    sft_model_path = training_args.sft_model_path or model_name
    print(f"Loading policy model from {sft_model_path}...")
    policy = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        **model_kwargs
    )
    print(f"Policy model loaded. Vocab size: {policy.config.vocab_size}")

    peft_config = get_peft_config(model_args)

    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            **model_kwargs
        )
    else:
        ref_policy = None
        print("Using PEFT (LoRA) - no separate reference model needed")

    target_model_path = prompt_args.target_model_path or model_name
    print(f"Loading target model from {target_model_path}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        **model_kwargs
    )
    target_model.eval()

    print("Creating prompt compression reward model...")
    reward_model = PromptCompressionRewardModel(
        target_model=target_model,
        tokenizer=tokenizer,
        compression_ratio_weight=prompt_args.compression_ratio_weight,
        quality_weight=prompt_args.quality_weight,
        quality_eval_frequency=prompt_args.quality_eval_frequency,
    )

    print("Loading value model...")
    value_model_path = training_args.reward_model_path or "EleutherAI/pythia-160m"
    try:
        value_model = AutoModelForSequenceClassification.from_pretrained(
            value_model_path,
            trust_remote_code=model_args.trust_remote_code,
            num_labels=1,
        )
    except:
        print(f"Could not load {value_model_path} as sequence classifier, using policy model base...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(sft_model_path)
        config.num_labels = 1
        value_model = AutoModelForSequenceClassification.from_pretrained(
            sft_model_path,
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=model_args.trust_remote_code,
        )

    if script_args.dataset_name:
        from datasets import load_dataset
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
            split=script_args.dataset_train_split
        )
        dataset_text_field = "prompt"
    else:
        dataset = create_sample_dataset(prompt_args.num_prompts)
        dataset_text_field = "prompt"
        print(f"Using sample dataset with {len(dataset)} prompts")

    eval_samples = min(20, len(dataset) // 10)
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

    def prepare_dataset(dataset, tokenizer, is_chat_model):
        def tokenize(element):
            prompts = element[dataset_text_field]
            original_prompts_list = prompts.copy()

            original_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

            formatted_prompts = []
            #few shot prompt with clear stop →
            for p in prompts:
                few_shot_template = """Compress these prompts by removing unnecessary words:

"Please provide a detailed explanation of what machine learning is" → "Explain machine learning"
"I would like to know the capital city of France" → "Capital of France?"
"{prompt}" →"""
                
                formatted = few_shot_template.format(prompt=p)
                formatted_prompts.append(formatted)

            if is_chat_model:
                chat_prompts = []
                for prompt in formatted_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    chat_prompts.append(formatted)
                formatted_prompts = chat_prompts
            
            outputs = tokenizer(
                formatted_prompts,
                padding=False,
                truncation=True,
                max_length=512,
            )

            outputs["original_prompt_text"] = original_prompts_list
            outputs["original_length"] = original_lengths
            
            return outputs

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer, is_chat_model)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer, is_chat_model)

    train_original_prompts = train_dataset["original_prompt_text"]

    if "original_prompt_text" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(["original_prompt_text", "original_length"])
    if "original_prompt_text" in eval_dataset.column_names:
        eval_dataset = eval_dataset.remove_columns(["original_prompt_text", "original_length"])

    print("\n" + "="*50)
    print("Testing generation BEFORE training:")
    print("="*50)
    test_prompt = "Please provide a detailed explanation of what machine learning is, including its key concepts and applications."
    
    few_shot_template = """Compress these prompts by removing unnecessary words:

"Please provide a detailed explanation of what machine learning is" → "Explain machine learning"
"I would like to know the capital city of France" → "Capital of France?"
"{prompt}" →"""
    
    formatted_test = few_shot_template.format(prompt=test_prompt)
    
    if is_chat_model:
        messages = [{"role": "user", "content": formatted_test}]
        formatted_test = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    test_inputs = tokenizer(formatted_test, return_tensors="pt")
    test_inputs = {k: v.to(policy.device) for k, v in test_inputs.items()}
    
    print(f"Original prompt ({len(tokenizer.encode(test_prompt))} tokens): {test_prompt}")
    print(f"\nGenerating compressed version...")
    with torch.no_grad():
        test_outputs = policy.generate(
            **test_inputs,
            max_new_tokens=prompt_args.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            stop_strings=["\n", "→"],
            tokenizer=tokenizer,
        )

    generated_ids = test_outputs[0][len(test_inputs['input_ids'][0]):]
    compressed = tokenizer.decode(generated_ids, skip_special_tokens=True)

    compressed = compressed.split('\n')[0].strip()
    if '→' in compressed:
        compressed = compressed.split('→')[0].strip()
    compressed = compressed.strip('"\'')
    
    compressed_length = len(tokenizer.encode(compressed, add_special_tokens=False))
    original_length = len(tokenizer.encode(test_prompt, add_special_tokens=False))
    compression_ratio = (original_length - compressed_length) / original_length if original_length > 0 else 0
    
    print(f"Compressed prompt ({compressed_length} tokens): {compressed}")
    print(f"Compression ratio: {compression_ratio:.2%} (original: {original_length} → compressed: {compressed_length})")
    print("="*50 + "\n")

    class CustomPPOTrainer(PPOTrainer):
        def __init__(self, *args, original_prompts=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.original_prompts = original_prompts
            self.batch_counter = 0
        
        def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
            """Override to pass original prompts to reward model"""
            return super().compute_rewards(scores, logprobs, ref_logprobs, masks)
    
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    reward_model.original_prompts_list = train_original_prompts
    
    print("Starting PPO training for prompt compression...")
    print(f"Compression ratio weight: {prompt_args.compression_ratio_weight}")
    print(f"Quality weight: {prompt_args.quality_weight}")
    print(f"Quality eval frequency: every {prompt_args.quality_eval_frequency} steps")
    print(f"Goal: Maximize compression ratio while maintaining quality")
    print(f"KL coefficient: {training_args.kl_coef}")
    print(f"Learning rate: {training_args.learning_rate}")
    
    trainer.train()

    # print(f"\nSaving model to {training_args.output_dir}")
    # trainer.save_model(training_args.output_dir)

    # print("\nGenerating sample compressed prompts...")
    # trainer.generate_completions()