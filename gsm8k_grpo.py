import re
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpo_trainer import GRPOTrainer
from grpo_config import GRPOConfig
from ensemble_model import EnsembleModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
ref_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

output_dir="outputs/Qwen-0.5B-GRPO-multi_ref"
run_name="Qwen-0.5B-GRPO-gsm8k-multi_ref"

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='GRPO Training Arguments')
    parser.add_argument('--multi_ref', type=bool, default=False, help='Use multiple reference models')
    parser.add_argument('--ensemble_type', type=str, default="geometric", help='Ensemble type, can be "geometric" or "arithmetic"')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for the ensemble')
    parser.add_argument('--num_iterations', type=int, default=4, help='Number of training iterations')
    parser.add_argument('--max_completion_length', type=int, default=1000, help='Maximum completion length')
    parser.add_argument('--num_generations', type=int, default=8, help='Number of generations')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    return parser.parse_args()

# Replace the hardcoded values with args
args = parse_args()

training_args = GRPOConfig(
    seed=0,
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=args.train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_generations=args.num_generations,
    max_prompt_length=256,
    max_completion_length=args.max_completion_length,
    num_train_epochs=1,
    save_steps=300,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=True,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda:0",
    report_to="wandb",
    num_iterations=args.num_iterations,
    multi_ref=args.multi_ref,
    ensemble_type=args.ensemble_type,
    alpha=args.alpha,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda:0")

if args.multi_ref:
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:1",  # First GPU
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model2 = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        device_map="cuda:1",  # Second GPU
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    ref_model = EnsembleModel(model1, model2, ensemble_type=args.ensemble_type, alpha=args.alpha)
else:
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
    ref_model=ref_model,
    #peft_config=peft_config
)
trainer.train()