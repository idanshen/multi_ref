import os
import re
import json
import torch
from typing import List, Dict, Any
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

###############################################################################
# Replicating the relevant logic from gsm8k_grpo.py
###############################################################################

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
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

def get_gsm8k_questions(split: str = "test"):
    data = load_dataset("openai/gsm8k", "main")[split]
    # Use the same transformation as gsm8k_grpo
    data = data.map(
        lambda x: {
            "prompt": [
               {"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": x["question"]}
            ],
            "answer": extract_hash_answer(x["answer"])
        }
    )
    return data

# The same reward functions from gsm8k_grpo.py:
def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has a strict format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has a looser XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


###############################################################################
# Main evaluation script
###############################################################################

def main():
    """
    - Loads the trained checkpoint folder (which should contain the model weights).
    - Uses vLLM to generate responses for the GSM8K test set.
    - Evaluates each completion with the same reward functions used during training.
    - Saves the resulting average rewards and a small sample of generations to a JSON file in the same directory.
    """

    # -------------------------------------------------------------------------
    # 1. Configuration: adjust paths/devices as needed
    # -------------------------------------------------------------------------
    checkpoint_path = "outputs/Qwen-0.5B-GRPO-multi_ref"  # Path to your GRPO checkpoint
    # e.g. "outputs/Qwen-0.5B-GRPO-multi_ref" or "my-checkpoint-folder"

    device = "cuda:0"   # If you have only 1 GPU, or "cuda:1" for the second GPU, etc.
    gpu_memory_ratio = 0.3  # Adjust if you'd like to reduce VRAM usage by vLLM
    max_model_len = 4096    # You can match this to your training config
    temperature = 1.0
    num_generations = 3     # The same number used in training, if relevant
    max_new_tokens = 200    # Same or similar to what was used during training

    # -------------------------------------------------------------------------
    # 2. Load the tokenizer and vLLM LLM
    # -------------------------------------------------------------------------
    # The "checkpoint_path" should contain the model's weights. If your final
    # merges were stored in another directory, adjust accordingly.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Instantiate the LLM from vLLM
    llm = LLM(
        model=checkpoint_path,
        device=device,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_ratio,
        max_model_len=max_model_len,
        enable_prefix_caching=True,  # speeds if same prompts are repeated
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=num_generations,  # how many completions per prompt
    )

    # -------------------------------------------------------------------------
    # 3. Load test data
    # -------------------------------------------------------------------------
    dataset = get_gsm8k_questions("test")
    # We might subset the dataset for demonstration:
    # dataset = dataset.select(range(50))  # for a quick test

    # -------------------------------------------------------------------------
    # 4. Generate completions with vLLM
    # -------------------------------------------------------------------------

    # vLLM expects a list of text prompts, so we must format them
    # consistently with how it was done in training.
    def apply_chat_template(sample: Dict[str, Any]) -> str:
        """Convert the conversation to text in a simplistic manner."""
        messages = sample["prompt"]
        text_list = []
        for msg in messages:
            if msg["role"] == "system":
                text_list.append("[system] " + msg["content"].strip() + "\n")
            elif msg["role"] == "user":
                text_list.append("[user] " + msg["content"].strip() + "\n")
            else:
                text_list.append(f"[{msg['role']}] {msg['content'].strip()}\n")
        return "".join(text_list)

    # Prepare the prompts
    all_prompts = [apply_chat_template(example) for example in dataset]

    # Generate with vLLM (in batches to avoid OOM for large test sets)
    batch_size = 16  # Adjust as needed
    completions_list = []
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i : i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        # Each item in outputs is an "RequestOutput" with `prompts` and `outputs`
        for out in outputs:
            # vLLM returns multiple completions (n = sampling_params.n)
            # out.outputs is a list of GenerationOutput
            batch_completions = []
            for gen_out in out.outputs:
                # decode the tokens if needed
                text = tokenizer.decode(gen_out.token_ids[len(gen_out.prompt_token_ids) :], skip_special_tokens=True)
                # Format as a chat message
                batch_completions.append([{"role": "assistant", "content": text}])
            completions_list.extend(batch_completions)

    # completions_list is length = len(all_prompts)*num_generations, but we want
    # them shaped so each prompt has 'num_generations' different completions.
    # to keep consistent, group them:
    grouped_completions = []
    for i in range(0, len(completions_list), num_generations):
        grouped_completions.append(completions_list[i : i + num_generations])

    # -------------------------------------------------------------------------
    # 5. Compute rewards
    # -------------------------------------------------------------------------
    # We'll use the same reward functions. They accept lists of prompts, completions, etc.
    # We'll iterate prompt-by-prompt, using each group of completions.

    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]
    reward_names = [
        "xmlcount",
        "soft_format",
        "strict_format",
        "int_reward",
        "correctness",
    ]

    all_rewards = {name: [] for name in reward_names}

    # The dataset has the "answer" for each entry, which we'll pass to correctness_reward_func
    for i, example in enumerate(dataset):
        prompt_group = [example["prompt"]] * num_generations
        completions_group = grouped_completions[i]
        # Each reward function can expect prompts=..., completions=..., answer=...
        # correctness expects the `answer` in a list with the same length as the number of generations
        # so replicate it:
        answer_list = [example["answer"]] * num_generations

        # Evaluate each reward function
        for func, name in zip(reward_funcs, reward_names):
            # Some reward funcs rely on "completions" only, others rely on "answer" as well
            if name == "correctness":
                vals = func(prompt_group, completions_group, answer_list)
            else:
                vals = func(completions_group)
            # We'll just average among the generations
            avg_val = sum(vals) / len(vals)
            all_rewards[name].append(avg_val)

    # Summarize the results
    results_summary = {}
    for name in reward_names:
        rewards_for_all_prompts = all_rewards[name]
        avg_reward = sum(rewards_for_all_prompts) / len(rewards_for_all_prompts)
        results_summary[name] = {
            "average_reward": avg_reward,
            "all_rewards": rewards_for_all_prompts,  # you could omit if huge
        }

    # -------------------------------------------------------------------------
    # 6. Save results to a JSON file in the same folder as your checkpoint
    # -------------------------------------------------------------------------
    save_path = os.path.join(checkpoint_path, "evaluation_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to {save_path}")

    # Optionally, print out a small sample
    print("\nSample results:")
    for rn in reward_names:
        print(f"  {rn}: {results_summary[rn]['average_reward']:.4f}")

if __name__ == "__main__":
    main() 