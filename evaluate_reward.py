import torch
import numpy as np
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
device = "cuda:0"
rm_model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModelForSequenceClassification.from_pretrained(
    rm_model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_name)

model_path = "Qwen/Qwen2.5-0.5B-Instruct"# "/data/pulkitag/models/idanshen/multi_ref/dpo_output/multi_ref_model_1.0"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

generate_kwargs = {
    "max_new_tokens": 500,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 0,
}

# Load the UltraFeedback binarized dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized")

# Get the test split
test_dataset = dataset["test"]

# Initialize a list to store scores
scores = []

# Process examples from the test split
for example in tqdm(test_dataset, desc="Generating and scoring responses"):
    prompt = example["chosen"][0]["content"]
    
    # Prepare conversation format for the model
    conv = [{"role": "user", "content": prompt}]
    conv_tokenized = tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
    
    # Generate response
    response = model.generate(conv_tokenized, **generate_kwargs)
    
    # Get the length of the prompt tokens to know where the response starts
    prompt_length = conv_tokenized.shape[1]
    
    # Decode only the new tokens (excluding prompt)
    response_text = tokenizer.decode(response[0][prompt_length:], skip_special_tokens=True)
    
    # Prepare conversation with response for scoring
    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_text[response_text.find(":")+1:]}]
    conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
    
    # Get the reward score
    with torch.no_grad():
        score = rm(conv_tokenized).logits[0][0].item()
    
    scores.append(score)
    
    # Print progress
    if len(scores) % 10 == 0:
        print(f"Processed {len(scores)} examples. Current mean score: {np.mean(scores):.4f}")

# Convert to numpy array for statistics
scores_array = np.array(scores)

# Calculate statistics
mean_score = np.mean(scores_array)
std_score = np.std(scores_array)
median_score = np.median(scores_array)
min_score = np.min(scores_array)
max_score = np.max(scores_array)

# Print statistics
print(f"\nResults Summary:")
print(f"Mean score: {mean_score:.4f}")
print(f"Standard deviation: {std_score:.4f}")
print(f"Median score: {median_score:.4f}")
print(f"Min score: {min_score:.4f}")
print(f"Max score: {max_score:.4f}")

# Save scores to file
save_path = os.path.join(model_path, "reward_scores.npy")
np.save(save_path, scores_array)
print(f"Scores saved to {save_path}")

# Also save a text file with summary statistics
stats_path = os.path.join(model_path, "reward_stats.txt")
with open(stats_path, "w") as f:
    f.write(f"Dataset: trl-lib/ultrafeedback_binarized (test split)\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Reward model: {rm_model_name}\n")
    f.write(f"Number of examples: {len(scores_array)}\n")
    f.write(f"Mean score: {mean_score:.4f}\n")
    f.write(f"Standard deviation: {std_score:.4f}\n")
    f.write(f"Median score: {median_score:.4f}\n")
    f.write(f"Min score: {min_score:.4f}\n")
    f.write(f"Max score: {max_score:.4f}\n")

print(f"Statistics saved to {stats_path}")
