from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
from ensemble_model import EnsembleModel

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and score responses using LLM and reward model')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help='Name or path of the base LLM')
    parser.add_argument('--model_name2', type=str, default=None,
                        help='Name or path of the second LLM for ensemble')
    parser.add_argument('--ensemble_alpha', type=float, default=0.5,
                        help='Weight for first model in ensemble (between 0 and 1)')
    parser.add_argument('--reward_model', type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
                        help='Name or path of the reward model')
    parser.add_argument('--dataset', type=str, default="ibm-research/AttaQ",
                        help='Dataset to use for prompts')
    parser.add_argument('--k', type=int, default=25,
                        help='Number of responses to generate per prompt')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    if args.model_name2:
        print("Loading ensemble models...")
        model1 = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="cuda:0",  # First GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model2 = AutoModelForCausalLM.from_pretrained(
            args.model_name2,
            device_map="cuda:1",  # Second GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model = EnsembleModel(model1, model2, alpha=args.ensemble_alpha)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="cuda:0",  # First GPU
            trust_remote_code=True
        )

    # Load ArmoRM model and tokenizer
    print("Loading ArmoRM model and tokenizer...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, 
        device_map="cuda:0",  # Second GPU
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)

    # Add CUDA availability check
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This script requires at least 2 CUDA GPUs to run")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset)

    # Process each prompt and generate responses
    all_scores = []
    for item in tqdm(dataset["train"].select(range(150)), desc="Generating responses"):
        prompt = item["input"]
        # Apply chat template to format the prompt properly
        formatted_prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ], tokenize=False)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        all_responses = []
        remaining = 25
        while remaining > 0:
            batch_size = min(args.k, remaining)
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                num_return_sequences=batch_size,
                pad_token_id=tokenizer.pad_token_id,
            )
            all_responses.extend(outputs)
            remaining -= batch_size
        outputs = all_responses
        
        batch_scores = []
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from response
            response = response[response.rindex('\nuser\n')+5:].strip()
            
            # Score the response using ArmoRM
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            reward_inputs = reward_tokenizer.apply_chat_template(messages, return_tensors="pt").to(reward_model.device)
            with torch.no_grad():
                reward_output = reward_model(reward_inputs)
                preference_score = reward_output.score.cpu().float().item()
            batch_scores.append(preference_score)
        
        all_scores.append(batch_scores)
    
    # Save all scores at the end
    model_name_short = args.model_name.split('/')[-1]
    if args.model_name2:
        model2_short = args.model_name2.split('/')[-1]
        model_name_short = f"{model_name_short}+{model2_short}_a{args.ensemble_alpha}"
    reward_model_short = args.reward_model.split('/')[-1]
    dataset_name_short = args.dataset.split('/')[-1]
    filename = f"scores_{model_name_short}_{reward_model_short}_{dataset_name_short}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)

if __name__ == "__main__":
    main()
