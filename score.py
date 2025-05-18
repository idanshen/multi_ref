#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Literal

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_util import (
    compute_maj_pred,
    compute_naive_pred,
    compute_weighted_pred,
    extract_completion_answers,
    extract_answer_probabilities,
    compute_soft_bon_pred,
    subsample_completions,
)


def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


def score(dataset: Dataset, model_name: str, num_proc: int = 1) -> Dataset:
    # dataset = dataset.map(
    #     lambda x: {"agg_scores": [aggregate_scores(s, "last") for s in x["scores"]]}
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    system_prompt = """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."""
    probs = []
    for i in tqdm(range(len(dataset)), desc="Extract answer probabilities"):
        result = extract_answer_probabilities(
            dataset[i],
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt
        )
        probs.append(result['probs'])
    dataset = dataset.add_column("probs", probs)
    dataset = dataset.map(
        extract_completion_answers,
        fn_kwargs={},
        num_proc=num_proc,
        desc=f"Extract answers",
    )
    subsets = [2**i for i in range(1024) if 2**i <= 1024]
    for n in tqdm(subsets, desc="Computing predictions"):
        dataset = dataset.map(
            subsample_completions,
            fn_kwargs={"n": n},
            num_proc=num_proc,
            desc=f"Subsample {n}",
        )
        dataset = dataset.map(
            compute_soft_bon_pred,
            fn_kwargs={"beta": 0.0, "n": n},
            num_proc=num_proc,
            desc=f"Compute soft BON pred",
        )
        dataset = dataset.map(
            compute_soft_bon_pred,
            fn_kwargs={"beta": 5.0, "n": n},
            num_proc=num_proc,
            desc=f"Compute soft BON pred",
        )
        dataset = dataset.map(
            compute_soft_bon_pred,
            fn_kwargs={"beta": 10.0, "n": n},
            num_proc=num_proc,
            desc=f"Compute soft BON pred",
        )
        dataset = dataset.map(
            compute_soft_bon_pred,
            fn_kwargs={"beta": 15.0, "n": n},
            num_proc=num_proc,
            desc=f"Compute soft BON pred",
        )
        dataset = dataset.map(
            compute_soft_bon_pred,
            fn_kwargs={"beta": 20.0, "n": n},
            num_proc=num_proc,
            desc=f"Compute soft BON pred",
        )
        dataset = dataset.map(
            compute_soft_bon_pred,
            fn_kwargs={"beta": 25.0, "n": n},
            num_proc=num_proc,
            desc=f"Compute soft BON pred",
        )
        dataset = dataset.map(
            compute_soft_bon_pred,
            fn_kwargs={"beta": 30.0, "n": n},
            num_proc=num_proc,
            desc=f"Compute soft BON pred",
        )
        # dataset = dataset.map(
        #     compute_weighted_pred,
        #     fn_kwargs={"n": n},
        #     num_proc=config.num_proc,
        #     desc=f"Compute weighted pred {n}",
        # )
        # dataset = dataset.map(
        #     compute_maj_pred,
        #     fn_kwargs={"n": n},
        #     num_proc=config.num_proc,
        #     desc=f"Compute majority pred {n}",
        # )
        # dataset = dataset.map(
        #     compute_naive_pred,
        #     fn_kwargs={"n": n},
        #     num_proc=config.num_proc,
        #     desc=f"Compute naive pred {n}",
        # )
        # # Nuke unused columns to keep dataset lean
        # dataset = dataset.remove_columns(
        #     [f"completions@{n}", f"agg_scores@{n}", f"preds@{n}"]
        # )
    return dataset

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_name = "HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions"
    dataset_subset = "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-1024--max_tokens-2048--bsz-8--seed-0--agg_strategy-last"

    dataset = load_dataset(dataset_name, dataset_subset)
    dataset = dataset["train"]

    d = score(dataset, model_name, num_proc=1)
    d.save_to_disk("dataset/test_2")
