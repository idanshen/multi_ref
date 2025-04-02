import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from copy import deepcopy


class EnsembleModel(GenerationMixin, torch.nn.Module):
    """
    Ensemble model that takes two models and produce output from the geometric mean of their distributions.
    """
    def __init__(self, model1, model2, ensemble_type="geometric", alpha=0.5):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.ensemble_type = ensemble_type
        self.alpha = alpha  # Weight for model1 (1-alpha for model2)
        self.device = self.model1.device
        self.config = model1.config
        self.can_generate = model1.can_generate
        self.generation_config = model1.generation_config
        self.main_input_name = model1.main_input_name
        
        # Add required attributes for cache support
        self._supports_cache_class = model1._supports_cache_class
        self.is_loaded_in_8bit = getattr(model1, "is_loaded_in_8bit", False)
        self.is_loaded_in_4bit = getattr(model1, "is_loaded_in_4bit", False)
        
        if hasattr(model1, "adjust_logits_during_generation"):
            self.adjust_logits_during_generation = model1.adjust_logits_during_generation

    def forward(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        """
        Expects past_key_values (if provided) to be a tuple:
            past_key_values = (past_from_model1, past_from_model2)
        Returns output on model1's device.
        """
        # Split past_key_values for each model if provided
        if past_key_values is None:
            past1 = None
            past2 = None
        else:
            past1, past2 = past_key_values

        # Remove problematic kwargs
        kwargs = {k: v for k, v in kwargs.items() 
                 if k not in ['past_key_values', 'attention_mask']}
        # Map cache_position to correct device if present
        if 'cache_position' in kwargs:
            kwargs1 = {**kwargs, 'cache_position': kwargs['cache_position'].to(self.model1.device)}
            kwargs2 = {**kwargs, 'cache_position': kwargs['cache_position'].to(self.model2.device)}
        else:
            kwargs1 = kwargs
            kwargs2 = kwargs

        # Forward pass for each model without attention mask
        outputs1 = self.model1(
            input_ids=input_ids.to(self.model1.device),
            past_key_values=past1,
            use_cache=False,
            **kwargs1
        )
        
        outputs2 = self.model2(
            input_ids=input_ids.to(self.model2.device),
            past_key_values=past2,
            use_cache=False,
            **kwargs2
        )

        # Obtain the raw logits and move them to model1's device
        logits1 = outputs1.logits
        logits2 = outputs2.logits

        # Convert logits to log-probabilities
        log_probs1 = F.log_softmax(logits1, dim=-1)
        log_probs2 = F.log_softmax(logits2, dim=-1)
        print(log_probs1.shape)
        # Compute weighted combination in log-space
        if self.ensemble_type == "geometric":
            avg_log_probs = self.alpha * log_probs1 + (1 - self.alpha) * log_probs2
        elif self.ensemble_type == "arithmetic":
            avg_log_probs = torch.logsumexp(torch.stack([log_probs1.to(log_probs2.device) + torch.log(torch.tensor(self.alpha, device=log_probs2.device)), log_probs2 + torch.log(torch.tensor(1 - self.alpha, device=log_probs2.device))]), dim=0)
        else:
            raise ValueError(f"Invalid ensemble type: {self.ensemble_type}")

        # Merge the caches from both models
        new_past = (outputs1.past_key_values, outputs2.past_key_values)

        # Return a CausalLMOutputWithPast. The generation routines only care about relative logits.
        return CausalLMOutputWithPast(logits=avg_log_probs, past_key_values=new_past)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """
        Simplified version for debugging
        """
        if past_key_values is None or type(past_key_values) != tuple:
            # Create separate copies for each model on their respective devices
            past_key_values = (
                deepcopy(past_key_values).to(self.model1.device) if past_key_values is not None else None,
                deepcopy(past_key_values).to(self.model2.device) if past_key_values is not None else None
            )
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            **kwargs
        }
