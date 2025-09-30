import profiling_decorator

import copy
import inspect
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sequence, Sized
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
#from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    PreTrainedTokenizer,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache


class HFRepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatSampler(
    ...     ["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4
    ... )
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count




class ReToolTrainer(Trainer):  # Change this line
    
    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        args: Optional[transformers.TrainingArguments] = None,
        reward_funcs: Optional[list[Callable]] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        # ReTool specific parameters - same as before
        eos_id: Optional[int] = None,
        interpreter_id: Optional[list[int]] = None,
        code_id: Optional[list[int]] = None,
        max_turns: int = 10,
        max_completion_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        min_p: Optional[float] = None,
        mask_truncated_completions: bool = True,
        **kwargs
    ):
        # Initialize parent Trainer (simpler call)        
        super().__init__(
            model=model,
            args=args,
            tokenizer=processing_class,  # Note: Trainer uses 'tokenizer', not 'processing_class'
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs
        )

        # Store processing_class for compatibility
        self.processing_class = processing_class or self.tokenizer
        
        # Processing class
        if processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        else:
            # Store processing_class for compatibility
            self.processing_class = processing_class or self.tokenizer
        if processing_class.pad_token is None:
            self.processing_class.pad_token = processing_class.eos_token

        
        # Add reward function handling (since Trainer doesn't have this)
        self.reward_funcs = reward_funcs or [self._binary_reward_function]

        # ReTool specific attributes
        self.eos_id = eos_id or self.processing_class.eos_token_id
        self.interpreter_id = interpreter_id or self._get_interpreter_token_ids()
        self.code_id = code_id or self._get_code_token_ids()
        self.max_turns = max_turns
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.mask_truncated_completions = mask_truncated_completions
        
        # ReTool specific logging
        self.reward_func_names = ["binary_correctness"]
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._textual_logs = {
            "prompt": [],
            "completion": [],
            "rewards": {"binary_correctness": []}
        }
        
        # Generation configuration for ReTool
        self.generation_config = GenerationConfig(
            max_new_tokens=50,  # Per turn, not total
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            eos_token_id=self.eos_id,  # default stop on EOS
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            return_dict_in_generate=True,
            use_cache=True,
            cache_implementation=args.cache_implementation, #args.cache_implementation = 'Offloaded Cache'
        )
    def _set_signature_columns_if_needed(self):
    # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
    # By default, this method sets `self._signature_columns` to the model's expected inputs.
    # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
    # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image"]
    
    def _get_train_sampler(self, dataset=None):
        """Override to use RepeatSampler for GRPO."""
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
            
        return HFRepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,  # e.g., 4 completions per prompt
            batch_size=self.args.generation_batch_size // self.num_generations,   # correction
            repeat_count=self.num_iterations * self.args.steps_per_generation,    # correction
            shuffle=True,
            seed=self.args.seed
        )
    
    def get_train_dataloader(self):
        """Override to ensure our custom sampler is used."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        
        sampler = self._get_train_sampler(train_dataset)
        dataloader_batch_size = self._train_batch_size * self.args.steps_per_generation
        
        return DataLoader(
            train_dataset,
            batch_size= self.args.generation_batch_size,  # < this is the change, HF was useing dataloader_batch_size
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_interpreter_token_ids(self) -> list[int]:
        """Get token IDs for <interpreter> and </interpreter> tags."""
        start_token = self.processing_class.encode("<interpreter>", add_special_tokens=False)[0]
        end_token = self.processing_class.encode("</interpreter>", add_special_tokens=False)[0]
        return [start_token, end_token]
    
    def _get_code_token_ids(self) -> list[int]:
        """Get token IDs for <code> and </code> tags."""
        start_token = self.processing_class.encode("<code>", add_special_tokens=False)[0]
        end_token = self.processing_class.encode("</code>", add_special_tokens=False)[0]
        return [start_token, end_token]
    
    def _binary_reward_function(self, prompts, completions, **kwargs) -> list[float]:
        """Default binary reward function for mathematical correctness."""
        rewards = []
        ground_truths = kwargs.get('ground_truths', [None] * len(completions))
        
        for completion, ground_truth in zip(completions, ground_truths):
            if self._is_correct_answer(completion, ground_truth):
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        return rewards
    
    def _execute_code(self, code_block: str) -> str:
        """
        Execute code in a sandbox environment.
        
        TODO: Implement actual code execution sandbox.
        For now, returns a placeholder.
        """
        # Placeholder implementation
        return f"Executed: {code_block[:50]}... -> Result: 42"
    

    def _check_equivalence(self, predicted, ground_truth):
        """Simple equivalence check - you can make this more sophisticated later."""
        # Simple string comparison for now
        return str(predicted).strip() == str(ground_truth).strip()

    def _is_correct_answer(self, completion_text, ground_truth):
        import re
        # Look for boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', completion_text)
        if match:
            predicted = match.group(1)
            return self._check_equivalence(predicted, ground_truth)
        return False

    def _compute_rewards(self, inputs, prompts, completions, completion_ids_list=None):
        """Calculate rewards for completions and combine them according to weights."""
        device = self.device  # Your device might be set differently
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Extract additional arguments from inputs if needed
        reward_kwargs = {}
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
            keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        
        # Add correct_answers to kwargs if present (common in math reasoning tasks)
        if "correct_answers" in reward_kwargs:
            reward_kwargs["solution"] = reward_kwargs["correct_answers"]  # Alias for compatibility

        # Calculate rewards for each function with non-zero weight
        for i, (reward_func, func_name) in enumerate(zip(self.reward_funcs, self.reward_func_names)):
            # Skip computation if weight is zero
            if abs(self.reward_weights[i].item()) < 1e-6:
                rewards_per_func[:, i] = float('nan')
                if self.verbose:
                    print(f"Skipping reward '{func_name}' (zero weight)")
                continue
            
            # Calculate reward
            try:
                # Call the reward function with appropriate arguments
                rewards = reward_func(
                    prompts=prompts, 
                    completions=completions,
                    completion_ids=completion_ids_list if completion_ids_list is not None else None,
                    **reward_kwargs
                )
                
                # Convert None values to NaN and ensure it's a tensor
                rewards = [r if r is not None else float('nan') for r in rewards]
                rewards_per_func[:, i] = torch.tensor(rewards, dtype=torch.float32, device=device)
                
                # Log reward statistics if verbose
                if self.verbose:
                    valid_rewards = [r for r in rewards if not (r is None or (isinstance(r, float) and math.isnan(r)))]
                    if valid_rewards:
                        print(f"Reward '{func_name}': min={min(valid_rewards):.4f}, max={max(valid_rewards):.4f}, "
                            f"mean={sum(valid_rewards)/len(valid_rewards):.4f}")
            except Exception as e:
                print(f"Error in reward function '{func_name}': {e}")
                rewards_per_func[:, i] = float('nan')
        
        # Combine rewards using weights
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        # Convert to list for easier handling
        final_rewards = rewards.cpu().tolist()
        
        return final_rewards
    

    def compute_rewards_and_advantages(self, inputs, prompts, completions, completion_ids_list=None):
        """Calculate rewards and compute advantages based on those rewards."""
        # First calculate rewards
        rewards = self.compute_rewards(inputs, prompts, completions, completion_ids_list)
        
        # Convert to tensor if not already
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # For now, simple advantage calculation
        advantages = rewards.clone()  # Simple case: advantages = rewards
        
        # If later I want to implement GRPO-style advantage calculation:
        if self.use_grouped_advantages:
            # Reshape rewards into groups (assuming self.num_generations is set)
            grouped_rewards = rewards.view(-1, self.num_generations)
            
            # Calculate statistics per group
            mean_grouped_rewards = grouped_rewards.mean(dim=1)
            std_grouped_rewards = grouped_rewards.std(dim=1)
            
            # Expand means and stds to match original shape
            mean_expanded = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_expanded = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            
            # Compute advantages: reward - baseline
            advantages = rewards - mean_expanded
            
            # Optionally normalize advantages
            if self.normalize_advantages:
                # Avoid division by zero
                std_expanded = torch.clamp(std_expanded, min=1e-8)
                advantages = advantages / std_expanded
        
        return advantages
    

    def _custom_generate(self, input_ids, attention_mask=None, past_key_values=None, max_new_tokens=50, eos_token_ids=None):
        """Custom generation function that avoids KV cache issues"""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        if eos_token_ids is None:
            eos_token_ids = [self.processing_class.eos_token_id]
        
        # Initialize
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        current_kv = past_key_values
        
        # Generate tokens in batches for efficiency
        all_tokens = []
        batch_size = 10  # Process this many tokens at once
        
        for start_idx in range(0, max_new_tokens, batch_size):
            # How many tokens to generate in this batch
            batch_tokens = min(batch_size, max_new_tokens - start_idx)
            
            # Accumulate new tokens
            new_tokens = []
            
            for _ in range(batch_tokens):
                # Forward pass with proper cache handling
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=current_ids if current_kv is None else current_ids[:, -1:],
                        attention_mask=current_mask if current_kv is None else current_mask[:, -1:],
                        past_key_values=DynamicCache.from_legacy_cache(current_kv) if current_kv is not None else None,
                        use_cache=True
                    )
                
                # Sample next token
                next_token_logits = outputs.logits[:, -1, :] / self.temperature
                filtered_logits = self._filter_logits(next_token_logits)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to accumulated tokens
                token_id = next_token.item()
                new_tokens.append(token_id)
                
                # Update for next iteration
                current_ids = torch.cat([current_ids, next_token], dim=1)
                token_mask = torch.ones((1, 1), device=current_mask.device, dtype=current_mask.dtype)
                current_mask = torch.cat([current_mask, token_mask], dim=1)
                current_kv = outputs.past_key_values
                
                # Check for stop tokens - include both EOS and code_end
                if token_id in eos_token_ids:
                    break
            
            # Add batch tokens to overall result
            all_tokens.extend(new_tokens)
            
            # Check if we hit a stop token
            if len(new_tokens) < batch_tokens:
                break
        
        # Convert to tensor
        result = torch.tensor([all_tokens], device=input_ids.device)
        return result, current_kv

    def _filter_logits(self, logits):
        """Apply top-k and top-p filtering"""
        if self.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
            logits[0, :] = torch.full_like(logits[0, :], float('-inf'))
            logits[0, top_k_indices[0]] = top_k_logits[0]
            
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        return logits

    def _retool_generate_with_interpreter(self, prompt_ids_batch, attention_mask_batch, eos_id, interpreter_id, code_id, max_turns=10):
        """Implementation with custom generation to avoid KV cache issues"""
        batch_size = prompt_ids_batch.size(0)
        batch_completion = []
        batch_interpreter_positions = []
        
        for i in range(batch_size):
            # Initialize
            current_input_id = prompt_ids_batch[i:i+1]
            current_attention_mask = attention_mask_batch[i:i+1]
            current_kv = None
            
            # Track completion (excludes prompt)
            cumulative_completion_ids = torch.empty((1, 0), dtype=torch.long, device=prompt_ids_batch.device)
            interpreter_positions = []
            
            for turn_idx in range(max_turns):
                # Check if input is empty
                if current_input_id.size(1) == 0:
                    break
                
                # Generate with custom function
                newly_generated_tokens, current_kv = self._custom_generate(
                    input_ids=current_input_id,
                    attention_mask=current_attention_mask,
                    past_key_values=current_kv,
                    max_new_tokens=self.max_completion_length,  # Use class attribute
                    eos_token_ids=[eos_id, code_id[1]]
                )
                
                # Add to completion
                cumulative_completion_ids = torch.cat([cumulative_completion_ids, newly_generated_tokens], dim=1)
                
                # Check last token
                last_token_id = newly_generated_tokens[0, -1].item() if newly_generated_tokens.size(1) > 0 else None
                
                # Check for end conditions
                if last_token_id == eos_id or turn_idx == max_turns - 1:
                    batch_completion.append(cumulative_completion_ids.squeeze(0))
                    batch_interpreter_positions.append(interpreter_positions)
                    break
                
                # Check for code end token
                if last_token_id == code_id[1]:
                    # Extract code from the full text
                    full_text = self.processing_class.decode(
                        torch.cat([prompt_ids_batch[i], cumulative_completion_ids[0]], dim=0)
                    )
                    code_match = re.search(r'<code>(.*?)</code>', full_text, re.DOTALL)
                    
                    if code_match:
                        code_block = code_match.group(1).strip()
                        interpreter_text = self._execute_code(code_block)
                        
                        # Format and add interpreter output
                        formatted_feedback = f"{self.processing_class.decode(interpreter_id[0])}{interpreter_text}{self.processing_class.decode(interpreter_id[1])}"
                        interpreter_ids = self.processing_class(
                            formatted_feedback,
                            return_tensors="pt",
                            add_special_tokens=False
                        ).input_ids.to(prompt_ids_batch.device)
                        
                        # Record positions
                        interpreter_start_idx = cumulative_completion_ids.size(1)
                        cumulative_completion_ids = torch.cat([cumulative_completion_ids, interpreter_ids], dim=1)
                        interpreter_end_idx = cumulative_completion_ids.size(1) - 1
                        interpreter_positions.append((interpreter_start_idx, interpreter_end_idx))
                        
                        # Set up for next turn
                        current_input_id = interpreter_ids
                        current_attention_mask = torch.ones_like(current_input_id)
                        # Keep current_kv from previous generation
                    else:
                        # No code block found despite </code> token
                        break
                else:
                    # Continue with the newly generated tokens
                    current_input_id = newly_generated_tokens
                    current_attention_mask = torch.ones_like(current_input_id)
            else:
                # Loop finished due to max_turns without a break
                batch_completion.append(cumulative_completion_ids.squeeze(0))
                batch_interpreter_positions.append(interpreter_positions)
        
        # Pad sequences
        if len(batch_completion) > 0:
            # Ensure padding_value is a valid integer
            padding_value = self.processing_class.pad_token_id
            if padding_value is None:
                padding_value = 0  # Use 0 as a default if pad_token_id is None
                
            padded_sequences = torch.nn.utils.rnn.pad_sequence(
                batch_completion, 
                batch_first=True, 
                padding_value=padding_value
            )
        else:
            padded_sequences = torch.empty((0, 0), dtype=torch.long, device=prompt_ids_batch.device)
        
        return padded_sequences, batch_interpreter_positions


    def _create_interpreter_mask(
        self, 
        completion_ids: torch.Tensor, 
        interpreter_positions: list[list[tuple[int, int]]]
    ) -> torch.Tensor:
        """
        Create interpreter mask from positions.
        
        Args:
            completion_ids: Tensor of shape (batch_size, seq_length)
            interpreter_positions: List[List[Tuple[start_idx, end_idx]]]
                                - Indices are relative to completion_ids
                                - start_idx: inclusive, end_idx: INCLUSIVE (unlike typical Python slicing)
        
        Returns:
            interpreter_mask: Tensor of shape (batch_size, seq_length)
                            1 = model-generated token, 0 = interpreter token
        """
        batch_size, seq_length = completion_ids.shape
        
        # Initialize mask with all 1s (assume all tokens are model-generated)
        interpreter_mask = torch.ones(batch_size, seq_length, dtype=torch.float, device=completion_ids.device)
        
        # For each sequence in the batch
        for batch_idx, positions_in_sequence in enumerate(interpreter_positions):
            # For each interpreter section in this sequence
            for start_idx, end_idx in positions_in_sequence:
                # Clamp indices to valid range
                start_idx = max(0, min(start_idx, seq_length - 1))
                end_idx = max(0, min(end_idx, seq_length - 1))
                
                # Zero out interpreter tokens (BOTH start and end inclusive)
                if start_idx <= end_idx:  # Changed from < to <=
                    interpreter_mask[batch_idx, start_idx:end_idx + 1] = 0  # Changed to end_idx + 1
        
        return interpreter_mask
    

def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
    
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]


        # use custom multi-turn-w-tool-use Generate completions
        completion_ids, interpreter_positions = self._retool_generate_with_interpreter(
            prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config,
            eos_id = self.eos_id, interpreter_id = self.interpreter_id, code_id = self.code_id 
        )
    

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        # compute interpreter mask
        interpreter_mask = self._create_interpreter_mask(completion_ids, interpreter_positions)
    

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)


        # no need to return old_per_token_logps

        # Extract ground truths from inputs  
        ground_truths = [x.get("answer") for x in inputs]  # Adjust key name as needed
        
        # Decode completions for reward computation
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Compute rewards and advantages
        advantages = self._compute_rewards_and_advantages(
            completions_text, 
            ground_truths, 
            device=device
        )


        # Log the metrics 
        if mode == "train":
            self.state.num_input_tokens_seen += attention_mask.sum().item()  # Skip gather
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        
        # Log completion lengths
        completion_lengths = completion_mask.sum(1)  # Skip gather
        self._metrics[mode]["completions/mean_length"].append(completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(completion_lengths.float().max().item())
        
        # Log terminated sequences
        terminated_with_eos = is_eos.any(dim=1)  # Skip gather
        term_completion_lengths = completion_lengths[terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        
        # Log rewards (simplified for single reward function)
        advantages_tensor = advantages 
        self._metrics[mode]["rewards/binary_correctness/mean"].append(advantages_tensor.mean().item())
        self._metrics[mode]["rewards/binary_correctness/std"].append(advantages_tensor.std().item())

        
        # Log texts for debugging
        self._textual_logs["prompt"].extend(prompts_text)
        self._textual_logs["completion"].extend(completions_text)
        self._textual_logs["rewards"]["binary_correctness"].extend(advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "interpreter_mask": interpreter_mask,
            "advantages": advantages
        }


# Get the per-token log probabilities for the completions for the model and the reference model
@profiling_decorator
def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
    batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
    all_logps = []
    for i in range(0, input_ids.size(0), batch_size):
        input_ids_batch = input_ids[i : i + batch_size]
        attention_mask_batch = attention_mask[i : i + batch_size]

        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
        ).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids_batch = input_ids_batch[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
        all_logps.append(logps)
    return torch.cat(all_logps, dim=0)


@staticmethod
def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask  = inputs["completion_ids"], inputs["completion_mask"]
        
        # Added for ReTool Trainer
        interpreter_mask = inputs["interpreter_mask"]
        final_mask = interpreter_mask * completion_mask
    
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        with torch.no_grad():
            ref_per_token_logps = self._get_per_token_logps(
                self.ref_model, input_ids, attention_mask, logits_to_keep
            )
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]

        old_per_token_logps = ref_per_token_logps
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

    
        # For PPO loss
        masked_loss = per_token_loss * final_mask
        total_valid_tokens = final_mask.sum() + 1e-8  # Avoid division by zero
        loss = masked_loss.sum() / total_valid_tokens

        """ --- """
    
        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * final_mask).sum() / final_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * final_mask).sum() / final_mask.sum()
        high_clip = (is_high_clipped * final_mask).sum() / final_mask.sum()
        clip_ratio = (is_region_clipped * final_mask).sum() / final_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

def train(self):
    """
    Comprehensive training loop for ReTool with GRPO.
    Adapted from train_with_batching to work as a method.
    """
    # Initialize
    self.model.train()
    if not hasattr(self, 'ref_model') or self.ref_model is None:
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
    
    # Setup tracking
    writer = SummaryWriter(self.args.logging_dir)
    training_history = []
    
    # Get dataloader with our custom sampler
    train_dataloader = self.get_train_dataloader()
    
    # Generation storage for reuse
    stored_generation_outputs = None
    generation_counter = 0
    global_step = 0
    
    for epoch in range(self.args.num_train_epochs):
        epoch_metrics = []
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # batch already has repeated prompts from our RepeatSampler
            # Shape: (generation_batch_size, ...) where generation_batch_size = unique_prompts * num_generations
            
            # Determine if we need new generations
            generate_new = (global_step % (self.args.steps_per_generation * self.num_iterations)) == 0
            
            if generate_new:
                print(f"Generating new completions at step {global_step}")
                with torch.no_grad():
                    # This is where ReTool magic happens - generate with code execution!
                    stored_generation_outputs = self._generate_and_score_completions(batch)
                generation_counter = 0
            
            # Now train on the stored generations
            # This replaces the mini/micro batch logic from your original function
            batch_loss = self._train_on_stored_generations(
                stored_generation_outputs,
                epoch_metrics
            )
            
            global_step += 1
            generation_counter += 1
            
            # Logging
            if global_step % self.args.logging_steps == 0:
                self._log_training_metrics(writer, epoch_metrics, global_step)
            
            # Optional: Check for training instability
            if self._should_stop_training(epoch_metrics):
                print("Training instability detected! Stopping early.")
                return training_history
        
        # End of epoch
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        epoch_summary = self._compute_epoch_summary(epoch_metrics, start_mem, end_mem)
        training_history.append(epoch_summary)
        
        # Log epoch results
        self._log_epoch_metrics(epoch, epoch_summary, writer)
        
        # Update scheduler if we have one
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step(epoch_summary['mean_reward'])
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
    
    writer.close()
    return training_history

def _train_on_stored_generations(self, generation_outputs, epoch_metrics):
    """
    Train on stored generations with mini/micro-batching.
    This replaces the inner loops of your train_with_batching.
    """
    # Extract components from generation_outputs
    # These already include code execution results and advantages!
    prompt_ids = generation_outputs['prompt_ids']
    completion_ids = generation_outputs['completion_ids']
    advantages = generation_outputs['advantages']
    completion_mask = generation_outputs['completion_mask']
    interpreter_mask = generation_outputs.get('interpreter_mask', completion_mask)
    
    batch_size = prompt_ids.size(0)
    
    # Mini-batch size: process multiple groups together
    # Each group has num_generations completions
    mini_batch_size = self.args.per_device_train_batch_size * self.num_generations
    
    # Micro-batch size: for memory efficiency within mini-batch
    micro_batch_size = max(self.num_generations, 4)  # At least one full group
    
    total_loss = 0
    num_updates = 0
    
    # Shuffle indices for this training iteration
    indices = torch.randperm(batch_size)
    
    # Process in mini-batches
    for mini_start in range(0, batch_size, mini_batch_size):
        mini_end = min(mini_start + mini_batch_size, batch_size)
        mini_indices = indices[mini_start:mini_end]
        
        self.optimizer.zero_grad()
        mini_batch_loss = 0
        num_micro_batches = 0
        
        # Process in micro-batches (gradient accumulation)
        for micro_start in range(0, len(mini_indices), micro_batch_size):
            micro_end = min(micro_start + micro_batch_size, len(mini_indices))
            micro_indices = mini_indices[micro_start:micro_end]
            
            # Create micro-batch
            micro_batch = {
                'prompt_ids': prompt_ids[micro_indices],
                'prompt_mask': generation_outputs['prompt_mask'][micro_indices],
                'completion_ids': completion_ids[micro_indices],
                'completion_mask': completion_mask[micro_indices],
                'interpreter_mask': interpreter_mask[micro_indices],
                'advantages': advantages[micro_indices]
            }
            
            # Compute GRPO loss (this uses your _compute_loss method)
            loss = self._compute_loss(self.model, micro_batch)
            
            # Scale for gradient accumulation
            scaled_loss = loss * (len(micro_indices) / len(mini_indices))
            scaled_loss.backward()
            
            mini_batch_loss += loss.item()
            num_micro_batches += 1
        
        # Gradient clipping and optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Track metrics
        batch_metrics = {
            'loss': mini_batch_loss / num_micro_batches,
            'gradient_norm': grad_norm.item(),
            'batch_size': len(mini_indices),
            'advantages_mean': advantages[mini_indices].mean().item(),
            'advantages_std': advantages[mini_indices].std().item()
        }
        epoch_metrics.append(batch_metrics)
        
        total_loss += mini_batch_loss
        num_updates += 1
    
    return total_loss / max(num_updates, 1)