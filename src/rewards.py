import re
from typing import List, Dict, Optional, Callable
import numpy as np

def format_reward(completions, **kwargs):
    """Reward function that checks if the code is enclosed within <code> and </code> tags,
    and the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r".*?<code>.*?</code>.*?<answer>.*?</answer>.*?"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content, re.DOTALL | re.MULTILINE) is not None for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions: list[list[dict[str, str]]], correct_answers: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion's answer matches the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, correct_answer in zip(contents, correct_answers):
        # Extract answer from the completion using regex
        answer_match = re.search(r'<answer>\\boxed{(.*?)}</answer>', content, re.DOTALL)
        if not answer_match:
            rewards.append(0.0)
            continue
            
        extracted_answer = answer_match.group(1).strip()
        
        # Check if the extracted answer matches the correct answer
        # You might need a more sophisticated comparison for mathematical expressions
        if extracted_answer == correct_answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

def code_execution_reward(completions, **kwargs):
    """Reward function that checks if the code execution was successful."""
    completion_contents = [completion[0]["content"] for completion in completions]
    # Check for error patterns in interpreter output
    error_patterns = [
        r'<interpreter>.*?Error.*?</interpreter>',
        r'<interpreter>.*?Exception.*?</interpreter>',
        r'<interpreter>.*?Traceback.*?</interpreter>'
    ]
    
    rewards = []
    for content in completion_contents:
        # Find all code-interpreter pairs
        code_blocks = re.findall(r'<code>.*?</code>\s*<interpreter>(.*?)</interpreter>', content, re.DOTALL)
        if not code_blocks:
            rewards.append(0.0)
            continue
            
        # Check each interpreter output for errors
        error_count = 0
        for interpreter_output in code_blocks:
            has_error = any(re.search(pattern, interpreter_output, re.DOTALL) for pattern in error_patterns)
            if has_error:
                error_count += 1
                
        # Calculate success rate
        if len(code_blocks) == 0:
            rewards.append(0.0)
        else:
            success_rate = 1.0 - (error_count / len(code_blocks))
            rewards.append(success_rate)
            
    return rewards

def len_reward(completions, **kwargs):
    """Reward shorter completions to encourage efficiency."""
    completion_contents = [completion[0]["content"] for completion in completions]
    lengths = [len(content) for content in completion_contents]
    
    # If all completions have the same length, return neutral rewards
    if min(lengths) == max(lengths):
        return [0.0] * len(completions)
    
    # Normalize lengths to [0, 1] range and invert (shorter = higher reward)
    normalized_lengths = [(length - min(lengths)) / (max(lengths) - min(lengths)) for length in lengths]
    rewards = [1.0 - norm_length for norm_length in normalized_lengths]
    
    # Scale to a smaller range to make this a secondary consideration
    scaled_rewards = [0.2 * reward for reward in rewards]
    
    return scaled_rewards

def code_ratio_reward(completions, **kwargs):
    """Reward appropriate code-to-text ratio."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        # Extract all code blocks
        code_blocks = re.findall(r'<code>(.*?)</code>', content, re.DOTALL)
        total_code_length = sum(len(code) for code in code_blocks)
        total_length = len(content)
        
        if total_length == 0:
            rewards.append(0.0)
            continue
            
        code_ratio = total_code_length / total_length
        
        # Reward an optimal ratio range (e.g., 0.2 to 0.4)
        if 0.2 <= code_ratio <= 0.4:
            rewards.append(0.3)  # Full reward
        elif 0.1 <= code_ratio < 0.2 or 0.4 < code_ratio <= 0.5:
            rewards.append(0.2)  # Partial reward
        elif 0.05 <= code_ratio < 0.1 or 0.5 < code_ratio <= 0.6:
            rewards.append(0.1)  # Minimal reward
        else:
            rewards.append(0.0)  # No reward
            
    return rewards

def code_timing_reward(completions, **kwargs):
    """Reward for invoking code at appropriate points in the reasoning process."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        # Calculate relative position of first code block
        first_code_pos = content.find('<code>')
        if first_code_pos == -1:
            rewards.append(0.0)
            continue
            
        relative_pos = first_code_pos / len(content)
        
        # Reward early-to-middle code invocation (between 10% and 40% of the way through)
        if 0.1 <= relative_pos <= 0.4:
            rewards.append(0.3)
        elif 0.05 <= relative_pos < 0.1 or 0.4 < relative_pos <= 0.5:
            rewards.append(0.2)
        elif 0.0 <= relative_pos < 0.05 or 0.5 < relative_pos <= 0.7:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
            
    return rewards

def get_reward_funcs(script_args) -> list[Callable]:
    """Create a registry of available reward functions and return those specified in script_args."""
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "code_execution": code_execution_reward,
        "length": len_reward,
        "code_ratio": code_ratio_reward,
        "code_timing": code_timing_reward,
    }
    
    # Get the specified reward functions
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    return reward_funcs