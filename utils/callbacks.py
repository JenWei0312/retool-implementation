import subprocess
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

class CurriculumLearningCallback(TrainerCallback):
    def __init__(self):
        self.current_stage = "format_stage"
        self.stages = {
            "format_stage": {
                "reward_weights": {"format": 1.0, "accuracy": 0.0, "code_execution": 0.0, 
                                  "length": 0.0, "code_ratio": 0.0, "code_timing": 0.0},
                "beta": 0.1, # Higher KL - stay close to base model format
                "steps": 1000
            },
            "code_execution_stage": {
                "reward_weights": {"format": 0.3, "accuracy": 0.0, "code_execution": 0.7, 
                                  "length": 0.0, "code_ratio": 0.0, "code_timing": 0.0},
                "beta": 0.05, # Medium KL
                "steps": 2000
            },
            "accuracy_stage": {
                "reward_weights": {"format": 0.2, "accuracy": 0.8, "code_execution": 0.0, 
                                  "length": 0.0, "code_ratio": 0.0, "code_timing": 0.0},
                "beta": 0.01, # Very low KL - allow exploration
                "steps": 3000
            },
            "refinement_stage": {
                "reward_weights": {"format": 0.1, "accuracy": 0.6, "code_execution": 0.1, 
                                  "length": 0.1, "code_ratio": 0.05, "code_timing": 0.05},
                "beta": 0.03, # Medium-low KL - stabilize learning
                "steps": 5000
            }
        }

        self.total_steps = sum(stage_config["steps"] for stage_config in self.stages.values())
        self.stage_transitions = self._calculate_stage_transitions()
    
    def _calculate_stage_transitions(self):
        """Calculate at which step each stage transition occurs."""
        transitions = {}
        current_step = 0
        for stage, config in self.stages.items():
            current_step += config["steps"]
            transitions[stage] = current_step
        return transitions
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update reward weights based on current training stage."""
        trainer = kwargs.get('trainer')
        if trainer is None:
            return
        
        # Check if it's time to transition to the next stage
        current_step = state.global_step
        
        # Determine current stage
        previous_stage = self.current_stage
        for stage, transition_step in self.stage_transitions.items():
            if current_step <= transition_step:
                self.current_stage = stage
                break
        
        # If stage changed, update weights and log the transition
        if previous_stage != self.current_stage:
            print(f"Transitioning from {previous_stage} to {self.current_stage} at step {current_step}")
        
        # Apply weights for current stage
        stage_weights = self.stages[self.current_stage]["reward_weights"]
        
        # Update trainer's reward weights
        # This assumes the trainer has a reward_weights attribute
        for i, func_name in enumerate(trainer.reward_func_names):
            if func_name in stage_weights:
                trainer.reward_weights[i] = stage_weights[func_name]

                

class CurriculumLearningCallback(TrainerCallback):
    """A callback to implement curriculum learning stages during training."""
    def __init__(self, debug=False):
        self.debug = debug
        self.current_stage = "format_stage"
        self.stages = {
                    "format_stage": {
                        "reward_weights": {"format": 1.0, "accuracy": 0.0, "code_execution": 0.0, 
                                        "length": 0.0, "code_ratio": 0.0, "code_timing": 0.0},
                        "beta": 0.1, # Higher KL - stay close to base model format
                        "steps": 1000
                    },
                    "code_execution_stage": {
                        "reward_weights": {"format": 0.3, "accuracy": 0.0, "code_execution": 0.7, 
                                        "length": 0.0, "code_ratio": 0.0, "code_timing": 0.0},
                        "beta": 0.05, # Medium KL
                        "steps": 2000
                    },
                    "accuracy_stage": {
                        "reward_weights": {"format": 0.2, "accuracy": 0.8, "code_execution": 0.0, 
                                        "length": 0.0, "code_ratio": 0.0, "code_timing": 0.0},
                        "beta": 0.01, # Very low KL - allow exploration
                        "steps": 3000
                    },
                    "refinement_stage": {
                        "reward_weights": {"format": 0.1, "accuracy": 0.6, "code_execution": 0.1, 
                                        "length": 0.1, "code_ratio": 0.05, "code_timing": 0.05},
                        "beta": 0.03, # Medium-low KL - stabilize learning
                        "steps": 5000
                    }
                }
        self.total_steps = sum(stage_config["steps"] for stage_config in self.stages.values())
        self.stage_transitions = self._calculate_stage_transitions()
        
        print(f"Curriculum learning initialized with {len(self.stages)} stages:")
        for stage, end_step in self.stage_transitions.items():
            print(f"  {stage}: ends at step {end_step}")
    
    def _calculate_stage_transitions(self):
        """Calculate at which step each stage transition occurs."""
        transitions = {}
        current_step = 0
        for stage, config in self.stages.items():
            current_step += config["steps"]
            transitions[stage] = current_step
        return transitions
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize reward weights and beta at the start of training."""
        trainer = kwargs.get('trainer')
        if trainer is None:
            return
        
        # Set initial weights and beta from first stage
        first_stage = list(self.stages.keys())[0]
        stage_config = self.stages[first_stage]
        
        # Update reward weights
        if hasattr(trainer, "reward_weights") and hasattr(trainer, "reward_func_names"):
            for i, func_name in enumerate(trainer.reward_func_names):
                if func_name in stage_config["reward_weights"]:
                    trainer.reward_weights[i] = stage_config["reward_weights"][func_name]
                    if self.debug:
                        print(f"Setting initial weight for {func_name}: {trainer.reward_weights[i]}")
        else:
            print("Warning: Trainer doesn't have reward_weights or reward_func_names attributes")
        
        # Update beta (KL coefficient)
        if hasattr(trainer, "beta"):
            trainer.beta = stage_config.get("beta", 0.1)
            if self.debug:
                print(f"Setting initial beta: {trainer.beta}")
        else:
            print("Warning: Trainer doesn't have a beta attribute")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update reward weights and beta based on current training stage."""
        trainer = kwargs.get('trainer')
        if trainer is None:
            return
        
        # Check if it's time to transition to the next stage
        current_step = state.global_step
        
        # Determine current stage
        previous_stage = self.current_stage
        for stage, transition_step in sorted(self.stage_transitions.items()):
            if current_step <= transition_step:
                self.current_stage = stage
                break
        
        # If stage changed, update weights and log the transition
        if previous_stage != self.current_stage:
            print(f"Transitioning from {previous_stage} to {self.current_stage} at step {current_step}")
            
        # Get config for current stage
        stage_config = self.stages[self.current_stage]
        
        # Update reward weights
        if hasattr(trainer, "reward_weights") and hasattr(trainer, "reward_func_names"):
            for i, func_name in enumerate(trainer.reward_func_names):
                if func_name in stage_config["reward_weights"]:
                    new_weight = stage_config["reward_weights"][func_name]
                    if trainer.reward_weights[i] != new_weight:
                        trainer.reward_weights[i] = new_weight
                        if self.debug:
                            print(f"Updated weight for {func_name}: {new_weight}")
        
        # Update beta (KL coefficient)
        if hasattr(trainer, "beta"):
            new_beta = stage_config.get("beta", 0.1)
            if trainer.beta != new_beta:
                trainer.beta = new_beta
                if self.debug:
                    print(f"Updated beta: {new_beta}")