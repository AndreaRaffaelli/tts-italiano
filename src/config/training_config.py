"""
Training Configuration Management
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from transformers import Seq2SeqTrainingArguments
import os


@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration with Italian TTS-specific settings"""
    
    # Basic training parameters
    output_dir: str = "speecht5_italian_enhanced"
    max_steps: int = 10000
    learning_rate: float = 3e-6
    weight_decay: float = 0.01
    warmup_steps: int = 1500
    
    # Batch size settings (will be auto-detected if None)
    per_device_train_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    
    # Scheduler settings
    lr_scheduler_type: str = "cosine_with_restarts"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization settings
    fp16: Optional[bool] = None  # Will be set based on GPU availability
    dataloader_pin_memory: bool = True
    gradient_checkpointing: bool = True
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Logging
    logging_steps: int = 25
    logging_dir: Optional[str] = None
    report_to: list = field(default_factory=lambda: ["tensorboard"])
    
    # Data settings
    dataloader_num_workers: int = 0  # Keep 0 for stability with audio data
    remove_unused_columns: bool = False
    label_names: list = field(default_factory=lambda: ["labels"])
    
    # Hub settings
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "end"
    
    # Italian TTS specific
    max_text_length: int = 300
    max_audio_length: float = 10.0  # seconds
    min_audio_length: float = 1.0   # seconds
    
    # Advanced settings
    prediction_loss_only: bool = False
    include_inputs_for_metrics: bool = False
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Set logging directory if not specified
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        
        # Set hub model id if pushing to hub
        if self.push_to_hub and self.hub_model_id is None:
            self.hub_model_id = f"your-username/{os.path.basename(self.output_dir)}"
    
    def to_training_arguments(self, 
                            has_gpu: bool = True,
                            batch_size: Optional[int] = None,
                            grad_acc_steps: Optional[int] = None) -> Seq2SeqTrainingArguments:
        """
        Convert to Hugging Face TrainingArguments
        
        Args:
            has_gpu: Whether GPU is available
            batch_size: Override batch size
            grad_acc_steps: Override gradient accumulation steps
        
        Returns:
            Seq2SeqTrainingArguments instance
        """
        # Set batch size and gradient accumulation
        if batch_size is not None:
            per_device_batch_size = batch_size
        else:
            per_device_batch_size = self.per_device_train_batch_size or (4 if has_gpu else 1)
        
        if grad_acc_steps is not None:
            gradient_accumulation = grad_acc_steps
        else:
            gradient_accumulation = self.gradient_accumulation_steps or (8 if has_gpu else 32)
        
        # Set FP16 based on GPU availability
        use_fp16 = self.fp16 if self.fp16 is not None else has_gpu
        
        return Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            
            # Scheduler
            lr_scheduler_type=self.lr_scheduler_type,
            
            # Optimization
            fp16=use_fp16,
            dataloader_pin_memory=self.dataloader_pin_memory,
            gradient_checkpointing=self.gradient_checkpointing,
            
            # Evaluation and saving
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            
            # Logging
            logging_steps=self.logging_steps,
            logging_dir=self.logging_dir,
            report_to=self.report_to,
            
            # Data
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=self.remove_unused_columns,
            label_names=self.label_names,
            
            # Hub
            push_to_hub=self.push_to_hub,
            hub_model_id=self.hub_model_id,
            hub_strategy=self.hub_strategy,
            
            # Advanced
            prediction_loss_only=self.prediction_loss_only,
            include_inputs_for_metrics=self.include_inputs_for_metrics,
        )
    
    def update_from_args(self, args):
        """Update config from command line arguments"""
        if hasattr(args, 'max_steps') and args.max_steps:
            self.max_steps = args.max_steps
        
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.learning_rate = args.learning_rate
        
        if hasattr(args, 'batch_size') and args.batch_size:
            self.per_device_train_batch_size = args.batch_size
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.output_dir = args.output_dir
        
        if hasattr(args, 'push_to_hub') and args.push_to_hub:
            self.push_to_hub = args.push_to_hub
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        
        # Convert to dict, handling non-serializable types
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Training config saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'EnhancedTrainingConfig':
        """Load configuration from JSON file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create instance and update with loaded values
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        print(f"📂 Training config loaded from {filepath}")
        return config


@dataclass  
class DataConfig:
    """Configuration for data processing"""
    
    # Cache settings
    cache_dir: str = "cache"
    force_reprocess: bool = False
    
    # Dataset settings
    target_sampling_rate: int = 16000
    max_dataset_size: Optional[int] = None  # Limit dataset size for testing
    
    # Preprocessing settings
    normalize_accents: bool = True
    expand_abbreviations: bool = True
    high_pass_filter: bool = True
    high_pass_freq: float = 80.0
    
    # Quality filtering
    min_text_length: int = 10
    max_text_length: int = 300
    min_audio_duration: float = 1.0
    max_audio_duration: float = 10.0
    max_special_char_ratio: float = 0.3
    
    # Speaker embedding settings
    speaker_model_source: str = "speechbrain/spkrec-xvect-voxceleb"
    speaker_model_cache_dir: str = "/tmp/speaker_model"
    
    # Train/validation split
    validation_split: float = 0.15
    split_seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for model setup"""
    
    # Model checkpoints
    tts_checkpoint: str = "microsoft/speecht5_tts"
    speaker_checkpoint: str = "speechbrain/spkrec-xvect-voxceleb"
    
    # Model cache
    cache_dir: Optional[str] = None
    
    # Device settings
    device: str = "auto"
    
    # Training optimizations
    use_gradient_checkpointing: bool = True
    use_cache: bool = False


def create_training_config(args) -> EnhancedTrainingConfig:
    """
    Create training configuration from command line arguments
    
    Args:
        args: Command line arguments
    
    Returns:
        EnhancedTrainingConfig instance
    """
    config = EnhancedTrainingConfig()
    config.update_from_args(args)
    return config


def create_data_config(args) -> DataConfig:
    """
    Create data configuration from command line arguments
    
    Args:
        args: Command line arguments
    
    Returns:
        DataConfig instance
    """
    config = DataConfig()
    
    if hasattr(args, 'cache_dir'):
        config.cache_dir = args.cache_dir
    
    if hasattr(args, 'force_reprocess'):
        config.force_reprocess = args.force_reprocess
    
    return config


def create_model_config(args) -> ModelConfig:
    """
    Create model configuration from command line arguments
    
    Args:
        args: Command line arguments
    
    Returns:
        ModelConfig instance
    """
    config = ModelConfig()
    
    if hasattr(args, 'cache_dir'):
        config.cache_dir = args.cache_dir
    
    return config


def print_config_summary(training_config: EnhancedTrainingConfig,
                        data_config: DataConfig,
                        model_config: ModelConfig):
    """Print summary of all configurations"""
    print("\n⚙️ Configuration Summary")
    print("=" * 50)
    
    print("🎯 Training Configuration:")
    print(f"   Output directory: {training_config.output_dir}")
    print(f"   Max steps: {training_config.max_steps:,}")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Batch size: {training_config.per_device_train_batch_size or 'Auto'}")
    print(f"   Gradient accumulation: {training_config.gradient_accumulation_steps or 'Auto'}")
    print(f"   Warmup steps: {training_config.warmup_steps:,}")
    print(f"   Scheduler: {training_config.lr_scheduler_type}")
    print(f"   FP16: {training_config.fp16 or 'Auto'}")
    
    print("\n📊 Data Configuration:")
    print(f"   Cache directory: {data_config.cache_dir}")
    print(f"   Force reprocess: {data_config.force_reprocess}")
    print(f"   Sampling rate: {data_config.target_sampling_rate:,} Hz")
    print(f"   Text length: {data_config.min_text_length}-{data_config.max_text_length} chars")
    print(f"   Audio duration: {data_config.min_audio_duration}-{data_config.max_audio_duration}s")
    print(f"   Validation split: {data_config.validation_split*100:.0f}%")
    
    print("\n🤖 Model Configuration:")
    print(f"   TTS checkpoint: {model_config.tts_checkpoint}")
    print(f"   Speaker checkpoint: {model_config.speaker_checkpoint}")
    print(f"   Device: {model_config.device}")
    print(f"   Gradient checkpointing: {model_config.use_gradient_checkpointing}")


def validate_config(config: EnhancedTrainingConfig) -> bool:
    """
    Validate training configuration
    
    Args:
        config: Training configuration to validate
    
    Returns:
        True if configuration is valid
    """
    errors = []
    
    # Check required directories
    if not config.output_dir:
        errors.append("Output directory is required")
    
    # Check numeric values
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.max_steps <= 0:
        errors.append("Max steps must be positive")
    
    if config.warmup_steps < 0:
        errors.append("Warmup steps cannot be negative")
    
    if config.warmup_steps >= config.max_steps:
        errors.append("Warmup steps should be less than max steps")
    
    # Check evaluation settings
    if config.eval_steps <= 0:
        errors.append("Eval steps must be positive")
    
    if config.save_steps <= 0:
        errors.append("Save steps must be positive")
    
    # Print errors if any
    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True
