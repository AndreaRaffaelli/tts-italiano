# config/__init__.py
"""
Configuration management for Enhanced TTS Training
"""

from .training_config import EnhancedTrainingConfig, DataConfig, ModelConfig

__all__ = ['EnhancedTrainingConfig', 'DataConfig', 'ModelConfig']


# data/__init__.py
"""
Data processing and management for Enhanced TTS Training
"""

from .preprocessing import (
    enhanced_audio_preprocessing, 
    improved_text_preprocessing,
    TextPreprocessor,
    AudioPreprocessor
)
from .dataset_loader import DatasetManager, load_speaker_model
from .data_collator import TTSDataCollatorWithPadding, create_data_collator

__all__ = [
    'enhanced_audio_preprocessing', 
    'improved_text_preprocessing',
    'TextPreprocessor',
    'AudioPreprocessor',
    'DatasetManager', 
    'load_speaker_model',
    'TTSDataCollatorWithPadding', 
    'create_data_collator'
]


# models/__init__.py
"""
Model setup and management for Enhanced TTS Training
"""

from .model_setup import (
    ModelManager,
    setup_models,
    validate_model_compatibility,
    print_model_summary,
    ModelConfig,
    ModelCheckpoint
)

__all__ = [
    'ModelManager',
    'setup_models',
    'validate_model_compatibility',
    'print_model_summary',
    'ModelConfig',
    'ModelCheckpoint'
]


# training/__init__.py
"""
Training logic for Enhanced TTS Training
"""

from .trainer import EnhancedTTSTrainer, TrainingMonitor, create_trainer

__all__ = ['EnhancedTTSTrainer', 'TrainingMonitor', 'create_trainer']


# utils/__init__.py
"""
Utility functions for Enhanced TTS Training
"""

from .cli_utils import print_step, ask_user_confirmation, print_dependencies_info
from .cache_utils import save_cache, load_cache, get_cache_paths
from .gpu_utils import check_gpu, get_optimal_batch_size, print_device_info

__all__ = [
    'print_step', 
    'ask_user_confirmation', 
    'print_dependencies_info',
    'save_cache', 
    'load_cache', 
    'get_cache_paths',
    'check_gpu', 
    'get_optimal_batch_size', 
    'print_device_info'
]
