"""
Enhanced TTS Training Logic
"""

import os
from pathlib import Path
from transformers import Seq2SeqTrainer
import torch

from utils.cli_utils import (
    print_step, ask_user_confirmation, print_dependencies_info,
    print_training_summary
)
from utils.cache_utils import get_cache_paths
from utils.gpu_utils import get_optimal_batch_size, print_memory_usage, cleanup_gpu_memory
from data.dataset_loader import DatasetManager, load_speaker_model
from data.data_collator import create_data_collator, validate_batch
from models.model_setup import setup_models, validate_model_compatibility, print_model_summary
from config.training_config import (
    create_training_config, create_data_config, create_model_config,
    print_config_summary, validate_config
)


class EnhancedTTSTrainer:
    """Enhanced TTS Trainer with modular design"""
    
    def __init__(self, args, has_gpu: bool):
        self.args = args
        self.has_gpu = has_gpu
        
        # Initialize configurations
        self.training_config = create_training_config(args)
        self.data_config = create_data_config(args)
        self.model_config = create_model_config(args)
        
        # Initialize components
        self.cache_paths = get_cache_paths(args.cache_dir)
        
        # Components that will be initialized during pipeline
        self.processor = None
        self.model = None
        self.speaker_model = None
        self.dataset_manager = None
        self.trainer = None
        
    def check_dependencies(self) -> bool:
        """Check if all dependencies are installed"""
        print_step(1, "Dependencies Check")
        print_dependencies_info()
        
        if not self.args.yes and not ask_user_confirmation("Are all dependencies installed?"):
            print("Please install dependencies first.")
            return False
        
        return True
    
    def setup_models(self) -> bool:
        """Setup all required models"""
        print_step(2, "Loading Models")
        
        try:
            # Load main models
            self.processor, self.model, self.speaker_model = setup_models(
                checkpoint=self.model_config.tts_checkpoint,
                device=self.model_config.device,
                cache_dir=self.model_config.cache_dir,
                optimize_for_training=True
            )
            
            # Validate compatibility
            if not validate_model_compatibility(self.processor, self.model):
                print("❌ Model compatibility check failed")
                return False
            
            # Print model summary
            print_model_summary(self.processor, self.model, self.speaker_model)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup models: {e}")
            return False
    
    def setup_dataset_manager(self) -> bool:
        """Setup dataset manager"""
        print_step(3, "Initializing Dataset Manager")
        
        try:
            self.dataset_manager = DatasetManager(
                cache_paths=self.cache_paths,
                processor=self.processor,
                speaker_model=self.speaker_model,
                force_reprocess=self.data_config.force_reprocess,
                target_sampling_rate=self.data_config.target_sampling_rate
            )
            
            print("✅ Dataset manager initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup dataset manager: {e}")
            return False
    
    def load_and_process_datasets(self):
        """Load and process all datasets"""
        print_step(4, "Loading and Processing Datasets")
        
        try:
            dataset_split = self.dataset_manager.load_and_process_datasets()
            return dataset_split
            
        except Exception as e:
            print(f"❌ Failed to load and process datasets: {e}")
            return None
    
    def setup_training_components(self, dataset_split) -> bool:
        """Setup training components (data collator, trainer, etc.)"""
        print_step(5, "Setting up Training Components")
        
        try:
            # Determine optimal batch size
            batch_size, grad_acc_steps = get_optimal_batch_size(
                self.has_gpu, self.args.batch_size
            )
            print(f"🔧 Using batch size: {batch_size}, gradient accumulation: {grad_acc_steps}")
            
            # Create data collator
            data_collator = create_data_collator(self.processor)
            
            # Test data collator with sample data
            if len(dataset_split["train"]) > 0:
                sample_features = dataset_split["train"].select(range(min(2, len(dataset_split["train"]))))
                is_valid, test_batch = data_collator.__call__(sample_features)
                if not is_valid:
                    print("❌ Data collator validation failed")
                    return False
                print("✅ Data collator test passed")
            
            # Create training arguments
            training_args = self.training_config.to_training_arguments(
                has_gpu=self.has_gpu,
                batch_size=batch_size,
                grad_acc_steps=grad_acc_steps
            )
            
            # Validate configuration
            if not validate_config(self.training_config):
                return False
            
            # Initialize trainer
            self.trainer = Seq2SeqTrainer(
                args=training_args,
                model=self.model,
                train_dataset=dataset_split["train"],
                eval_dataset=dataset_split["test"],
                data_collator=data_collator,
                processing_class=self.processor,
            )
            
            print("✅ Training components setup completed")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup training components: {e}")
            return False
    
    def run_training(self) -> bool:
        """Execute the training process"""
        print_step(6, "Starting Training")
        
        if not self.args.yes and not ask_user_confirmation("Start training now?"):
            print("Training cancelled by user.")
            return False
        
        try:
            print("🚀 Training started...")
            print_memory_usage()
            
            # Save configuration
            config_path = os.path.join(self.training_config.output_dir, "training_config.json")
            self.training_config.save_to_file(config_path)
            
            # Start training
            train_result = self.trainer.train()
            
            print("✅ Training completed successfully!")
            print_memory_usage()
            
            # Save the model
            print("💾 Saving model and processor...")
            self.trainer.save_model(self.training_config.output_dir)
            self.processor.save_pretrained(self.training_config.output_dir)
            print("✅ Model saved!")
            
            # Handle hub push
            if self.training_config.push_to_hub:
                if self.args.yes:
                    print("🤖 Auto-pushing to Hugging Face Hub (--yes mode)")
                    try:
                        self.trainer.push_to_hub()
                        print("✅ Model pushed to hub!")
                    except Exception as e:
                        print(f"⚠️ Failed to push to hub: {e}")
                elif ask_user_confirmation("Push model to Hugging Face Hub?"):
                    try:
                        self.trainer.push_to_hub()
                        print("✅ Model pushed to hub!")
                    except Exception as e:
                        print(f"⚠️ Failed to push to hub: {e}")
            
            # Print training summary
            print_training_summary(
                train_result, 
                self.training_config.output_dir,
                self.data_config.cache_dir
            )
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Try reducing batch size or gradient accumulation steps")
            
            # Clean up GPU memory
            cleanup_gpu_memory()
            return False
    
    def run_training_pipeline(self) -> bool:
        """Run the complete training pipeline"""
        print("🚀 Starting Enhanced TTS Training Pipeline")
        print("=" * 60)
        
        # Print configuration summary
        print_config_summary(self.training_config, self.data_config, self.model_config)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            return False
        
        # Step 2: Setup models
        if not self.setup_models():
            return False
        
        # Step 3: Setup dataset manager
        if not self.setup_dataset_manager():
            return False
        
        # Step 4: Load and process datasets
        dataset_split = self.load_and_process_datasets()
        if dataset_split is None:
            return False
        
        # Step 5: Setup training components
        if not self.setup_training_components(dataset_split):
            return False
        
        # Step 6: Run training
        if not self.run_training():
            return False
        
        print("\n🎉 Enhanced TTS Training Pipeline Completed Successfully!")
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        cleanup_gpu_memory()
        
        # Clear large objects
        self.trainer = None
        self.model = None
        self.speaker_model = None
        
        print("🧹 Resources cleaned up")


class TrainingMonitor:
    """Monitor training progress and performance"""
    
    def __init__(self, log_interval: int = 25):
        self.log_interval = log_interval
        self.step_count = 0
        self.loss_history = []
        self.memory_history = []
    
    def log_step(self, loss: float, learning_rate: float, step: int):
        """Log a training step"""
        self.step_count = step
        self.loss_history.append(loss)
        
        if step % self.log_interval == 0:
            # Log memory usage if CUDA available
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / 1e9
                self.memory_history.append(memory_usage)
                print(f"Step {step}: Loss={loss:.4f}, LR={learning_rate:.2e}, Memory={memory_usage:.1f}GB")
            else:
                print(f"Step {step}: Loss={loss:.4f}, LR={learning_rate:.2e}")
    
    def get_statistics(self) -> dict:
        """Get training statistics"""
        if not self.loss_history:
            return {}
        
        stats = {
            "total_steps": self.step_count,
            "final_loss": self.loss_history[-1],
            "best_loss": min(self.loss_history),
            "avg_loss": sum(self.loss_history) / len(self.loss_history),
        }
        
        if self.memory_history:
            stats.update({
                "avg_memory": sum(self.memory_history) / len(self.memory_history),
                "peak_memory": max(self.memory_history),
            })
        
        return stats
    
    def plot_training_curves(self, output_path: str = None):
        """Plot training curves"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss curve
            axes[0].plot(self.loss_history)
            axes[0].set_title("Training Loss")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True)
            
            # Memory usage curve
            if self.memory_history:
                axes[1].plot(self.memory_history)
                axes[1].set_title("Memory Usage")
                axes[1].set_xlabel("Step")
                axes[1].set_ylabel("Memory (GB)")
                axes[1].grid(True)
            else:
                axes[1].text(0.5, 0.5, "No memory data", ha='center', va='center')
                axes[1].set_title("Memory Usage (N/A)")
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"📊 Training curves saved to {output_path}")
            
            plt.show()
            
        except ImportError:
            print("⚠️ matplotlib not available - cannot plot training curves")
        except Exception as e:
            print(f"⚠️ Failed to plot training curves: {e}")


def create_trainer(args, has_gpu: bool) -> EnhancedTTSTrainer:
    """
    Factory function to create an enhanced TTS trainer
    
    Args:
        args: Command line arguments
        has_gpu: Whether GPU is available
    
    Returns:
        EnhancedTTSTrainer instance
    """
    return EnhancedTTSTrainer(args, has_gpu)


def run_training_from_config(config_path: str, 
                           cache_dir: str = "cache",
                           force_reprocess: bool = False,
                           yes_mode: bool = False) -> bool:
    """
    Run training from a saved configuration file
    
    Args:
        config_path: Path to configuration JSON file
        cache_dir: Cache directory override
        force_reprocess: Force reprocessing override
        yes_mode: Non-interactive mode
    
    Returns:
        True if training completed successfully
    """
    try:
        from config.training_config import EnhancedTrainingConfig
        
        # Load configuration
        config = EnhancedTrainingConfig.load_from_file(config_path)
        
        # Create mock args object
        class MockArgs:
            def __init__(self):
                self.cache_dir = cache_dir
                self.force_reprocess = force_reprocess
                self.yes = yes_mode
                self.batch_size = config.per_device_train_batch_size
                self.output_dir = config.output_dir
                self.max_steps = config.max_steps
                self.learning_rate = config.learning_rate
                self.push_to_hub = config.push_to_hub
        
        args = MockArgs()
        
        # Check GPU and create trainer
        from utils.gpu_utils import check_gpu
        has_gpu = check_gpu()
        
        trainer = create_trainer(args, has_gpu)
        
        # Override trainer config with loaded config
        trainer.training_config = config
        
        # Run training
        return trainer.run_training_pipeline()
        
    except Exception as e:
        print(f"❌ Failed to run training from config: {e}")
        return False


def resume_training(checkpoint_path: str, 
                   additional_steps: int = 1000,
                   yes_mode: bool = False) -> bool:
    """
    Resume training from a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint directory
        additional_steps: Additional steps to train
        yes_mode: Non-interactive mode
    
    Returns:
        True if resume successful
    """
    try:
        from models.model_setup import ModelCheckpoint
        
        checkpoint = ModelCheckpoint(os.path.dirname(checkpoint_path))
        model, processor, metadata = checkpoint.load_checkpoint(checkpoint_path)
        
        print(f"📂 Resuming from step {metadata.get('step', 0)}")
        print(f"📈 Previous loss: {metadata.get('loss', 'N/A')}")
        
        # Create mock args for resuming
        class MockArgs:
            def __init__(self):
                self.cache_dir = "cache"
                self.force_reprocess = False
                self.yes = yes_mode
                self.batch_size = None
                self.output_dir = checkpoint_path
                self.max_steps = metadata.get('step', 0) + additional_steps
                self.learning_rate = 1e-6  # Lower LR for resume
                self.push_to_hub = False
        
        args = MockArgs()
        
        # Continue with existing training setup...
        print("⚠️ Resume training functionality needs full implementation")
        print("💡 For now, use --force-reprocess with lower learning rate")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to resume training: {e}")
        return False
