"""
CLI Utilities for Enhanced TTS Training
"""


def print_step(step_num, description):
    """Print a formatted step header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")


def ask_user_confirmation(message):
    """Ask user for confirmation before proceeding"""
    response = input(f"{message} (y/n): ").lower().strip()
    return response in ['y', 'yes']


def print_dependencies_info():
    """Print installation information for dependencies"""
    print("Make sure you have installed:")
    print("- pip install datasets==3.6.0")
    print("- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("- pip install soundfile speechbrain accelerate")
    print("- pip install git+https://github.com/huggingface/transformers.git")


def print_dataset_stats(dataset, name="Dataset"):
    """Print dataset statistics"""
    print(f"📊 {name} statistics:")
    print(f"   Total samples: {len(dataset)}")
    
    # Sample some text lengths if available
    if "text" in dataset.column_names:
        sample_texts = dataset.select(range(min(100, len(dataset))))["text"]
        text_lengths = [len(text) for text in sample_texts if text]
        if text_lengths:
            avg_text_len = sum(text_lengths) / len(text_lengths)
            print(f"   Avg text length: {avg_text_len:.1f} characters")
            print(f"   Text length range: {min(text_lengths)}-{max(text_lengths)}")


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_training_summary(trainer_result, output_dir, cache_dir):
    """Print training completion summary"""
    print("\n🎉 Training Summary")
    print("==================")
    print(f"✅ Training completed successfully!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"📂 Cache stored in: {cache_dir}")
    
    if trainer_result and hasattr(trainer_result, 'log_history'):
        final_loss = None
        for log_entry in reversed(trainer_result.log_history):
            if 'train_loss' in log_entry:
                final_loss = log_entry['train_loss']
                break
        
        if final_loss:
            print(f"📈 Final training loss: {final_loss:.4f}")
