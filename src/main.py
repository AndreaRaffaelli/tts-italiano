#!/usr/bin/env python
"""
Enhanced TTS Training - Main Entry Point
"""

import argparse
import sys
import torch
from pathlib import Path

from utils.cli_utils import print_step, ask_user_confirmation
from utils.cache_utils import get_cache_paths, clear_cache
from utils.gpu_utils import check_gpu
from training.trainer import EnhancedTTSTrainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced SpeechT5 Italian Fine-tuning Script')
    
    # Data processing options
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing of cached data')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached data and start fresh')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Directory to store cache files (default: cache)')
    
    # Training options
    parser.add_argument('--output-dir', type=str, default='speecht5_italian_enhanced',
                       help='Output directory for the model (default: speecht5_italian_enhanced)')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum training steps (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override automatic batch size detection')
    parser.add_argument('--learning-rate', type=float, default=3e-6,
                       help='Learning rate (default: 3e-6)')
    
    # Hub options
    parser.add_argument('--push-to-hub', action='store_true',
                       help='Push model to Hugging Face Hub after training')
    
    # Interface options
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Run without interactive prompts (assume yes to all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    print("🚀 Enhanced Italian TTS Training Pipeline")
    print("==========================================")
    
    if args.yes:
        print("🤖 Running in non-interactive mode (--yes flag)")
    
    # Clear cache if requested
    if args.clear_cache:
        print("🗑️ Clearing cache...")
        cache_paths = get_cache_paths(args.cache_dir)
        clear_cache(cache_paths)
        print("✅ Cache cleared")
        sys.exit(0)
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    if not args.yes and not ask_user_confirmation("Do you want to continue with the training?"):
        print("Training cancelled by user.")
        return
    
    # Initialize trainer
    trainer = EnhancedTTSTrainer(args, has_gpu)
    
    try:
        # Run the complete training pipeline
        result = trainer.run_training_pipeline()
        
        if result:
            print("\n🎉 Enhanced Fine-tuning completed successfully!")
            print(f"📁 Model saved in: {args.output_dir}/")
            print(f"📂 Cache files saved in: {Path(args.cache_dir).absolute()}/")
            print("\n💡 Next time, run without --force-reprocess to use cached data!")
        else:
            print("❌ Training failed or was cancelled")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Optimize threading for training
    torch.set_num_threads(1)
    main()
