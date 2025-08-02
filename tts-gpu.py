#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import pickle
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path

# Add interactive prompts and logging
def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")

def ask_user_confirmation(message):
    """Ask user for confirmation before proceeding"""
    response = input(f"{message} (y/n): ").lower().strip()
    return response in ['y', 'yes']

def check_gpu():
    """Check GPU availability and memory"""
    print("\n🔍 Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🆔 CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("❌ CUDA is not available - training will be slow!")
        return False

def save_cache(data, cache_path, description="data"):
    """Save data to cache file"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 {description} saved to cache: {cache_path}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to save {description} to cache: {e}")
        return False

def load_cache(cache_path, description="data"):
    """Load data from cache file"""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"📂 {description} loaded from cache: {cache_path}")
            return data
        else:
            print(f"📂 No cache found for {description}")
            return None
    except Exception as e:
        print(f"⚠️ Failed to load {description} from cache: {e}")
        return None

def get_cache_paths():
    """Get standardized cache file paths"""
    cache_dir = Path("cache")
    return {
        'raw_dataset': cache_dir / "raw_dataset.pkl",
        'processed_dataset': cache_dir / "processed_dataset.pkl",
        'speaker_counts': cache_dir / "speaker_counts.pkl",
        'filtered_dataset': cache_dir / "filtered_dataset.pkl",
        'final_dataset': cache_dir / "final_dataset.pkl"
    }

# Main execution
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SpeechT5 Italian Fine-tuning Script')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing of cached data')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached data and start fresh')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Directory to store cache files (default: cache)')
    args = parser.parse_args()
    
    print("🎤 SpeechT5 Italian Fine-tuning Script")
    print("=====================================")
    
    # Get cache paths
    cache_paths = get_cache_paths()
    if args.cache_dir != 'cache':
        cache_paths = {k: Path(args.cache_dir) / v.name for k, v in cache_paths.items()}
    
    # Clear cache if requested
    if args.clear_cache:
        print("🗑️ Clearing cache...")
        for cache_path in cache_paths.values():
            if cache_path.exists():
                cache_path.unlink()
                print(f"   Deleted: {cache_path}")
        print("✅ Cache cleared")
        exit(0)
    
    # Check GPU first
    has_gpu = check_gpu()
    
    if not ask_user_confirmation("Do you want to continue with the training?"):
        print("Training cancelled by user.")
        return
    
    # Step 1: Install dependencies (commented out - run manually if needed)
    print_step(1, "Dependencies")
    print("Make sure you have installed:")
    print("- pip install datasets==3.6.0")
    print("- pip install soundfile speechbrain accelerate")
    print("- pip install git+https://github.com/huggingface/transformers.git")
    
    if not ask_user_confirmation("Are all dependencies installed?"):
        print("Please install dependencies first.")
        return
    
    # Step 2: Load dataset
    print_step(2, "Loading VoxPopuli Italian Dataset")
    
    # Try to load from cache first
    dataset = None
    if not args.force_reprocess:
        dataset = load_cache(cache_paths['raw_dataset'], "raw dataset")
    
    if dataset is None:
        try:
            from datasets import load_dataset, Audio
            dataset = load_dataset("facebook/voxpopuli", "it", split="train", trust_remote_code=True)
            print(f"✅ Dataset loaded successfully: {len(dataset)} samples")
            
            # Cache the raw dataset
            save_cache(dataset, cache_paths['raw_dataset'], "raw dataset")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return
    else:
        print(f"✅ Using cached dataset: {len(dataset)} samples")
    
    # Step 3: Prepare audio
    print_step(3, "Preparing Audio Data")
    SAMPLING_RATE = 16000
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    print(f"✅ Audio resampled to {SAMPLING_RATE}Hz")
    
    # Step 4: Load processor
    print_step(4, "Loading SpeechT5 Processor")
    try:
        from transformers import SpeechT5Processor
        checkpoint = "microsoft/speecht5_tts"
        processor = SpeechT5Processor.from_pretrained(checkpoint)
        print(f"✅ Processor loaded from {checkpoint}")
    except Exception as e:
        print(f"❌ Error loading processor: {e}")
        return
    
    # Step 5: Text preprocessing
    print_step(5, "Text Preprocessing and Vocabulary Check")
    tokenizer = processor.tokenizer
    
    def extract_all_chars(batch):
        all_text = " ".join(batch["normalized_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}
    
    print("🔍 Analyzing vocabulary...")
    vocabs = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )
    
    dataset_vocab = set(vocabs["vocab"][0])
    tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
    missing_chars = dataset_vocab - tokenizer_vocab
    
    print(f"📊 Dataset vocabulary size: {len(dataset_vocab)}")
    print(f"📊 Tokenizer vocabulary size: {len(tokenizer_vocab)}")
    print(f"⚠️ Missing characters: {missing_chars}")
    
    # Text cleanup
    replacements = [
        ("à", "a"), ("è", "e"), ("í", "i"), ("ì", "i"),
        ("ò", "o"), ("ó", "o"), ("ù", "u"), ("ú", "u"),
    ]
    
    def cleanup_text(inputs):
        for src, dst in replacements:
            inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
        return inputs
    
    dataset = dataset.map(cleanup_text)
    print("✅ Text cleanup completed")
    
    # Step 6: Speaker analysis and filtering
    print_step(6, "Speaker Analysis and Filtering")
    
    # Try to load speaker counts from cache
    speaker_counts = None
    filtered_dataset = None
    
    if not args.force_reprocess:
        speaker_counts = load_cache(cache_paths['speaker_counts'], "speaker counts")
        filtered_dataset = load_cache(cache_paths['filtered_dataset'], "filtered dataset")
    
    if speaker_counts is None:
        speaker_counts = defaultdict(int)
        for speaker_id in dataset["speaker_id"]:
            speaker_counts[speaker_id] += 1
        
        # Save speaker counts to cache
        save_cache(dict(speaker_counts), cache_paths['speaker_counts'], "speaker counts")
    else:
        speaker_counts = defaultdict(int, speaker_counts)
    
    print(f"📊 Total speakers: {len(speaker_counts)}")
    print(f"📊 Average samples per speaker: {sum(speaker_counts.values()) / len(speaker_counts):.1f}")
    
    # Plot speaker distribution
    plt.figure(figsize=(10, 6))
    plt.hist(speaker_counts.values(), bins=20)
    plt.ylabel("Number of Speakers")
    plt.xlabel("Number of Examples")
    plt.title("Speaker Distribution")
    plt.show()
    
    if filtered_dataset is None:
        def select_speaker(speaker_id):
            return 100 <= speaker_counts[speaker_id] <= 400
        
        filtered_dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
        
        # Save filtered dataset to cache
        save_cache(filtered_dataset, cache_paths['filtered_dataset'], "filtered dataset")
    
    dataset = filtered_dataset
    print(f"✅ Filtered dataset: {len(dataset)} samples from {len(set(dataset['speaker_id']))} speakers")
    
    # Step 7: Load speaker embedding model
    print_step(7, "Loading Speaker Embedding Model")
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
        from accelerate.test_utils.testing import get_backend
        
        spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        device, _, _ = get_backend()
        print(f"🔧 Using device: {device}")
        
        speaker_model = EncoderClassifier.from_hparams(
            source=spk_model_name,
            run_opts={"device": device},
            savedir=os.path.join("/tmp", spk_model_name),
        )
        print("✅ Speaker embedding model loaded")
    except Exception as e:
        print(f"❌ Error loading speaker model: {e}")
        return
    
    def create_speaker_embedding(waveform):
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings
    
    # Step 8: Dataset preprocessing
    print_step(8, "Dataset Preprocessing")
    
    # Try to load processed dataset from cache
    processed_dataset = None
    if not args.force_reprocess:
        processed_dataset = load_cache(cache_paths['processed_dataset'], "processed dataset")
    
    if processed_dataset is None:
        def prepare_dataset(example):
            audio = example["audio"]
            example = processor(
                text=example["normalized_text"],
                audio_target=audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_attention_mask=False,
            )
            example["labels"] = example["labels"][0]
            example["speaker_embeddings"] = create_speaker_embedding(audio["array"])
            return example
        
        print("🔄 Processing dataset (this may take a while)...")
        print("⚠️ Long sequences (>600 tokens) will be truncated - this is normal")
        
        processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
        
        # Save processed dataset to cache
        save_cache(processed_dataset, cache_paths['processed_dataset'], "processed dataset")
        
        print("✅ Dataset preprocessing completed and cached")
    else:
        print("✅ Using cached processed dataset")
    
    dataset = processed_dataset
    
    # Filter by length and create final dataset
    final_dataset = None
    if not args.force_reprocess:
        final_dataset = load_cache(cache_paths['final_dataset'], "final dataset")
    
    if final_dataset is None:
        def is_not_too_long(input_ids):
            return len(input_ids) < 300
        
        final_dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
        
        # Save final dataset to cache
        save_cache(final_dataset, cache_paths['final_dataset'], "final dataset")
        
        print(f"✅ Dataset filtered and cached: {len(final_dataset)} samples")
    else:
        print(f"✅ Using cached final dataset: {len(final_dataset)} samples")
    
    dataset = final_dataset
    
    # Step 9: Train/test split
    print_step(9, "Creating Train/Test Split")
    dataset = dataset.train_test_split(test_size=0.1)
    print(f"📊 Train samples: {len(dataset['train'])}")
    print(f"📊 Test samples: {len(dataset['test'])}")
    
    # Step 10: Data collator
    print_step(10, "Setting up Data Collator")
    
    @dataclass
    class TTSDataCollatorWithPadding:
        processor: Any
        
        def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
            label_features = [{"input_values": feature["labels"]} for feature in features]
            speaker_features = [feature["speaker_embeddings"] for feature in features]
            
            batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")
            batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)
            del batch["decoder_attention_mask"]
            
            # Handle reduction factor
            if hasattr(model, 'config') and model.config.reduction_factor > 1:
                target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
                target_lengths = target_lengths.new(
                    [length - length % model.config.reduction_factor for length in target_lengths]
                )
                max_length = max(target_lengths)
                batch["labels"] = batch["labels"][:, :max_length]
            
            batch["speaker_embeddings"] = torch.tensor(speaker_features)
            return batch
    
    data_collator = TTSDataCollatorWithPadding(processor=processor)
    print("✅ Data collator ready")
    
    # Step 11: Load model and setup training
    print_step(11, "Loading Model and Setting up Training")
    try:
        from transformers import SpeechT5ForTextToSpeech, Seq2SeqTrainingArguments, Seq2SeqTrainer
        
        model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
        model.config.use_cache = False
        print("✅ Model loaded")
        
        # Adjust batch sizes based on GPU memory
        if has_gpu:
            # Check GPU memory and adjust accordingly
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb < 8:
                batch_size = 1
                grad_acc_steps = 32
                print("⚠️ Low GPU memory detected - using smaller batch sizes")
            elif gpu_memory_gb < 16:
                batch_size = 2
                grad_acc_steps = 16
            else:
                batch_size = 4
                grad_acc_steps = 8
        else:
            batch_size = 1
            grad_acc_steps = 32
        
        print(f"🔧 Batch size: {batch_size}, Gradient accumulation: {grad_acc_steps}")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir="speecht5_finetuned_voxpopuli_it",
            per_device_train_batch_size=batch_size,
            gradient_checkpointing=False,
            gradient_accumulation_steps=grad_acc_steps,
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=4000,
            fp16=has_gpu,  # Only use fp16 if GPU is available
            eval_strategy="steps",
            per_device_eval_batch_size=1,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            greater_is_better=False,
            label_names=["labels"],
            push_to_hub=False,  # Set to True if you want to push to hub
        )
        
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            processing_class=processor
        )
        
        print("✅ Trainer initialized")
        
    except Exception as e:
        print(f"❌ Error setting up training: {e}")
        return
    
    # Step 12: Start training
    print_step(12, "Starting Training")
    if not ask_user_confirmation("Start training now?"):
        print("Training cancelled by user.")
        return
    
    try:
        print("🚀 Training started...")
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Save the model
        print("💾 Saving model and processor...")
        trainer.save_model("speecht5_finetuned_voxpopuli_it")
        processor.save_pretrained("speecht5_finetuned_voxpopuli_it")
        print("✅ Model saved!")
        
        if ask_user_confirmation("Push model to Hugging Face Hub?"):
            trainer.push_to_hub()
            print("✅ Model pushed to hub!")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Try reducing batch size or gradient accumulation steps")
        return
    
    print("\n🎉 Fine-tuning completed successfully!")
    print("📁 Model saved in: speecht5_finetuned_voxpopuli_it/")
    print(f"📂 Cache files saved in: {Path(args.cache_dir).absolute()}/")
    print("\n💡 Next time, run without --force-reprocess to use cached data!")

if __name__ == "__main__":
    main()