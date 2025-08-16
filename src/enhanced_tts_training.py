#!/usr/bin/env python
# Training TTS migliorato per italiano con CLI completa

import torch
import torchaudio
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    SpeechT5Processor, SpeechT5ForTextToSpeech, 
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from speechbrain.inference.classifiers import EncoderClassifier
import numpy as np
from scipy import signal
from pathlib import Path
import os
import sys
import pickle
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import re

# ========================================
# CLI UTILITIES (dal tuo script originale)
# ========================================

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

def get_cache_paths(cache_dir="cache"):
    """Get standardized cache file paths"""
    cache_dir = Path(cache_dir)
    return {
        'raw_dataset': cache_dir / "enhanced_raw_dataset.pkl",
        'processed_dataset': cache_dir / "enhanced_processed_dataset.pkl",
        'speaker_counts': cache_dir / "enhanced_speaker_counts.pkl",
        'filtered_dataset': cache_dir / "enhanced_filtered_dataset.pkl",
        'final_dataset': cache_dir / "enhanced_final_dataset.pkl",
        'combined_dataset': cache_dir / "enhanced_combined_dataset.pkl"
    }

# ========================================
# PREPROCESSING FUNCTIONS
# ========================================

def enhanced_audio_preprocessing(waveform, sample_rate=16000):
    """Preprocessing audio migliorato per TTS"""
    
    # 1. Normalizzazione volume
    if torch.max(torch.abs(waveform)) > 0:
        waveform = waveform / torch.max(torch.abs(waveform))
    
    # 2. Rimozione silenzio iniziale/finale più aggressiva
    # Usa un threshold più basso per preservare consonanti sorde
    silence_threshold = 0.01
    non_silent = torch.abs(waveform) > silence_threshold
    if non_silent.any():
        start_idx = torch.where(non_silent)[0][0]
        end_idx = torch.where(non_silent)[0][-1]
        waveform = waveform[start_idx:end_idx + 1]
    
    # 3. High-pass filter per rimuovere rumori bassi
    if len(waveform) > 0:
        waveform_np = waveform.numpy()
        sos = signal.butter(4, 80, 'hp', fs=sample_rate, output='sos')
        waveform_filtered = signal.sosfilt(sos, waveform_np)
        waveform = torch.tensor(waveform_filtered, dtype=torch.float32)
    
    # 4. Limitazione lunghezza (importante per stabilità)
    max_length = sample_rate * 10  # 10 secondi max
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    
    return waveform

def improved_text_preprocessing(text):
    """Preprocessing testo migliorato per italiano"""
    
    if not text or not isinstance(text, str):
        return ""
    
    # Normalizzazione caratteri italiani
    replacements = {
        'à': 'a', 'è': 'e', 'é': 'e', 'í': 'i', 'ì': 'i',
        'ò': 'o', 'ó': 'o', 'ù': 'u', 'ú': 'u', 'ü': 'u',
        'ç': 'c', 'ñ': 'n'
    }
    
    # Applica sostituzioni
    for src, dst in replacements.items():
        text = text.replace(src, dst)
        text = text.replace(src.upper(), dst.upper())
    
    # Rimuovi caratteri speciali ma mantieni punteggiatura italiana
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', ' ', text)
    
    # Normalizza spazi
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Gestisci abbreviazioni comuni italiane
    abbreviations = {
        'dott.': 'dottor', 'dr.': 'dottor',
        'prof.': 'professor', 'ing.': 'ingegner',
        'sig.': 'signor', 'sig.ra': 'signora',
        'avv.': 'avvocato', 'on.': 'onorevole'
    }
    
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
        text = text.replace(abbr.capitalize(), full.capitalize())
    
    return text

# ========================================
# DATASET LOADING
# ========================================

def load_multiple_italian_datasets(cache_paths, force_reprocess=False):
    """Carica e combina multiple datasets italiani con cache"""
    
    # Prova a caricare da cache
    if not force_reprocess:
        combined = load_cache(cache_paths['combined_dataset'], "combined dataset")
        if combined is not None:
            return combined
    
    datasets = []
    

    target_sampling_rate = 16000  # pick one rate for all datasets

    print("📂 Loading VoxPopuli...")
    vox = load_dataset("facebook/voxpopuli", "it", split="train[:50000]", trust_remote_code=True)
    if "normalized_text" in vox.column_names:
        vox = vox.rename_column("normalized_text", "text")
    datasets.append(vox)

    print("📂 Loading Mozilla Common Voice...")
    cv = load_dataset("mozilla-foundation/common_voice_11_0", "it", split="train[:30000]", trust_remote_code=True)
    cv = cv.rename_column("sentence", "text")
    keep_columns = ["audio", "text"]
    remove_columns = [col for col in cv.column_names if col not in keep_columns]
    cv = cv.remove_columns(remove_columns)
    cv = cv.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
    datasets.append(cv)

    # Now they should match
    combined = concatenate_datasets(datasets)

    # 3. Combina datasets
    if datasets:
        combined = concatenate_datasets(datasets)
        print(f"✅ Combined dataset: {len(combined)} samples")
        
        # Salva in cache
        save_cache(combined, cache_paths['combined_dataset'], "combined dataset")
        
        return combined
    else:
        print("❌ No datasets loaded - using VoxPopuli fallback")
        fallback = load_dataset("facebook/voxpopuli", "it", split="train", trust_remote_code=True)
        if "normalized_text" in fallback.column_names:
            fallback = fallback.rename_column("normalized_text", "text")
        return fallback

# ========================================
# MAIN TRAINING PIPELINE
# ========================================

def enhanced_training_pipeline(args):
    """Pipeline di training migliorato con CLI"""
    
    print("🚀 Enhanced Italian TTS Training Pipeline")
    print("==========================================")
    
    if args.yes:
        print("🤖 Running in non-interactive mode (--yes flag)")
    
    # Get cache paths
    cache_paths = get_cache_paths(args.cache_dir)
    
    # Check GPU first
    has_gpu = check_gpu()
    
    if not args.yes and not ask_user_confirmation("Do you want to continue with the training?"):
        print("Training cancelled by user.")
        return
    
    # Step 1: Dependencies check
    print_step(1, "Dependencies Check")
    print("Make sure you have installed:")
    print("- pip install datasets==3.6.0")
    print("- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("- pip install soundfile speechbrain accelerate")
    print("- pip install git+https://github.com/huggingface/transformers.git")
    
    if not args.yes and not ask_user_confirmation("Are all dependencies installed?"):
        print("Please install dependencies first.")
        return
    
    # Step 2: Load datasets
    print_step(2, "Loading Multiple Italian Datasets")
    dataset = load_multiple_italian_datasets(cache_paths, args.force_reprocess)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Step 3: Text preprocessing
    print_step(3, "Text Preprocessing")
    
    def preprocess_text(example):
        text = example["text"]
        example["text"] = improved_text_preprocessing(text)
        return example
    
    # Prova a caricare da cache
    processed_text_dataset = None
    if not args.force_reprocess:
        processed_text_dataset = load_cache(cache_paths['raw_dataset'], "text preprocessed dataset")
    
    if processed_text_dataset is None:
        processed_text_dataset = dataset.map(preprocess_text)
        save_cache(processed_text_dataset, cache_paths['raw_dataset'], "text preprocessed dataset")
    
    dataset = processed_text_dataset
    
    # Step 4: Quality filtering
    print_step(4, "Quality Filtering")
    
    def quality_filter(example):
        text = example["text"]
        audio_length = len(example["audio"]["array"]) / 16000
        
        # Filtri qualità
        if len(text) < 10 or len(text) > 300:  # Lunghezza testo ragionevole
            return False
        if audio_length < 1.0 or audio_length > 10.0:  # Durata audio ragionevole
            return False
        if not text.strip():  # Testo non vuoto
            return False
        
        # Controlla se ha troppi caratteri speciali
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / len(text) > 0.3:  # Max 30% caratteri speciali
            return False
        
        return True
    
    # Prova a caricare da cache
    filtered_dataset = None
    if not args.force_reprocess:
        filtered_dataset = load_cache(cache_paths['filtered_dataset'], "quality filtered dataset")
    
    if filtered_dataset is None:
        filtered_dataset = dataset.filter(quality_filter)
        save_cache(filtered_dataset, cache_paths['filtered_dataset'], "quality filtered dataset")
        print(f"✅ After quality filter: {len(filtered_dataset)} samples")
    else:
        print(f"✅ Using cached filtered dataset: {len(filtered_dataset)} samples")
    
    dataset = filtered_dataset
    
    # Step 5: Load models
    print_step(5, "Loading Models")
    checkpoint = "microsoft/speecht5_tts"
    processor = SpeechT5Processor.from_pretrained(checkpoint)
    model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
    model.config.use_cache = False
    
    # Speaker model
    try:
        from accelerate.test_utils.testing import get_backend
        device, _, _ = get_backend()
        print(f"🔧 Using device: {device}")
    except:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": device},
        savedir="/tmp/speaker_model"
    )
    
    # ---------- Step 6 (Two-phase CPU+GPU preprocessing) ----------
    print_step(6, "Dataset Preprocessing (Audio + Speaker Embeddings)")

    # PHASE A: CPU-only preprocessing (parallel)
    def _cpu_prepare(example):
        """CPU-only: convert audio->waveform, light preprocessing, filter by length.
        Return a dict with 'text' and 'waveform' (numpy float32) or empty waveform to drop later.
        """
        try:
            audio = example["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)

            # Your existing CPU-only preprocessing function (must NOT use CUDA)
            waveform = enhanced_audio_preprocessing(waveform)

            # Drop too-short files
            if waveform.numel() < 8000:  # < 0.5s at 16k
                return {"text": example.get("text", ""), "waveform": []}

            return {"text": example.get("text", ""), "waveform": waveform.numpy().astype("float32")}
        except Exception as e:
            # keep consistent return shape to avoid pickling problems
            print(f"⚠️ CPU preprocess error: {e}")
            return {"text": example.get("text", ""), "waveform": []}


    # PHASE B: GPU finalization (single process, safe to use CUDA)
    def _gpu_finalize(example):
        """Single-process GPU work: tokenization via processor + speaker embeddings."""
        try:
            # If waveform was flagged empty, propagate empty to be filtered out
            if not example.get("waveform"):
                return {"drop": True}

            # Convert numpy -> torch
            waveform = torch.tensor(example["waveform"], dtype=torch.float32)

            # Processor: convert audio -> model input (this may allocate on CPU; fine)
            processed = processor(
                text=example.get("text", ""),
                audio_target=waveform.numpy(),
                sampling_rate=16000,
                return_attention_mask=False,
            )

            # length check
            if len(processed["input_ids"]) > 250:
                return {"drop": True}

            # fix labels shape if needed
            processed["labels"] = processed["labels"][0]

            # Speaker embeddings: use GPU safely (this function runs on main process)
            with torch.no_grad():
                # Ensure speaker_model is on the right device
                device = next(speaker_model.parameters()).device
                waveform = waveform.to(device)
                speaker_emb = speaker_model.encode_batch(waveform.unsqueeze(0))
                speaker_emb = torch.nn.functional.normalize(speaker_emb, dim=2)

            processed["speaker_embeddings"] = speaker_emb.squeeze().cpu().numpy()

            # processed now contains input_ids, labels, speaker_embeddings, etc.
            return processed

        except Exception as e:
            print(f"⚠️ GPU finalize error: {e}")
            return {"drop": True}


    # Try loading processed cache first
    processed_dataset = None
    if not args.force_reprocess:
        processed_dataset = load_cache(cache_paths['processed_dataset'], "processed dataset")

    if processed_dataset is None:
        print("🔄 Phase A — CPU preprocessing (parallel)...")
        # Phase A: run CPU-only preprocessing in parallel
        cpu_num_proc = min(4, os.cpu_count() or 1)  # tune as you like
        cpu_columns_to_remove = [c for c in dataset.column_names if c not in ("audio", "text")]

        # Keep 'text' and create 'waveform' column
        dataset_cpu = dataset.map(
            _cpu_prepare,
            remove_columns=dataset.column_names,
            num_proc=cpu_num_proc,
            desc="CPU audio preprocessing",
        )

        # Filter out entries with empty waveform
        dataset_cpu = dataset_cpu.filter(lambda x: len(x["waveform"]) > 0, num_proc=1)

        print(f"✅ Phase A complete: {len(dataset_cpu)} samples after CPU filtering")

        # Phase B: GPU single-process pass
        print("🔄 Phase B — GPU processing (single process) ...")
        # Make sure models are on GPU before calling the single-process map
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        speaker_model.to(device)
        speaker_model.eval()
        # processor typically is CPU-side but fine to have it in scope

        # Run finalize with num_proc=1 (no forking -> safe CUDA)
        processed_dataset = dataset_cpu.map(
            _gpu_finalize,
            num_proc=1,
            desc="GPU finalize: tokens + speaker embeddings",
        )

        # Filter out any entries flagged as drop
        def _keep(x):
            # If drop key exists or processed fields missing, remove
            if x.get("drop", False):
                return False
            # Basic sanity check for speaker_embeddings presence
            return ("speaker_embeddings" in x) and (len(x["speaker_embeddings"]) > 0)

        processed_dataset = processed_dataset.filter(_keep, num_proc=1)

        # Optionally remove intermediate waveform column if present
        if "waveform" in processed_dataset.column_names:
            processed_dataset = processed_dataset.remove_columns(["waveform"])

        # Save cache
        save_cache(processed_dataset, cache_paths['processed_dataset'], "processed dataset")
        print(f"✅ Processed dataset: {len(processed_dataset)} samples")
    else:
        print(f"✅ Using cached processed dataset: {len(processed_dataset)} samples")
    # ---------- End Step 6 ----------

    # Step 7: Final filtering
    print_step(7, "Final Length Filtering")
    
    def is_not_too_long(input_ids):
        return len(input_ids) < 300
    
    # Prova a caricare da cache
    final_dataset = None
    if not args.force_reprocess:
        final_dataset = load_cache(cache_paths['final_dataset'], "final dataset")
    
    if final_dataset is None:
        final_dataset = processed_dataset.filter(is_not_too_long, input_columns=["input_ids"])
        save_cache(final_dataset, cache_paths['final_dataset'], "final dataset")
        print(f"✅ Final dataset: {len(final_dataset)} samples")
    else:
        print(f"✅ Using cached final dataset: {len(final_dataset)} samples")
    
    dataset = final_dataset
    
    # Step 8: Train/validation split
    print_step(8, "Creating Train/Validation Split")
    dataset_split = dataset.train_test_split(test_size=0.15, seed=42)
    print(f"📊 Train samples: {len(dataset_split['train'])}")
    print(f"📊 Validation samples: {len(dataset_split['test'])}")
    
    # Step 9: Data collator
    print_step(9, "Setting up Data Collator")
    
    @dataclass
    class TTSDataCollatorWithPadding:
        processor: Any
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
            label_features = [{"input_values": feature["labels"]} for feature in features]
            speaker_features = [feature["speaker_embeddings"] for feature in features]
            
            batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")
            batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)
            del batch["decoder_attention_mask"]
            
            batch["speaker_embeddings"] = torch.tensor(speaker_features)
            return batch
    
    data_collator = TTSDataCollatorWithPadding(processor=processor)
    
    # Step 10: Training setup
    print_step(10, "Training Configuration")
    
    # Adjust batch sizes based on GPU memory
    if has_gpu:
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
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=3e-6,  # Learning rate più conservativo
        weight_decay=0.01,
        warmup_steps=1500,
        max_steps=args.max_steps,
        
        # Scheduler
        lr_scheduler_type="cosine_with_restarts",
        
        # Ottimizzazioni
        fp16=has_gpu,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        
        # Evaluation e saving
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Logging
        logging_steps=25,
        report_to=["tensorboard"],
        
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=args.push_to_hub,
    )
    
    # Step 11: Initialize trainer
    print_step(11, "Initializing Trainer")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        data_collator=data_collator,
        processing_class=processor,
    )
    
    # Step 12: Start training
    print_step(12, "Starting Training")
    if not args.yes and not ask_user_confirmation("Start training now?"):
        print("Training cancelled by user.")
        return
    
    try:
        print("🚀 Training started...")
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Save the model
        print("💾 Saving model and processor...")
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print("✅ Model saved!")
        
        # Push to hub if requested
        if args.push_to_hub:
            if args.yes:
                print("🤖 Auto-pushing to Hugging Face Hub (--yes mode)")
                try:
                    trainer.push_to_hub()
                    print("✅ Model pushed to hub!")
                except Exception as e:
                    print(f"⚠️ Failed to push to hub: {e}")
            elif ask_user_confirmation("Push model to Hugging Face Hub?"):
                trainer.push_to_hub()
                print("✅ Model pushed to hub!")
                
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Try reducing batch size or gradient accumulation steps")
        return
    
    print("\n🎉 Enhanced Fine-tuning completed successfully!")
    print(f"📁 Model saved in: {args.output_dir}/")
    print(f"📂 Cache files saved in: {Path(args.cache_dir).absolute()}/")
    print("\n💡 Next time, run without --force-reprocess to use cached data!")
    
    return trainer, model, processor

# ========================================
# MAIN FUNCTION
# ========================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced SpeechT5 Italian Fine-tuning Script')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing of cached data')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached data and start fresh')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Directory to store cache files (default: cache)')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Run without interactive prompts (assume yes to all)')
    parser.add_argument('--output-dir', type=str, default='speecht5_italian_enhanced',
                       help='Output directory for the model (default: speecht5_italian_enhanced)')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum training steps (default: 10000)')
    parser.add_argument('--push-to-hub', action='store_true',
                       help='Push model to Hugging Face Hub after training')
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        print("🗑️ Clearing cache...")
        cache_paths = get_cache_paths(args.cache_dir)
        for cache_path in cache_paths.values():
            if cache_path.exists():
                cache_path.unlink()
                print(f"   Deleted: {cache_path}")
        print("✅ Cache cleared")
        sys.exit(0)
    
    # Run training pipeline
    torch.set_num_threads(1)
    enhanced_training_pipeline(args)

if __name__ == "__main__":
    main()