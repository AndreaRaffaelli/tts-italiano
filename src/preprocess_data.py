#!/usr/bin/env python
# SpeechT5 Data Preprocessing - CPU Optimized

import torch
import torchaudio
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import SpeechT5Processor
import numpy as np
from scipy import signal
from pathlib import Path
import os
import sys
import pickle
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import json
from datetime import datetime

# ========================================
# CLI UTILITIES
# ========================================

def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")

def ask_user_confirmation(message):
    """Ask user for confirmation before proceeding"""
    response = input(f"{message} (y/n): ").lower().strip()
    return response in ['y', 'yes']

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
        'combined_dataset': cache_dir / "enhanced_combined_dataset.pkl",
        'preprocessed_ready': cache_dir / "preprocessed_ready_for_training.pkl",
        'metadata': cache_dir / "preprocessing_metadata.json"
    }

def save_metadata(cache_dir, dataset_info):
    """Save preprocessing metadata"""
    cache_paths = get_cache_paths(cache_dir)
    metadata = {
        'preprocessing_date': datetime.now().isoformat(),
        'dataset_info': dataset_info,
        'total_samples': dataset_info.get('total_samples', 0),
        'preprocessing_complete': True,
        'cache_dir': str(Path(cache_dir).absolute())
    }
    
    try:
        with open(cache_paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Metadata saved: {cache_paths['metadata']}")
    except Exception as e:
        print(f"⚠️ Failed to save metadata: {e}")

# ========================================
# PREPROCESSING FUNCTIONS
# ========================================

def enhanced_audio_preprocessing(waveform, sample_rate=16000):
    """Preprocessing audio migliorato per TTS"""
    
    # 1. Normalizzazione volume
    if torch.max(torch.abs(waveform)) > 0:
        waveform = waveform / torch.max(torch.abs(waveform))
    
    # 2. Rimozione silenzio iniziale/finale più aggressiva
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

def load_multiple_italian_datasets(cache_paths, force_reprocess=False, max_samples_per_dataset=None):
    """Carica e combina multiple datasets italiani con cache"""
    
    # Prova a caricare da cache
    if not force_reprocess:
        combined = load_cache(cache_paths['combined_dataset'], "combined dataset")
        if combined is not None:
            return combined
    
    datasets = []
    target_sampling_rate = 16000
    
    try:
        print("📂 Loading VoxPopuli...")
        max_vox = max_samples_per_dataset or 50000
        vox = load_dataset("facebook/voxpopuli", "it", split=f"train[:{max_vox}]", trust_remote_code=True)
        if "normalized_text" in vox.column_names:
            vox = vox.rename_column("normalized_text", "text")
        vox = vox.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
        datasets.append(vox)
        print(f"✅ VoxPopuli loaded: {len(vox)} samples")
    except Exception as e:
        print(f"⚠️ Failed to load VoxPopuli: {e}")

    try:
        print("📂 Loading Mozilla Common Voice...")
        max_cv = max_samples_per_dataset or 30000
        cv = load_dataset("mozilla-foundation/common_voice_11_0", "it", split=f"train[:{max_cv}]", trust_remote_code=True)
        cv = cv.rename_column("sentence", "text")
        keep_columns = ["audio", "text"]
        remove_columns = [col for col in cv.column_names if col not in keep_columns]
        cv = cv.remove_columns(remove_columns)
        cv = cv.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
        datasets.append(cv)
        print(f"✅ Common Voice loaded: {len(cv)} samples")
    except Exception as e:
        print(f"⚠️ Failed to load Common Voice: {e}")

    # Combina datasets
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
# MAIN PREPROCESSING PIPELINE
# ========================================

def preprocessing_pipeline(args):
    """Pipeline di preprocessing completo - solo CPU"""
    
    print("📊 SpeechT5 Italian Data Preprocessing Pipeline")
    print("================================================")
    print("🖥️  CPU-Only Mode - Optimized for Data Preparation")
    
    if args.yes:
        print("🤖 Running in non-interactive mode (--yes flag)")
    
    # Get cache paths
    cache_paths = get_cache_paths(args.cache_dir)
    
    # Step 1: Dependencies check
    print_step(1, "Dependencies Check")
    print("CPU-optimized preprocessing requires:")
    print("- pip install datasets==3.6.0")
    print("- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("- pip install soundfile librosa scipy")
    print("- pip install transformers")
    
    if not args.yes and not ask_user_confirmation("Are all dependencies installed?"):
        print("Please install dependencies first.")
        return
    
    # Step 2: Load datasets
    print_step(2, "Loading Multiple Italian Datasets")
    dataset = load_multiple_italian_datasets(cache_paths, args.force_reprocess, args.max_samples)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    print(f"📊 Initial dataset size: {len(dataset)} samples")
    
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
        print("🔄 Processing text (parallel)...")
        num_proc = min(args.num_workers, os.cpu_count() or 1)
        processed_text_dataset = dataset.map(
            preprocess_text, 
            num_proc=num_proc,
            desc="Text preprocessing"
        )
        save_cache(processed_text_dataset, cache_paths['raw_dataset'], "text preprocessed dataset")
    
    dataset = processed_text_dataset
    
    # Step 4: Quality filtering
    print_step(4, "Quality Filtering")
    
    def quality_filter(example):
        text = example["text"]
        audio_length = len(example["audio"]["array"]) / 16000
        
        # Filtri qualità
        if len(text) < 10 or len(text) > 300:
            return False
        if audio_length < 1.0 or audio_length > 10.0:
            return False
        if not text.strip():
            return False
        
        # Controlla caratteri speciali
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / len(text) > 0.3:
            return False
        
        return True
    
    # Prova a caricare da cache
    filtered_dataset = None
    if not args.force_reprocess:
        filtered_dataset = load_cache(cache_paths['filtered_dataset'], "quality filtered dataset")
    
    if filtered_dataset is None:
        print("🔄 Applying quality filters...")
        filtered_dataset = dataset.filter(quality_filter, num_proc=1)  # Single process for stability
        save_cache(filtered_dataset, cache_paths['filtered_dataset'], "quality filtered dataset")
        print(f"✅ After quality filter: {len(filtered_dataset)} samples")
    else:
        print(f"✅ Using cached filtered dataset: {len(filtered_dataset)} samples")
    
    dataset = filtered_dataset
    
    # Step 5: Audio preprocessing
    print_step(5, "Audio Preprocessing (CPU Optimized)")
    
    def cpu_audio_preprocess(example):
        """CPU-only audio preprocessing"""
        try:
            audio = example["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)
            
            # Apply audio preprocessing
            waveform = enhanced_audio_preprocessing(waveform)
            
            # Filter out too short audio
            if waveform.numel() < 8000:  # < 0.5s at 16kHz
                return {"text": example["text"], "waveform": [], "drop": True}
            
            return {
                "text": example["text"],
                "waveform": waveform.numpy().astype("float32"),
                "drop": False
            }
        except Exception as e:
            print(f"⚠️ Audio preprocessing error: {e}")
            return {"text": example["text"], "waveform": [], "drop": True}
    
    # Prova a caricare da cache
    audio_processed_dataset = None
    if not args.force_reprocess:
        audio_processed_dataset = load_cache(cache_paths['processed_dataset'], "audio processed dataset")
    
    if audio_processed_dataset is None:
        print("🔄 Processing audio (CPU optimized)...")
        num_proc = min(args.num_workers, os.cpu_count() or 1)
        
        audio_processed_dataset = dataset.map(
            cpu_audio_preprocess,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="Audio preprocessing"
        )
        
        # Filter out dropped samples
        audio_processed_dataset = audio_processed_dataset.filter(
            lambda x: not x.get("drop", False),
            num_proc=1
        )
        
        # Remove the drop column
        if "drop" in audio_processed_dataset.column_names:
            audio_processed_dataset = audio_processed_dataset.remove_columns(["drop"])
        
        save_cache(audio_processed_dataset, cache_paths['processed_dataset'], "audio processed dataset")
        print(f"✅ Audio processing complete: {len(audio_processed_dataset)} samples")
    else:
        print(f"✅ Using cached audio processed dataset: {len(audio_processed_dataset)} samples")
    
    dataset = audio_processed_dataset
    
    # Step 6: Tokenization (CPU-only, no speaker embeddings yet)
    print_step(6, "Text Tokenization")
    
    print("📥 Loading SpeechT5 processor...")
    checkpoint = "microsoft/speecht5_tts"
    processor = SpeechT5Processor.from_pretrained(checkpoint)
    
    def tokenize_text(example):
        """Tokenize text only - no speaker embeddings"""
        try:
            text = example["text"]
            waveform = example["waveform"]
            
            if len(waveform) == 0:
                return {"input_ids": [], "labels": [], "drop": True}
            
            # Process text to get input_ids
            text_inputs = processor.tokenizer(
                text,
                return_attention_mask=False,
                return_tensors="np"
            )
            
            # Process audio to get labels
            waveform_tensor = torch.tensor(waveform)
            audio_inputs = processor.feature_extractor(
                waveform_tensor.numpy(),
                sampling_rate=16000,
                return_tensors="np"
            )
            
            # Check length limits
            if len(text_inputs["input_ids"][0]) > 250:
                return {"input_ids": [], "labels": [], "drop": True}
            
            return {
                "input_ids": text_inputs["input_ids"][0],
                "labels": audio_inputs["input_values"][0],
                "text": text,
                "drop": False
            }
        except Exception as e:
            print(f"⚠️ Tokenization error: {e}")
            return {"input_ids": [], "labels": [], "drop": True}
    
    # Prova a caricare da cache
    tokenized_dataset = None
    if not args.force_reprocess:
        tokenized_dataset = load_cache(cache_paths['final_dataset'], "tokenized dataset")
    
    if tokenized_dataset is None:
        print("🔄 Tokenizing text and audio...")
        tokenized_dataset = dataset.map(
            tokenize_text,
            remove_columns=dataset.column_names,
            num_proc=1,  # Single process for processor stability
            desc="Tokenization"
        )
        
        # Filter out dropped samples
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: not x.get("drop", False),
            num_proc=1
        )
        
        # Remove drop column
        if "drop" in tokenized_dataset.column_names:
            tokenized_dataset = tokenized_dataset.remove_columns(["drop"])
        
        save_cache(tokenized_dataset, cache_paths['final_dataset'], "tokenized dataset")
        print(f"✅ Tokenization complete: {len(tokenized_dataset)} samples")
    else:
        print(f"✅ Using cached tokenized dataset: {len(tokenized_dataset)} samples")
    
    # Step 7: Final preparation and save
    print_step(7, "Final Dataset Preparation")
    
    # Save final preprocessed dataset ready for training
    final_cache_path = cache_paths['preprocessed_ready']
    save_cache(tokenized_dataset, final_cache_path, "final preprocessed dataset")
    
    # Generate dataset statistics
    dataset_info = {
        'total_samples': len(tokenized_dataset),
        'avg_text_length': np.mean([len(ex['input_ids']) for ex in tokenized_dataset]),
        'avg_audio_length': np.mean([len(ex['labels']) for ex in tokenized_dataset]),
        'preprocessing_settings': {
            'max_samples_per_dataset': args.max_samples,
            'num_workers': args.num_workers,
            'cache_dir': args.cache_dir,
        }
    }
    
    # Save metadata
    save_metadata(args.cache_dir, dataset_info)
    
    print(f"\n🎉 Preprocessing completed successfully!")
    print(f"📊 Final dataset: {dataset_info['total_samples']} samples")
    print(f"📁 Preprocessed data saved to: {final_cache_path}")
    print(f"📋 Metadata saved to: {get_cache_paths(args.cache_dir)['metadata']}")
    print(f"\n💡 Ready for training! Transfer the '{args.cache_dir}' folder to your GPU machine.")
    print(f"💡 On GPU machine, use: python train_model.py --preprocessed-data {args.cache_dir}")
    
    return tokenized_dataset, dataset_info

# ========================================
# MAIN FUNCTION
# ========================================

def main():
    parser = argparse.ArgumentParser(description='SpeechT5 Italian Data Preprocessing (CPU Only)')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing of cached data')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached data and start fresh')
    parser.add_argument('--cache-dir', type=str, default='preprocessed_data',
                       help='Directory to store processed data (default: preprocessed_data)')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Run without interactive prompts')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum samples per dataset (for testing)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of CPU workers for parallel processing')
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        print("🗑️ Clearing preprocessing cache...")
        cache_paths = get_cache_paths(args.cache_dir)
        for cache_path in cache_paths.values():
            if cache_path.exists():
                cache_path.unlink()
                print(f"   Deleted: {cache_path}")
        print("✅ Cache cleared")
        sys.exit(0)
    
    # Run preprocessing pipeline
    preprocessing_pipeline(args)

if __name__ == "__main__":
    main()