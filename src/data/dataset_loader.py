"""
Dataset Loading and Management
"""

import os
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import SpeechT5Processor
from typing import Dict, Any, Optional
from pathlib import Path

from .preprocessing import TextPreprocessor, AudioPreprocessor, quality_filter, length_filter
from utils.cache_utils import load_cache, save_cache
from utils.cli_utils import print_dataset_stats


class DatasetManager:
    """Manages dataset loading, preprocessing, and caching"""
    
    def __init__(self, cache_paths: Dict[str, Path], 
                 processor: SpeechT5Processor,
                 speaker_model: EncoderClassifier,
                 force_reprocess: bool = False,
                 target_sampling_rate: int = 16000):
        self.cache_paths = cache_paths
        self.processor = processor
        self.speaker_model = speaker_model
        self.force_reprocess = force_reprocess
        self.target_sampling_rate = target_sampling_rate
        
        # Initialize preprocessors
        self.text_preprocessor = TextPreprocessor()
        self.audio_preprocessor = AudioPreprocessor(sample_rate=target_sampling_rate)
    
    def load_multiple_italian_datasets(self):
        """Load and combine multiple Italian datasets with caching"""
        
        # Try to load from cache first
        if not self.force_reprocess:
            combined = load_cache(self.cache_paths['combined_dataset'], "combined dataset")
            if combined is not None:
                print_dataset_stats(combined, "Cached Combined Dataset")
                return combined
        
        print("🔄 Loading Italian datasets...")
        datasets = []
        
        # Load VoxPopuli
        print("📂 Loading VoxPopuli...")
        try:
            vox = load_dataset("facebook/voxpopuli", "it", split="train[:50000]", trust_remote_code=True)
            if "normalized_text" in vox.column_names:
                vox = vox.rename_column("normalized_text", "text")
            vox = vox.cast_column("audio", Audio(sampling_rate=self.target_sampling_rate))
            datasets.append(vox)
            print(f"   ✅ VoxPopuli loaded: {len(vox)} samples")
        except Exception as e:
            print(f"   ⚠️ Failed to load VoxPopuli: {e}")
        
        # Load Mozilla Common Voice
        print("📂 Loading Mozilla Common Voice...")
        try:
            cv = load_dataset("mozilla-foundation/common_voice_11_0", "it", split="train[:30000]", trust_remote_code=True)
            cv = cv.rename_column("sentence", "text")
            keep_columns = ["audio", "text"]
            remove_columns = [col for col in cv.column_names if col not in keep_columns]
            cv = cv.remove_columns(remove_columns)
            cv = cv.cast_column("audio", Audio(sampling_rate=self.target_sampling_rate))
            datasets.append(cv)
            print(f"   ✅ Common Voice loaded: {len(cv)} samples")
        except Exception as e:
            print(f"   ⚠️ Failed to load Common Voice: {e}")
        
        # Combine datasets
        if datasets:
            combined = concatenate_datasets(datasets)
            print(f"✅ Combined dataset: {len(combined)} samples")
            
            # Save to cache
            save_cache(combined, self.cache_paths['combined_dataset'], "combined dataset")
            print_dataset_stats(combined, "Combined Dataset")
            return combined
        else:
            print("❌ No datasets loaded - using VoxPopuli fallback")
            fallback = load_dataset("facebook/voxpopuli", "it", split="train", trust_remote_code=True)
            if "normalized_text" in fallback.column_names:
                fallback = fallback.rename_column("normalized_text", "text")
            return fallback
    
    def preprocess_text(self, dataset):
        """Apply text preprocessing to dataset"""
        
        # Try to load from cache
        if not self.force_reprocess:
            processed = load_cache(self.cache_paths['raw_dataset'], "text preprocessed dataset")
            if processed is not None:
                print_dataset_stats(processed, "Cached Text Preprocessed Dataset")
                return processed
        
        print("🔄 Preprocessing text...")
        
        def preprocess_example(example):
            text = example.get("text", "")
            example["text"] = self.text_preprocessor.preprocess(text)
            return example
        
        processed = dataset.map(preprocess_example, desc="Text preprocessing")
        
        # Save to cache
        save_cache(processed, self.cache_paths['raw_dataset'], "text preprocessed dataset")
        print_dataset_stats(processed, "Text Preprocessed Dataset")
        
        return processed
    
    def apply_quality_filters(self, dataset):
        """Apply quality filtering to dataset"""
        
        # Try to load from cache
        if not self.force_reprocess:
            filtered = load_cache(self.cache_paths['filtered_dataset'], "quality filtered dataset")
            if filtered is not None:
                print_dataset_stats(filtered, "Cached Quality Filtered Dataset")
                return filtered
        
        print("🔄 Applying quality filters...")
        
        original_length = len(dataset)
        filtered = dataset.filter(quality_filter, desc="Quality filtering")
        filtered_length = len(filtered)
        
        print(f"   Filtered: {original_length} → {filtered_length} samples")
        print(f"   Kept: {filtered_length/original_length*100:.1f}%")
        
        # Save to cache
        save_cache(filtered, self.cache_paths['filtered_dataset'], "quality filtered dataset")
        print_dataset_stats(filtered, "Quality Filtered Dataset")
        
        return filtered
    
    def process_audio_and_embeddings(self, dataset):
        """Process audio and generate speaker embeddings"""
        
        # Try to load from cache
        if not self.force_reprocess:
            processed = load_cache(self.cache_paths['processed_dataset'], "processed dataset")
            if processed is not None:
                print_dataset_stats(processed, "Cached Processed Dataset")
                return processed
        
        print("🔄 Processing audio and generating embeddings...")
        
        # CPU preprocessing phase
        def cpu_prepare(example):
            """CPU-only preprocessing"""
            try:
                audio = example["audio"]
                waveform = torch.tensor(audio["array"], dtype=torch.float32)
                
                # Apply audio preprocessing
                waveform = self.audio_preprocessor.preprocess(waveform)
                
                if waveform is None or len(waveform) < 8000:  # < 0.5s at 16kHz
                    return {"text": example.get("text", ""), "waveform": []}
                
                return {"text": example.get("text", ""), "waveform": waveform.numpy().astype("float32")}
            except Exception as e:
                print(f"⚠️ CPU preprocess error: {e}")
                return {"text": example.get("text", ""), "waveform": []}
        
        # GPU finalization phase
        def gpu_finalize(example):
            """GPU processing for tokenization and speaker embeddings"""
            try:
                if not example.get("waveform") or len(example["waveform"]) == 0:
                    return {"drop": True}
                
                waveform = torch.tensor(example["waveform"], dtype=torch.float32)
                
                # Process with SpeechT5 processor
                processed = self.processor(
                    text=example.get("text", ""),
                    audio_target=waveform.numpy(),
                    sampling_rate=self.target_sampling_rate,
                    return_attention_mask=False,
                )
                
                # Length check
                if len(processed["input_ids"]) > 250:
                    return {"drop": True}
                
                # Fix labels shape
                if len(processed["labels"].shape) > 2:
                    processed["labels"] = processed["labels"][0]
                
                # Generate speaker embeddings
                with torch.no_grad():
                    device = next(self.speaker_model.parameters()).device
                    waveform = waveform.to(device)
                    speaker_emb = self.speaker_model.encode_batch(waveform.unsqueeze(0))
                    speaker_emb = torch.nn.functional.normalize(speaker_emb, dim=2)
                
                processed["speaker_embeddings"] = speaker_emb.squeeze().cpu().numpy()
                
                return processed
                
            except Exception as e:
                print(f"⚠️ GPU finalize error: {e}")
                return {"drop": True}
        
        # Phase A: CPU preprocessing
        print("   Phase A: CPU preprocessing...")
        cpu_num_proc = min(4, os.cpu_count() or 1)
        dataset_cpu = dataset.map(
            cpu_prepare,
            remove_columns=dataset.column_names,
            num_proc=cpu_num_proc,
            desc="CPU audio preprocessing",
        )
        
        # Filter empty waveforms
        dataset_cpu = dataset_cpu.filter(lambda x: len(x["waveform"]) > 0, num_proc=1)
        print(f"   After CPU phase: {len(dataset_cpu)} samples")
        
        # Phase B: GPU processing
        print("   Phase B: GPU processing...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker_model.to(device)
        self.speaker_model.eval()
        
        processed_dataset = dataset_cpu.map(
            gpu_finalize,
            num_proc=1,
            desc="GPU processing: tokens + embeddings",
        )
        
        # Filter dropped samples
        def keep_valid(x):
            if x.get("drop", False):
                return False
            return ("speaker_embeddings" in x) and (len(x["speaker_embeddings"]) > 0)
        
        processed_dataset = processed_dataset.filter(keep_valid, num_proc=1)
        
        # Remove intermediate columns
        if "waveform" in processed_dataset.column_names:
            processed_dataset = processed_dataset.remove_columns(["waveform"])
        
        # Save to cache
        save_cache(processed_dataset, self.cache_paths['processed_dataset'], "processed dataset")
        print_dataset_stats(processed_dataset, "Processed Dataset")
        
        return processed_dataset
    
    def apply_final_filters(self, dataset):
        """Apply final length filtering"""
        
        # Try to load from cache
        if not self.force_reprocess:
            final = load_cache(self.cache_paths['final_dataset'], "final dataset")
            if final is not None:
                print_dataset_stats(final, "Cached Final Dataset")
                return final
        
        print("🔄 Applying final filters...")
        
        original_length = len(dataset)
        final_dataset = dataset.filter(length_filter, input_columns=["input_ids"])
        final_length = len(final_dataset)
        
        print(f"   Final filtering: {original_length} → {final_length} samples")
        print(f"   Kept: {final_length/original_length*100:.1f}%")
        
        # Save to cache
        save_cache(final_dataset, self.cache_paths['final_dataset'], "final dataset")
        print_dataset_stats(final_dataset, "Final Dataset")
        
        return final_dataset
    
    def create_train_val_split(self, dataset, test_size: float = 0.15, seed: int = 42):
        """Create train/validation split"""
        print(f"🔄 Creating train/validation split ({test_size*100:.0f}% validation)...")
        
        dataset_split = dataset.train_test_split(test_size=test_size, seed=seed)
        
        print(f"📊 Train samples: {len(dataset_split['train'])}")
        print(f"📊 Validation samples: {len(dataset_split['test'])}")
        
        return dataset_split
    
    def load_and_process_datasets(self):
        """Main method to load and process all datasets"""
        print("🚀 Starting dataset loading and processing pipeline...")
        
        # Step 1: Load raw datasets
        dataset = self.load_multiple_italian_datasets()
        
        # Step 2: Text preprocessing
        dataset = self.preprocess_text(dataset)
        
        # Step 3: Quality filtering
        dataset = self.apply_quality_filters(dataset)
        
        # Step 4: Audio processing and embeddings
        dataset = self.process_audio_and_embeddings(dataset)
        
        # Step 5: Final filtering
        dataset = self.apply_final_filters(dataset)
        
        # Step 6: Train/validation split
        dataset_split = self.create_train_val_split(dataset)
        
        print("✅ Dataset processing pipeline completed!")
        return dataset_split


def load_speaker_model(device: str = "auto"):
    """Load the speaker embedding model"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔄 Loading speaker model on {device}...")
    
    try:
        speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": device},
            savedir="/tmp/speaker_model"
        )
        print("✅ Speaker model loaded successfully")
        return speaker_model
    except Exception as e:
        print(f"❌ Failed to load speaker model: {e}")
        raise


def get_dataset_info(dataset):
    """Get comprehensive dataset information"""
    info = {
        "total_samples": len(dataset),
        "columns": dataset.column_names,
    }
    
    if "text" in dataset.column_names:
        sample_texts = dataset.select(range(min(1000, len(dataset))))["text"]
        text_lengths = [len(text) for text in sample_texts if text]
        if text_lengths:
            info["avg_text_length"] = sum(text_lengths) / len(text_lengths)
            info["min_text_length"] = min(text_lengths)
            info["max_text_length"] = max(text_lengths)
    
    if "input_ids" in dataset.column_names:
        sample_ids = dataset.select(range(min(1000, len(dataset))))["input_ids"]
        id_lengths = [len(ids) for ids in sample_ids if ids]
        if id_lengths:
            info["avg_token_length"] = sum(id_lengths) / len(id_lengths)
            info["min_token_length"] = min(id_lengths)
            info["max_token_length"] = max(id_lengths)
    
    return info


def print_comprehensive_dataset_stats(dataset, name="Dataset"):
    """Print comprehensive dataset statistics"""
    info = get_dataset_info(dataset)
    
    print(f"\n📊 {name} Statistics:")
    print("-" * 40)
    print(f"Total samples: {info['total_samples']}")
    print(f"Columns: {', '.join(info['columns'])}")
    
    if "avg_text_length" in info:
        print(f"Text length: {info['avg_text_length']:.1f} chars (avg)")
        print(f"Text range: {info['min_text_length']}-{info['max_text_length']} chars")
    
    if "avg_token_length" in info:
        print(f"Token length: {info['avg_token_length']:.1f} tokens (avg)")
        print(f"Token range: {info['min_token_length']}-{info['max_token_length']} tokens")
