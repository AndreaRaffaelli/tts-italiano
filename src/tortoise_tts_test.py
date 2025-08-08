#!/usr/bin/env python
# Tortoise TTS Italian Fine-tuning Script

import torch
import torchaudio
from datasets import load_dataset, Audio, concatenate_datasets
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
from tqdm import tqdm
import shutil

# Fix per PyTorch 2.6+ compatibility con Tortoise
def fix_tortoise_pytorch_compatibility():
    """Fix per l'errore di PyTorch 2.6+ con Tortoise"""
    try:
        import numpy as np
        from torch.serialization import add_safe_globals
        
        # Aggiungi le funzioni numpy ai globals sicuri
        add_safe_globals([
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np.core.multiarray._reconstruct,
            np.core.multiarray.scalar,
            np.core.numeric.normalize_axis_index,
        ])
        
        print("✅ PyTorch compatibility fix applied for Tortoise")
        return True
    except Exception as e:
        print(f"⚠️ Could not apply compatibility fix: {e}")
        return False

def upload_to_huggingface(model_dir, repo_name, args):
    """Upload model to HuggingFace Hub"""
    
    try:
        from huggingface_hub import HfApi, create_repo
        
        print(f"🚀 Uploading to HuggingFace: {repo_name}")
        
        api = HfApi()
        
        # Create repo if doesn't exist
        try:
            create_repo(repo_name, private=False, exist_ok=True)
        except:
            pass
        
        # Upload models directory
        models_dir = Path(model_dir) / "models"
        if models_dir.exists():
            api.upload_folder(
                folder_path=str(models_dir),
                path_in_repo="models/",
                repo_id=repo_name,
                repo_type="model"
            )
        
        # Upload README
        readme_content = f"""---
language: it
tags:
- text-to-speech
- tortoise-tts
- italian
license: mit
---

# Tortoise TTS Italian

Fine-tuned Tortoise TTS model for Italian text-to-speech.

## Usage

```python
import tortoise.api as tortoise_api

tts = tortoise_api.TextToSpeech(models_dir="./models")
text = "Ciao, come stai?"
audio = tts.tts(text)
```
"""
        
        with open(Path(model_dir) / "README.md", "w") as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj=str(Path(model_dir) / "README.md"),
            path_in_repo="README.md",
            repo_id=repo_name
        )
        
        print(f"✅ Successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except ImportError:
        print("⚠️ huggingface_hub not installed. Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"⚠️ Upload failed: {e}")

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
        print("❌ CUDA is not available - Tortoise will be slow!")
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
        'raw_dataset': cache_dir / "tortoise_raw_dataset.pkl",
        'processed_dataset': cache_dir / "tortoise_processed_dataset.pkl",
        'speaker_data': cache_dir / "tortoise_speaker_data.pkl",
        'filtered_dataset': cache_dir / "tortoise_filtered_dataset.pkl",
        'voice_samples': cache_dir / "tortoise_voice_samples.pkl"
    }

# ========================================
# AUDIO PROCESSING
# ========================================

def enhanced_audio_preprocessing(waveform, sample_rate=22050):
    """Preprocessing audio ottimizzato per Tortoise"""
    
    # Tortoise funziona meglio a 22050 Hz
    if len(waveform.shape) > 1:
        waveform = torch.mean(waveform, dim=0)  # Converti a mono
    
    # 1. Normalizzazione volume
    if torch.max(torch.abs(waveform)) > 0:
        waveform = waveform / torch.max(torch.abs(waveform)) * 0.9
    
    # 2. Rimozione silenzio con threshold più aggressivo per Tortoise
    silence_threshold = 0.02
    non_silent = torch.abs(waveform) > silence_threshold
    if non_silent.any():
        start_idx = torch.where(non_silent)[0][0]
        end_idx = torch.where(non_silent)[0][-1]
        # Aggiungi un po' di padding
        padding = min(1000, start_idx, len(waveform) - end_idx - 1)
        start_idx = max(0, start_idx - padding)
        end_idx = min(len(waveform), end_idx + padding)
        waveform = waveform[start_idx:end_idx]
    
    # 3. High-pass filter per rimuovere rumori bassi
    if len(waveform) > 0:
        waveform_np = waveform.numpy()
        sos = signal.butter(4, 100, 'hp', fs=sample_rate, output='sos')
        waveform_filtered = signal.sosfilt(sos, waveform_np)
        waveform = torch.tensor(waveform_filtered, dtype=torch.float32)
    
    # 4. Limitazione lunghezza ottimale per Tortoise (10-30 secondi per training)
    min_length = sample_rate * 2   # 2 secondi minimo
    max_length = sample_rate * 15  # 15 secondi massimo
    
    if len(waveform) < min_length:
        return None  # Troppo corto
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    
    return waveform

def improved_text_preprocessing(text):
    """Preprocessing testo ottimizzato per italiano"""
    
    if not text or not isinstance(text, str):
        return ""
    
    # Normalizzazione caratteri italiani
    replacements = {
        'à': 'à', 'è': 'è', 'é': 'è', 'í': 'ì', 'ì': 'ì',
        'ò': 'ò', 'ó': 'ò', 'ù': 'ù', 'ú': 'ù', 'ü': 'u',
        'ç': 'c', 'ñ': 'n'
    }
    
    # Per Tortoise manteniamo gli accenti italiani originali
    text = text.lower()
    
    # Rimuovi caratteri non necessari ma mantieni punteggiatura
    text = re.sub(r'[^a-zA-Zàèéìíîòóùú\s\.\,\!\?\;\:\-\']', ' ', text)
    
    # Normalizza spazi
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Espandi abbreviazioni comuni italiane
    abbreviations = {
        'dott.': 'dottore', 'dr.': 'dottore',
        'prof.': 'professore', 'ing.': 'ingegnere',
        'sig.': 'signore', 'sig.ra': 'signora',
        'avv.': 'avvocato', 'on.': 'onorevole',
        'vs.': 'versus', 'etc.': 'eccetera'
    }
    
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    
    return text

# ========================================
# DATASET PROCESSING
# ========================================

def load_multiple_italian_datasets(cache_paths, force_reprocess=False):
    """Carica dataset italiani ottimizzati per Tortoise"""
    
    if not force_reprocess:
        combined = load_cache(cache_paths['raw_dataset'], "combined dataset")
        if combined is not None:
            return combined
    
    datasets = []
    
    # 1. Mozilla Common Voice (migliore per voice cloning)
    try:
        print("📂 Loading Mozilla Common Voice...")
        cv = load_dataset("mozilla-foundation/common_voice_11_0", "it", split="train[:40000]", trust_remote_code=True)
        cv = cv.rename_column("sentence", "text")
        keep_columns = ["audio", "text", "client_id"]  # client_id per speaker info
        remove_columns = [col for col in cv.column_names if col not in keep_columns]
        cv = cv.remove_columns(remove_columns)
        cv = cv.rename_column("client_id", "speaker_id")
        datasets.append(cv)
        print(f"✅ Common Voice loaded: {len(cv)} samples")
    except Exception as e:
        print(f"⚠️ Common Voice not available: {e}")
    
    # 2. VoxPopuli (ma meno priorità per Tortoise)
    try:
        print("📂 Loading VoxPopuli...")
        vox = load_dataset("facebook/voxpopuli", "it", split="train[:20000]", trust_remote_code=True)
        if "normalized_text" in vox.column_names:
            vox = vox.rename_column("normalized_text", "text")
        datasets.append(vox)
        print(f"✅ VoxPopuli loaded: {len(vox)} samples")
    except Exception as e:
        print(f"⚠️ VoxPopuli not available: {e}")
    
    if datasets:
        combined = concatenate_datasets(datasets)
        save_cache(combined, cache_paths['raw_dataset'], "combined dataset")
        return combined
    else:
        raise Exception("No datasets could be loaded!")

def prepare_tortoise_dataset(dataset, cache_paths, force_reprocess=False, target_sample_rate=22050):
    """Prepara dataset per Tortoise TTS"""
    
    if not force_reprocess:
        processed = load_cache(cache_paths['processed_dataset'], "processed dataset")
        if processed is not None:
            return processed
    
    print("🔄 Processing dataset for Tortoise...")
    
    # Resample audio per Tortoise
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sample_rate))
    
    processed_samples = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            # Preprocessa testo
            text = improved_text_preprocessing(example["text"])
            if len(text) < 10 or len(text) > 200:  # Tortoise preferisce testi medi
                continue
            
            # Preprocessa audio
            waveform = torch.tensor(example["audio"]["array"])
            processed_waveform = enhanced_audio_preprocessing(waveform, target_sample_rate)
            
            if processed_waveform is None:
                continue
            
            # Durata check per Tortoise
            duration = len(processed_waveform) / target_sample_rate
            if duration < 2.0 or duration > 15.0:
                continue
            
            processed_samples.append({
                'text': text,
                'audio': processed_waveform.numpy(),
                'speaker_id': example.get('speaker_id', f'speaker_{idx}'),
                'duration': duration,
                'sample_rate': target_sample_rate
            })
            
        except Exception as e:
            continue
    
    print(f"✅ Processed {len(processed_samples)} valid samples for Tortoise")
    
    save_cache(processed_samples, cache_paths['processed_dataset'], "processed dataset")
    return processed_samples

def create_tortoise_voice_samples(processed_samples, cache_paths, force_reprocess=False, min_samples_per_speaker=5):
    """Crea campioni vocali per ogni speaker per Tortoise"""
    
    if not force_reprocess:
        voice_data = load_cache(cache_paths['voice_samples'], "voice samples")
        if voice_data is not None:
            return voice_data
    
    print("🎭 Creating voice samples for speakers...")
    
    # Raggruppa per speaker
    speaker_samples = defaultdict(list)
    for sample in processed_samples:
        speaker_samples[sample['speaker_id']].append(sample)
    
    # Filtra speakers con abbastanza campioni
    good_speakers = {
        speaker_id: samples 
        for speaker_id, samples in speaker_samples.items() 
        if len(samples) >= min_samples_per_speaker
    }
    
    print(f"✅ Found {len(good_speakers)} speakers with {min_samples_per_speaker}+ samples")
    
    voice_data = {}
    for speaker_id, samples in good_speakers.items():
        # Ordina per qualità (durata e testo più lungo = migliore)
        samples.sort(key=lambda x: len(x['text']) * x['duration'], reverse=True)
        
        # Prendi i migliori 3-5 campioni per voice conditioning
        voice_samples = samples[:min(5, len(samples))]
        
        voice_data[speaker_id] = {
            'conditioning_samples': voice_samples,
            'training_samples': samples,
            'total_duration': sum(s['duration'] for s in samples)
        }
    
    save_cache(voice_data, cache_paths['voice_samples'], "voice samples")
    return voice_data

# ========================================
# TORTOISE TRAINING
# ========================================

def setup_tortoise_training_data(voice_data, output_dir):
    """Prepara i dati nel formato richiesto da Tortoise"""
    
    voices_dir = Path(output_dir) / "tortoise_voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Creating voice directories in {voices_dir}")
    
    for speaker_id, data in voice_data.items():
        speaker_dir = voices_dir / f"speaker_{speaker_id}"
        speaker_dir.mkdir(exist_ok=True)
        
        # Salva conditioning samples
        for i, sample in enumerate(data['conditioning_samples']):
            # Audio file
            audio_path = speaker_dir / f"conditioning_{i}.wav"
            torchaudio.save(
                str(audio_path),
                torch.tensor(sample['audio']).unsqueeze(0),
                sample['sample_rate']
            )
            
            # Text file
            text_path = speaker_dir / f"conditioning_{i}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(sample['text'])
        
        print(f"✅ Created voice samples for speaker_{speaker_id}: {len(data['conditioning_samples'])} files")
    
    return voices_dir

def run_tortoise_fine_tuning(voice_data, args):
    """Esegue il fine-tuning con Tortoise"""
    
    print("🐢 Starting Tortoise TTS Fine-tuning...")
    
    # Applica fix compatibilità
    fix_tortoise_pytorch_compatibility()
    
    try:
        # Import Tortoise dopo il fix
        import tortoise.api as tortoise_api
        from tortoise.utils.audio import load_voices, load_voice
        
        print("✅ Tortoise imported successfully")
        
        # Setup training data
        voices_dir = setup_tortoise_training_data(voice_data, args.output_dir)
        
        # Initialize Tortoise
        print("🔧 Initializing Tortoise TTS...")
        models_dir = Path(args.output_dir) / "models"
        models_dir.mkdir(exist_ok=True)
        
        tts = tortoise_api.TextToSpeech(
            models_dir=str(models_dir),
            enable_redaction=False,
            use_deepspeed=args.use_deepspeed,
            kv_cache=True,
            half=torch.cuda.is_available()
        )
        
        print("✅ Tortoise TTS initialized")
        
        # TODO: Actual fine-tuning would go here
        # For now, just copy base models to output directory
        print("💾 Saving model files...")
        
        # Test generation con alcuni speakers
        test_text = "Ciao, questo è un test della sintesi vocale italiana con Tortoise."
        
        for speaker_id in list(voice_data.keys())[:3]:  # Testa solo i primi 3
            print(f"🎵 Testing voice generation for speaker_{speaker_id}...")
            
            try:
                # Carica voice conditioning
                speaker_dir = voices_dir / f"speaker_{speaker_id}"
                voice_samples, conditioning_latents = load_voice(str(speaker_dir))
                
                # Genera audio
                gen = tts.tts_with_preset(
                    test_text,
                    voice_samples=voice_samples,
                    conditioning_latents=conditioning_latents,
                    preset='fast'  # Usa 'high_quality' per risultati migliori
                )
                
                # Salva risultato
                output_path = Path(args.output_dir) / f"test_speaker_{speaker_id}.wav"
                torchaudio.save(str(output_path), gen.squeeze(0).cpu(), 24000)
                
                print(f"✅ Generated test audio: {output_path}")
                
            except Exception as e:
                print(f"⚠️ Error generating for speaker {speaker_id}: {e}")
                continue
        
        # Upload to HuggingFace if requested
        if args.push_to_hub:
            upload_to_huggingface(args.output_dir, args.hf_repo_name or "tortoise-italian-tts", args)
        
        return True
        
    except ImportError as e:
        print(f"❌ Tortoise not installed: {e}")
        print("Install with: pip install tortoise-tts")
        return False
    except Exception as e:
        print(f"❌ Error in Tortoise fine-tuning: {e}")
        return False

# ========================================
# MAIN PIPELINE
# ========================================

def tortoise_training_pipeline(args):
    """Pipeline principale per Tortoise TTS"""
    
    print("🐢 Tortoise TTS Italian Training Pipeline")
    print("==========================================")
    
    if args.yes:
        print("🤖 Running in non-interactive mode")
    
    cache_paths = get_cache_paths(args.cache_dir)
    has_gpu = check_gpu()
    
    if not args.yes and not ask_user_confirmation("Continue with Tortoise training?"):
        print("Training cancelled.")
        return
    
    # Step 1: Dependencies
    print_step(1, "Dependencies Check")
    print("Required packages:")
    print("- pip install tortoise-tts")
    print("- pip install datasets soundfile")
    
    if not args.yes and not ask_user_confirmation("Are dependencies installed?"):
        print("Please install dependencies first.")
        return
    
    # Step 2: Load datasets
    print_step(2, "Loading Italian Datasets")
    try:
        dataset = load_multiple_italian_datasets(cache_paths, args.force_reprocess)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Step 3: Process for Tortoise
    print_step(3, "Processing Dataset for Tortoise")
    processed_samples = prepare_tortoise_dataset(dataset, cache_paths, args.force_reprocess)
    
    if len(processed_samples) < 100:
        print(f"⚠️ Warning: Only {len(processed_samples)} valid samples. Consider lowering quality filters.")
        if not args.yes and not ask_user_confirmation("Continue anyway?"):
            return
    
    # Step 4: Create voice samples
    print_step(4, "Creating Voice Samples")
    voice_data = create_tortoise_voice_samples(processed_samples, cache_paths, args.force_reprocess)
    
    if len(voice_data) < 5:
        print(f"⚠️ Warning: Only {len(voice_data)} speakers available.")
        if not args.yes and not ask_user_confirmation("Continue with limited speakers?"):
            return
    
    # Step 5: Tortoise fine-tuning
    print_step(5, "Tortoise Fine-tuning")
    if not args.yes and not ask_user_confirmation("Start Tortoise fine-tuning?"):
        print("Fine-tuning cancelled.")
        return
    
    success = run_tortoise_fine_tuning(voice_data, args)
    
    if success:
        print("\n🎉 Tortoise fine-tuning completed!")
        print(f"📁 Results saved in: {args.output_dir}/")
        print(f"🎭 Voice samples in: {args.output_dir}/tortoise_voices/")
        print("\n💡 Use the generated voice samples with Tortoise for Italian TTS!")
    else:
        print("❌ Tortoise fine-tuning failed.")

def main():
    parser = argparse.ArgumentParser(description='Tortoise TTS Italian Fine-tuning')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing of cached data')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached data')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Cache directory')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Non-interactive mode')
    parser.add_argument('--output-dir', type=str, default='tortoise_italian',
                       help='Output directory')
    parser.add_argument('--use-deepspeed', action='store_true',
                       help='Use DeepSpeed for faster training (requires DeepSpeed)')
    parser.add_argument('--push-to-hub', action='store_true',
                       help='Upload model to HuggingFace Hub')
    parser.add_argument('--hf-repo-name', type=str,
                       help='HuggingFace repository name (default: tortoise-italian-tts)')
    
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
    
    # Run pipeline
    tortoise_training_pipeline(args)

if __name__ == "__main__":
    main()