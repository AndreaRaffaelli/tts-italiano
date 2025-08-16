"""
Audio and Text Preprocessing Functions
"""

import torch
import re
import numpy as np
from scipy import signal
from typing import Optional


def enhanced_audio_preprocessing(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Enhanced audio preprocessing for TTS training
    
    Args:
        waveform: Input audio waveform
        sample_rate: Sample rate of the audio
    
    Returns:
        Preprocessed waveform
    """
    # 1. Volume normalization
    if torch.max(torch.abs(waveform)) > 0:
        waveform = waveform / torch.max(torch.abs(waveform))
    
    # 2. More aggressive silence removal
    # Use lower threshold to preserve voiceless consonants
    silence_threshold = 0.01
    non_silent = torch.abs(waveform) > silence_threshold
    if non_silent.any():
        start_idx = torch.where(non_silent)[0][0]
        end_idx = torch.where(non_silent)[0][-1]
        waveform = waveform[start_idx:end_idx + 1]
    
    # 3. High-pass filter to remove low-frequency noise
    if len(waveform) > 0:
        waveform_np = waveform.numpy()
        sos = signal.butter(4, 80, 'hp', fs=sample_rate, output='sos')
        waveform_filtered = signal.sosfilt(sos, waveform_np)
        waveform = torch.tensor(waveform_filtered, dtype=torch.float32)
    
    # 4. Length limitation (important for stability)
    max_length = sample_rate * 10  # 10 seconds max
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    
    return waveform


def improved_text_preprocessing(text: str) -> str:
    """
    Enhanced text preprocessing for Italian
    
    Args:
        text: Input text string
    
    Returns:
        Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Italian character normalization
    replacements = {
        'à': 'a', 'è': 'e', 'é': 'e', 'í': 'i', 'ì': 'i',
        'ò': 'o', 'ó': 'o', 'ù': 'u', 'ú': 'u', 'ü': 'u',
        'ç': 'c', 'ñ': 'n'
    }
    
    # Apply substitutions (both lower and upper case)
    for src, dst in replacements.items():
        text = text.replace(src, dst)
        text = text.replace(src.upper(), dst.upper())
    
    # Remove special characters but keep Italian punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', ' ', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Handle common Italian abbreviations
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


def quality_filter(example: dict) -> bool:
    """
    Filter examples based on quality criteria
    
    Args:
        example: Dataset example with 'text' and 'audio' fields
    
    Returns:
        True if example passes quality filters
    """
    text = example.get("text", "")
    
    # Audio length check
    if "audio" in example and "array" in example["audio"]:
        audio_length = len(example["audio"]["array"]) / 16000
    else:
        return False
    
    # Text length filter
    if len(text) < 10 or len(text) > 300:
        return False
    
    # Audio duration filter
    if audio_length < 1.0 or audio_length > 10.0:
        return False
    
    # Non-empty text filter
    if not text.strip():
        return False
    
    # Special characters ratio filter
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) > 0 and special_chars / len(text) > 0.3:  # Max 30% special chars
        return False
    
    return True


def length_filter(input_ids: list, max_length: int = 300) -> bool:
    """
    Filter based on input sequence length
    
    Args:
        input_ids: Tokenized input sequence
        max_length: Maximum allowed length
    
    Returns:
        True if sequence is within length limit
    """
    return len(input_ids) < max_length


class TextPreprocessor:
    """Text preprocessing class with configurable options"""
    
    def __init__(self, 
                 normalize_accents: bool = True,
                 expand_abbreviations: bool = True,
                 max_special_char_ratio: float = 0.3):
        self.normalize_accents = normalize_accents
        self.expand_abbreviations = expand_abbreviations
        self.max_special_char_ratio = max_special_char_ratio
        
        self.replacements = {
            'à': 'a', 'è': 'e', 'é': 'e', 'í': 'i', 'ì': 'i',
            'ò': 'o', 'ó': 'o', 'ù': 'u', 'ú': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n'
        }
        
        self.abbreviations = {
            'dott.': 'dottor', 'dr.': 'dottor',
            'prof.': 'professor', 'ing.': 'ingegner',
            'sig.': 'signor', 'sig.ra': 'signora',
            'avv.': 'avvocato', 'on.': 'onorevole'
        }
    
    def preprocess(self, text: str) -> str:
        """Main preprocessing method"""
        if not text or not isinstance(text, str):
            return ""
        
        # Accent normalization
        if self.normalize_accents:
            for src, dst in self.replacements.items():
                text = text.replace(src, dst)
                text = text.replace(src.upper(), dst.upper())
        
        # Remove unwanted characters
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', ' ', text)
        
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Expand abbreviations
        if self.expand_abbreviations:
            for abbr, full in self.abbreviations.items():
                text = text.replace(abbr, full)
                text = text.replace(abbr.capitalize(), full.capitalize())
        
        return text
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text passes quality criteria"""
        if not text or len(text) < 10 or len(text) > 300:
            return False
        
        # Check special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / len(text) > self.max_special_char_ratio:
            return False
        
        return True


class AudioPreprocessor:
    """Audio preprocessing class with configurable options"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 max_duration: float = 10.0,
                 min_duration: float = 1.0,
                 silence_threshold: float = 0.01,
                 high_pass_freq: float = 80.0):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.silence_threshold = silence_threshold
        self.high_pass_freq = high_pass_freq
    
    def preprocess(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        """Main audio preprocessing method"""
        if waveform is None or len(waveform) == 0:
            return None
        
        # Volume normalization
        if torch.max(torch.abs(waveform)) > 0:
            waveform = waveform / torch.max(torch.abs(waveform))
        
        # Silence removal
        non_silent = torch.abs(waveform) > self.silence_threshold
        if non_silent.any():
            start_idx = torch.where(non_silent)[0][0]
            end_idx = torch.where(non_silent)[0][-1]
            waveform = waveform[start_idx:end_idx + 1]
        else:
            return None  # All silence
        
        # Duration check
        duration = len(waveform) / self.sample_rate
        if duration < self.min_duration or duration > self.max_duration:
            return None
        
        # High-pass filtering
        if self.high_pass_freq > 0:
            waveform_np = waveform.numpy()
            sos = signal.butter(4, self.high_pass_freq, 'hp', 
                              fs=self.sample_rate, output='sos')
            waveform_filtered = signal.sosfilt(sos, waveform_np)
            waveform = torch.tensor(waveform_filtered, dtype=torch.float32)
        
        # Length limitation
        max_length = int(self.sample_rate * self.max_duration)
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        
        return waveform
    
    def is_valid_audio(self, waveform: torch.Tensor) -> bool:
        """Check if audio passes quality criteria"""
        if waveform is None or len(waveform) == 0:
            return False
        
        duration = len(waveform) / self.sample_rate
        return self.min_duration <= duration <= self.max_duration
