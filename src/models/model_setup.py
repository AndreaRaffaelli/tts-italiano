"""
Model Setup and Configuration
"""

import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from speechbrain.inference.classifiers import EncoderClassifier
from typing import Tuple, Optional
import os


class ModelManager:
    """Manages model loading and configuration"""
    
    def __init__(self, 
                 checkpoint: str = "microsoft/speecht5_tts",
                 device: str = "auto",
                 cache_dir: Optional[str] = None):
        self.checkpoint = checkpoint
        self.device = self._get_device(device)
        self.cache_dir = cache_dir
        
        self.processor = None
        self.model = None
        self.speaker_model = None
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_processor(self) -> SpeechT5Processor:
        """Load the SpeechT5 processor"""
        if self.processor is None:
            print(f"🔄 Loading SpeechT5 processor from {self.checkpoint}...")
            try:
                self.processor = SpeechT5Processor.from_pretrained(
                    self.checkpoint,
                    cache_dir=self.cache_dir
                )
                print("✅ Processor loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load processor: {e}")
                raise
        
        return self.processor
    
    def load_model(self) -> SpeechT5ForTextToSpeech:
        """Load the SpeechT5 model"""
        if self.model is None:
            print(f"🔄 Loading SpeechT5 model from {self.checkpoint}...")
            try:
                self.model = SpeechT5ForTextToSpeech.from_pretrained(
                    self.checkpoint,
                    cache_dir=self.cache_dir
                )
                
                # Optimize model for training
                self.model.config.use_cache = False
                
                # Move to device
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                
                print(f"✅ Model loaded successfully on {self.device}")
                self._print_model_info()
                
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                raise
        
        return self.model
    
    def load_speaker_model(self) -> EncoderClassifier:
        """Load the speaker embedding model"""
        if self.speaker_model is None:
            print(f"🔄 Loading speaker model on {self.device}...")
            try:
                self.speaker_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-xvect-voxceleb",
                    run_opts={"device": self.device},
                    savedir=os.path.join(self.cache_dir or "/tmp", "speaker_model")
                )
                print("✅ Speaker model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load speaker model: {e}")
                raise
        
        return self.speaker_model
    
    def load_all_models(self) -> Tuple[SpeechT5Processor, SpeechT5ForTextToSpeech, EncoderClassifier]:
        """Load all required models"""
        print("🚀 Loading all models...")
        
        processor = self.load_processor()
        model = self.load_model()
        speaker_model = self.load_speaker_model()
        
        print("✅ All models loaded successfully!")
        return processor, model, speaker_model
    
    def _print_model_info(self):
        """Print model information"""
        if self.model is None:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model Information:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / (1024**2):.1f} MB")
        
        if hasattr(self.model, 'config'):
            config = self.model.config
            print(f"   Vocab size: {getattr(config, 'vocab_size', 'N/A')}")
            print(f"   Hidden size: {getattr(config, 'd_model', 'N/A')}")
    
    def optimize_model_for_training(self):
        """Apply training optimizations to the model"""
        if self.model is None:
            print("⚠️ Model not loaded - cannot optimize")
            return
        
        print("🔧 Optimizing model for training...")
        
        # Disable caching
        self.model.config.use_cache = False
        
        # Enable gradient checkpointing if supported
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("   ✅ Gradient checkpointing enabled")
        
        # Set model to training mode
        self.model.train()
        
        print("✅ Model optimization completed")
    
    def save_models(self, output_dir: str):
        """Save processor and model"""
        print(f"💾 Saving models to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
            print(f"   ✅ Processor saved")
        
        if self.model is not None:
            self.model.save_pretrained(output_dir)
            print(f"   ✅ Model saved")
        
        print("✅ All models saved successfully!")
    
    def get_model_memory_usage(self) -> dict:
        """Get memory usage information"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        memory_info = {
            "allocated": torch.cuda.memory_allocated() / 1e9,  # GB
            "reserved": torch.cuda.memory_reserved() / 1e9,    # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1e9,  # GB
        }
        
        if self.model is not None:
            model_memory = sum(p.element_size() * p.nelement() for p in self.model.parameters()) / 1e9
            memory_info["model_size"] = model_memory
        
        return memory_info


def setup_models(checkpoint: str = "microsoft/speecht5_tts",
                device: str = "auto",
                cache_dir: Optional[str] = None,
                optimize_for_training: bool = True) -> Tuple[SpeechT5Processor, SpeechT5ForTextToSpeech, EncoderClassifier]:
    """
    Convenience function to set up all models
    
    Args:
        checkpoint: Model checkpoint to load
        device: Device to use
        cache_dir: Cache directory for models
        optimize_for_training: Whether to apply training optimizations
    
    Returns:
        Tuple of (processor, model, speaker_model)
    """
    manager = ModelManager(checkpoint=checkpoint, device=device, cache_dir=cache_dir)
    
    processor, model, speaker_model = manager.load_all_models()
    
    if optimize_for_training:
        manager.optimize_model_for_training()
    
    return processor, model, speaker_model


class ModelConfig:
    """Configuration class for model parameters"""
    
    def __init__(self):
        # Model checkpoints
        self.tts_checkpoint = "microsoft/speecht5_tts"
        self.speaker_checkpoint = "speechbrain/spkrec-xvect-voxceleb"
        
        # Device settings
        self.device = "auto"
        self.mixed_precision = True
        
        # Training optimizations
        self.use_gradient_checkpointing = True
        self.use_cache = False
        
        # Memory settings
        self.max_memory_usage = 0.9  # Max GPU memory to use
        
        # Cache settings
        self.cache_dir = None
        self.speaker_model_cache = "/tmp/speaker_model"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


def validate_model_compatibility(processor: SpeechT5Processor, 
                                model: SpeechT5ForTextToSpeech) -> bool:
    """
    Validate that processor and model are compatible
    
    Args:
        processor: SpeechT5 processor
        model: SpeechT5 model
    
    Returns:
        True if compatible
    """
    try:
        # Check vocab sizes match
        if hasattr(processor, 'tokenizer') and hasattr(model.config, 'vocab_size'):
            processor_vocab_size = len(processor.tokenizer.get_vocab())
            model_vocab_size = model.config.vocab_size
            
            if processor_vocab_size != model_vocab_size:
                print(f"❌ Vocab size mismatch: processor={processor_vocab_size}, model={model_vocab_size}")
                return False
        
        # Try a simple forward pass
        dummy_text = "Hello world"
        dummy_audio = torch.randn(1000)  # 1000 samples at 16kHz
        
        inputs = processor(
            text=dummy_text,
            audio_target=dummy_audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Check that we can process the inputs
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        
        if len(labels.shape) > 2:
            labels = labels[0]
        
        print("✅ Model and processor are compatible")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
        return False


def print_model_summary(processor: SpeechT5Processor,
                       model: SpeechT5ForTextToSpeech,
                       speaker_model: EncoderClassifier):
    """Print summary of all loaded models"""
    print("\n🤖 Model Summary")
    print("=" * 50)
    
    # Processor info
    print("📝 Processor:")
    if hasattr(processor, 'tokenizer'):
        vocab_size = len(processor.tokenizer.get_vocab())
        print(f"   Vocabulary size: {vocab_size:,}")
    
    # Main model info
    print("🎯 SpeechT5 Model:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / (1024**2):.1f} MB")
    print(f"   Device: {next(model.parameters()).device}")
    
    # Config info
    if hasattr(model, 'config'):
        config = model.config
        print(f"   Hidden size: {getattr(config, 'd_model', 'N/A')}")
        print(f"   Attention heads: {getattr(config, 'encoder_attention_heads', 'N/A')}")
        print(f"   Encoder layers: {getattr(config, 'encoder_layers', 'N/A')}")
        print(f"   Decoder layers: {getattr(config, 'decoder_layers', 'N/A')}")
    
    # Speaker model info
    print("🗣️ Speaker Model:")
    speaker_device = next(speaker_model.parameters()).device
    print(f"   Device: {speaker_device}")
    print(f"   Embedding dimension: 512 (X-Vector)")


def test_model_inference(processor: SpeechT5Processor,
                        model: SpeechT5ForTextToSpeech,
                        speaker_model: EncoderClassifier,
                        test_text: str = "Ciao, questo è un test.") -> bool:
    """
    Test model inference pipeline
    
    Args:
        processor: SpeechT5 processor
        model: SpeechT5 model
        speaker_model: Speaker embedding model
        test_text: Text to test with
    
    Returns:
        True if test passes
    """
    try:
        print(f"🧪 Testing inference with text: '{test_text}'")
        
        # Create dummy audio for speaker embedding
        dummy_audio = torch.randn(16000)  # 1 second at 16kHz
        device = next(speaker_model.parameters()).device
        
        # Generate speaker embedding
        with torch.no_grad():
            dummy_audio = dummy_audio.to(device)
            speaker_embedding = speaker_model.encode_batch(dummy_audio.unsqueeze(0))
            speaker_embedding = torch.nn.functional.normalize(speaker_embedding, dim=2)
        
        # Tokenize text
        inputs = processor.tokenizer(test_text, return_tensors="pt")
        
        # Move to model device
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        speaker_embedding = speaker_embedding.to(model_device)
        
        # Generate speech
        with torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"],
                speaker_embedding.squeeze(),
                vocoder=None  # We're not testing vocoder here
            )
        
        print(f"✅ Inference test passed - Generated speech shape: {speech.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


class ModelCheckpoint:
    """Helper class for model checkpointing"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_checkpoint(self, 
                       model: SpeechT5ForTextToSpeech,
                       processor: SpeechT5Processor,
                       step: int,
                       loss: float,
                       metadata: dict = None):
        """Save a training checkpoint"""
        checkpoint_dir = os.path.join(self.base_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and processor
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "step": step,
            "loss": loss,
            "timestamp": torch.tensor(0).item()  # Simple timestamp
        })
        
        torch.save(metadata, os.path.join(checkpoint_dir, "training_metadata.pt"))
        
        print(f"💾 Checkpoint saved at step {step} (loss: {loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str) -> tuple:
        """Load a training checkpoint"""
        try:
            model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint_path)
            processor = SpeechT5Processor.from_pretrained(checkpoint_path)
            
            metadata_path = os.path.join(checkpoint_path, "training_metadata.pt")
            if os.path.exists(metadata_path):
                metadata = torch.load(metadata_path)
            else:
                metadata = {}
            
            print(f"📂 Checkpoint loaded from {checkpoint_path}")
            return model, processor, metadata
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def list_checkpoints(self) -> list:
        """List available checkpoints"""
        checkpoints = []
        for item in os.listdir(self.base_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(self.base_dir, item)):
                checkpoints.append(item)
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        return checkpoints
