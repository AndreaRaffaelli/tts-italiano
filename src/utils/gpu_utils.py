"""
GPU and Hardware Utilities
"""

import torch
from typing import Tuple


def check_gpu() -> bool:
    """Check GPU availability and memory"""
    print("\n🔍 Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"🏷️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🏆 CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("❌ CUDA is not available - training will be slow!")
        return False


def get_optimal_batch_size(has_gpu: bool, override_batch_size: int = None) -> Tuple[int, int]:
    """
    Determine optimal batch size and gradient accumulation steps
    
    Args:
        has_gpu: Whether GPU is available
        override_batch_size: Manual override for batch size
    
    Returns:
        Tuple of (batch_size, gradient_accumulation_steps)
    """
    if override_batch_size:
        return override_batch_size, max(1, 32 // override_batch_size)
    
    if has_gpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory_gb < 8:
            batch_size = 1
            grad_acc_steps = 32
            print("⚠️ Low GPU memory detected - using smaller batch sizes")
        elif gpu_memory_gb < 16:
            batch_size = 2
            grad_acc_steps = 16
        elif gpu_memory_gb < 24:
            batch_size = 4
            grad_acc_steps = 8
        else:
            batch_size = 8
            grad_acc_steps = 4
    else:
        batch_size = 1
        grad_acc_steps = 32
    
    return batch_size, grad_acc_steps


def get_device_info() -> dict:
    """Get detailed device information"""
    info = {
        'has_cuda': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if info['has_cuda']:
        info['device_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['memory_total'] = props.total_memory
        info['memory_available'] = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        info['compute_capability'] = f"{props.major}.{props.minor}"
        info['multiprocessor_count'] = props.multi_processor_count
    
    return info


def print_device_info():
    """Print detailed device information"""
    info = get_device_info()
    
    print("\n🔧 Device Information:")
    print("-" * 30)
    
    if info['has_cuda']:
        print(f"GPU: {info['device_name']}")
        print(f"Memory: {info['memory_total'] / 1e9:.1f} GB total")
        print(f"Available: {info['memory_available'] / 1e9:.1f} GB")
        print(f"Compute Capability: {info['compute_capability']}")
        print(f"Multiprocessors: {info['multiprocessor_count']}")
    else:
        print("CPU training only")
        import psutil
        print(f"CPU cores: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")


def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cache cleared")


def get_memory_usage() -> dict:
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return {"allocated": 0, "cached": 0, "total": 0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,  # GB
        "cached": torch.cuda.memory_reserved() / 1e9,      # GB  
        "total": torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    }


def print_memory_usage():
    """Print current GPU memory usage"""
    usage = get_memory_usage()
    if usage["total"] > 0:
        print(f"📊 GPU Memory - Allocated: {usage['allocated']:.1f}GB, "
              f"Cached: {usage['cached']:.1f}GB, Total: {usage['total']:.1f}GB")
