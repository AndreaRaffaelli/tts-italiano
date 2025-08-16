"""
Cache Management Utilities
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def save_cache(data: Any, cache_path: Path, description: str = "data") -> bool:
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


def load_cache(cache_path: Path, description: str = "data") -> Optional[Any]:
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


def get_cache_paths(cache_dir: str = "cache") -> Dict[str, Path]:
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


def clear_cache(cache_paths: Dict[str, Path]) -> None:
    """Clear all cache files"""
    for cache_name, cache_path in cache_paths.items():
        if cache_path.exists():
            cache_path.unlink()
            print(f"   Deleted: {cache_path}")


def get_cache_size(cache_paths: Dict[str, Path]) -> Dict[str, int]:
    """Get size of cache files in bytes"""
    sizes = {}
    for cache_name, cache_path in cache_paths.items():
        if cache_path.exists():
            sizes[cache_name] = cache_path.stat().st_size
        else:
            sizes[cache_name] = 0
    return sizes


def print_cache_info(cache_paths: Dict[str, Path]) -> None:
    """Print information about cache files"""
    print("\n📂 Cache Information:")
    print("-" * 40)
    
    sizes = get_cache_size(cache_paths)
    total_size = sum(sizes.values())
    
    for cache_name, size in sizes.items():
        if size > 0:
            size_mb = size / (1024 * 1024)
            status = "✅ Exists"
            print(f"{cache_name:20} {status} ({size_mb:.1f} MB)")
        else:
            print(f"{cache_name:20} ❌ Missing")
    
    if total_size > 0:
        total_mb = total_size / (1024 * 1024)
        print(f"{'Total cache size':20} {total_mb:.1f} MB")
    else:
        print("No cache files found")
