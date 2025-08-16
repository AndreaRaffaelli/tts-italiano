#!/usr/bin/env python
"""
Setup script for Enhanced TTS Training
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="enhanced-tts-training",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Enhanced TTS Training Pipeline for Italian SpeechT5",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enhanced-tts-training",
    
    packages=find_packages(),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    
    python_requires=">=3.8",
    
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "logging": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=23.0.0", 
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "enhanced-tts-train=main:main",
        ],
    },
    
    include_package_data=True,
    
    project_urls={
        "Bug Reports": "https://github.com/yourusername/enhanced-tts-training/issues",
        "Source": "https://github.com/yourusername/enhanced-tts-training",
        "Documentation": "https://github.com/yourusername/enhanced-tts-training/wiki",
    },
    
    keywords=[
        "tts", "text-to-speech", "italian", "speecht5", 
        "transformers", "machine-learning", "deep-learning"
    ],
)
