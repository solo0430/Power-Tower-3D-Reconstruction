#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="power-tower-3d-reconstruction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="3D reconstruction of power towers from point cloud for EM simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Power-Tower-3D-Reconstruction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tower-reconstruct=src.cli:main",
        ],
    },
)
