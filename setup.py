"""Setup script for Meta-TTA-TSM package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="meta-tta-tsm",
    version="1.0.0",
    description=(
        "Meta-Topological Test-Time Adaptive Score Matching: "
        "Topology-conditioned score functions with meta-learning "
        "and test-time adaptation for missing data."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/meta-tta-tsm",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
