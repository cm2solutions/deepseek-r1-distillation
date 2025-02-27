from setuptools import setup, find_packages

setup(
    name="deepseek-distill",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "pyyaml>=6.0",
        "numpy>=1.23.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",  # Optional for experiment tracking
        "accelerate>=0.20.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Knowledge distillation for DeepSeek R1 models",
    keywords="deepseek, nlp, distillation, transformers",
    url="https://github.com/cm2solutions/deepseek-r1-distillation",
)
