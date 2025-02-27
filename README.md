# DeepSeek R1 Knowledge Distillation

This repository contains an implementation of knowledge distillation techniques specifically designed for DeepSeek R1 models. Knowledge distillation allows us to transfer the capabilities of larger, more powerful "teacher" models to smaller, more efficient "student" models.

## Overview

DeepSeek R1 is a powerful language model architecture, but deploying the full-sized model in production environments can be resource-intensive. This implementation provides a framework for distilling knowledge from larger R1 variants to smaller ones while retaining as much performance as possible.

## Features

- Support for various distillation techniques:
  - Vanilla Knowledge Distillation (KD)
  - Response-based Distillation
  - Feature-based Distillation
  - Relation-based Distillation
  - Intermediate Layer Distillation
- Configurable distillation parameters
- Training examples for different model sizes
- Evaluation utilities

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- DeepSeek R1 model checkpoint access

## Installation

```bash
git clone https://github.com/cm2solutions/deepseek-r1-distillation.git
cd deepseek-r1-distillation
pip install -e .
```

## Quick Start

1. Configure your distillation settings in `config/distill_config.yaml`
2. Prepare your teacher and student models
3. Run the distillation process:

```bash
python train.py --config config/distill_config.yaml
```

## Configuration Options

The distillation process can be customized through the configuration file:

- `teacher_model`: Path or name of the teacher model
- `student_model`: Path or name of the student model
- `alpha`: Weight for the distillation loss
- `temperature`: Temperature parameter for softening probability distributions
- `distill_layers`: Which layers to use for intermediate distillation
- `distill_method`: Distillation technique to use

## Distillation Methods

### Vanilla Knowledge Distillation

The basic approach involves training the student model to mimic the output distribution of the teacher model.

### Feature-based Distillation

This method aligns the internal representations of the student with those of the teacher.

### Relation-based Distillation

This approach focuses on transferring the relationships between different examples or tokens.

### Intermediate Layer Distillation

This technique distills knowledge from the intermediate layers of the teacher model.

## License

MIT

## Citation

If you use this code for your research, please cite:

```
@misc{deepseek-r1-distillation,
  author = {Your Name},
  title = {DeepSeek R1 Knowledge Distillation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cm2solutions/deepseek-r1-distillation}}
}
```