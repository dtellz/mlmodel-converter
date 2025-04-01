# MLModel Converter Tool Roadmap

This document outlines the planned future development for the MLModel Converter Tool project.

## Short-term Goals

### 1. Unified CLI Interface

Create a unified command-line interface that offers both conversion and fine-tuning options:

```bash
python ml_toolkit.py --convert  # Original conversion functionality
python ml_toolkit.py --fine-tune  # LoRA fine-tuning functionality
```

This would provide a more consistent user experience and simplify the documentation.

### 2. Refactor Shared Code

Extract common functionality (like the CoreML conversion) into shared modules that both scripts can use:

- Create a `utils/` directory with shared modules
- Move CoreML conversion logic to `utils/conversion.py`
- Move device detection to `utils/device.py`
- Create a unified model loading function

### 3. Project Renaming

Consider renaming from "mlmodel-converter" to something like "mlmodel-toolkit" or "ios-ml-toolkit" to reflect the expanded scope that now includes both conversion and fine-tuning.

## Medium-term Goals

### 1. Support for More Model Types

Expand beyond GPT-style models to support:
- BERT and other encoder models
- Vision models (ViT, ResNet, etc.)
- Multimodal models

### 2. Enhanced Fine-tuning Options

Add support for:
- Custom datasets (beyond the current motivational quotes)
- More fine-tuning techniques beyond LoRA (QLoRA, adapter tuning)
- Hyperparameter optimization

### 3. Improved Testing and Validation

- Add unit tests for core functionality
- Create validation scripts to verify model performance before and after conversion
- Add benchmarking tools for iOS performance

## Long-term Goals

### 1. Web Interface

Develop a simple web UI for users who prefer graphical interfaces over command line:
- Model selection
- Training parameter configuration
- Progress visualization
- Model testing interface

### 2. iOS Demo App

Create a companion iOS app that demonstrates how to use the converted models:
- Text generation example
- Integration patterns and best practices
- Performance benchmarks on different devices

### 3. Model Registry

Implement a simple model registry to track:
- Model versions
- Training parameters
- Performance metrics
- Deployment history

## Contributing

If you're interested in contributing to any of these roadmap items, please open an issue to discuss your approach before submitting a pull request.
