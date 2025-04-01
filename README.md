# MLModel Converter Tool

A simple Python tool for converting Hugging Face NLP models into Apple's Core ML format (`.mlmodel` or `.mlpackage`) for seamless integration into iOS/macOS apps, running inference on device.

<p align="center">
  <img src="lg.png" alt="MLModel Converter Tool" width="300">
</p>


---

## Overview

This converter allows you to quickly transform pre-trained Hugging Face models (such as GPT-2 for text generation) into Core ML models suitable for on-device inference on Apple devices.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management

## Setup

Clone and initialize the project:

```bash
git clone <repo_link>
cd mlmodel-converter
uv install
```

## Usage

### Converting a Hugging Face model

```bash
uv run python main.py
```

The CLI will prompt you to enter the following parameters:

- `model_name`: The Hugging Face model name to convert (e.g., 'sarahai/your-inspiration')
- `file_extension`: The target file extension to use for the output Core ML model (mlmodel or mlpackage)
- `output_filename`: The output Core ML model filename (without .mlmodel or .mlpackage extension)

### Fine-tuning with LoRA before conversion

The tool now supports fine-tuning models using Low-Rank Adaptation (LoRA) before converting them to CoreML format. This is particularly useful for customizing models for specific tasks while keeping them small enough for mobile deployment.

```bash
uv run python lora_fine_tune.py --model_id distilgpt2 --num_train_epochs 3 --batch_size 4
```

#### Fine-tuning parameters

- `--model_id`: The Hugging Face model ID to fine-tune (default: 'distilgpt2')
- `--output_dir`: Directory to save training results (default: './results')
- `--merged_model_dir`: Directory to save the merged model (default: './merged_model')
- `--coreml_model_path`: Path to save the CoreML model (default: './Motivator.mlmodel')
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--lora_r`: LoRA attention dimension (default: 8)
- `--lora_alpha`: LoRA alpha parameter (default: 16)
- `--lora_dropout`: LoRA dropout rate (default: 0.05)
- `--max_length`: Maximum sequence length (default: 64)

The script will:
1. Load the specified model
2. Fine-tune it on a dataset of motivational quotes using LoRA
3. Merge the LoRA weights with the base model
4. Convert the fine-tuned model to CoreML format
5. Test the converted model with a sample input

This feature is optimized for Apple Silicon (M-series chips) and produces models ready for iOS deployment.

### Output

This generates a .mlmodel or .mlpackage file in the project directory (e.g., `<output_filename>.mlmodel` or `<output_filename>.mlpackage`) ready for integration into Xcode.

### Integration into Xcode

Drag the generated .mlmodel file into your Xcode project’s file navigator. Xcode automatically generates a Swift class to interact with the model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

[MIT](https://choosealicense.com/licenses/mit/) @ [dtellz](https://github.com/dtellz)
 