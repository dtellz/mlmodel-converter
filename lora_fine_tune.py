from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset
import coremltools as ct
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA and convert to CoreML")
    parser.add_argument("--model_id", type=str, default="distilgpt2", help="Model ID to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--merged_model_dir", type=str, default="./merged_model", help="Directory to save merged model")
    parser.add_argument("--coreml_model_path", type=str, default="./Motivator.mlmodel", help="Path to save CoreML model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--use_fp16", action="store_true", help="Use mixed precision training if available")
    return parser.parse_args()

class QuoteDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()  # For causal language modeling
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def load_and_prepare_model(args):
    print(f"Loading model: {args.model_id}")
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Load model and move to device
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def load_dataset_and_prepare(tokenizer, args):
    print("Loading dataset...")
    dataset = load_dataset("Abirate/english_quotes")
    texts = dataset["train"]["quote"]
    
    # Filter out quotes that are too long to reduce training time and memory usage
    filtered_texts = [text for text in texts if len(text.split()) < args.max_length // 2]
    print(f"Dataset loaded: {len(filtered_texts)} quotes after filtering")
    
    # You can add your own custom quotes here
    # custom_quotes = ["Your custom quote 1", "Your custom quote 2"]
    # filtered_texts.extend(custom_quotes)
    
    print("Tokenizing dataset...")
    tokenized = tokenizer(
        filtered_texts, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=args.max_length
    )
    
    return QuoteDataset(tokenized)

def get_target_modules(model):
    """Inspect model architecture to find appropriate target modules for LoRA."""
    import re
    
    # Get all named modules
    named_modules = dict(model.named_modules())
    
    # Print some module names for debugging
    print("Available modules (sample):")
    sample_modules = list(named_modules.keys())[:10]  # Show first 10 modules
    for name in sample_modules:
        print(f"  - {name}")
    
    # For distilgpt2, the attention modules are typically in the transformer blocks
    # Let's find modules that might be attention-related
    attention_pattern = re.compile(r'.*\.(k|q|v|out|attention|attn).*')
    attention_modules = [name for name in named_modules.keys() if attention_pattern.match(name)]
    
    print("\nPotential attention modules:")
    for name in attention_modules[:10]:  # Show first 10 attention modules
        print(f"  - {name}")
    
    # For distilgpt2, we'll target specific linear layers in the attention mechanism
    # Based on the architecture, we'll determine the appropriate target modules
    if any("transformer.h" in name for name in named_modules.keys()):
        # This is likely a GPT-2 style model (including distilgpt2)
        return ["attn.c_attn"]  # This is the common attention projection in GPT-2 models
    
    # Fallback to a common pattern
    return ["query", "value"]

def apply_lora_and_train(model, train_dataset, args, device):
    print("Applying LoRA configuration...")
    
    # Get the correct target modules for the model
    target_modules = get_target_modules(model)
    print(f"Using target modules: {target_modules}")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Set up training arguments based on device
    fp16 = args.use_fp16 and torch.cuda.is_available()  # Only use fp16 with CUDA
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=fp16,  # Only use mixed precision on CUDA
        gradient_accumulation_steps=4,  # Helps with memory usage
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    return model

def merge_lora_weights(model, args):
    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()
    
    # Create directory if it doesn't exist
    os.makedirs(args.merged_model_dir, exist_ok=True)
    
    # Save the merged model
    merged_model.save_pretrained(args.merged_model_dir)
    return merged_model

def convert_to_coreml(model, tokenizer, args, device):
    print("Converting to CoreML format...")
    
    # Explicitly move model to CPU for CoreML conversion
    model = model.to("cpu")
    
    # Create a wrapped model similar to main.py
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids):
            outputs = self.model(input_ids=input_ids, return_dict=False)
            return outputs[0]  # Return logits
    
    # Wrap the model
    wrapped_model = WrappedModel(model)
    wrapped_model.eval()
    
    # Prepare example input for tracing (on CPU)
    example_text = "Believe in yourself!"
    example_input = tokenizer(example_text, return_tensors="pt")["input_ids"].to("cpu")
    
    # Trace the model
    print("Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, example_input)
    
    # Define input shape with dynamic sequence length
    input_shape = ct.Shape(shape=(1, ct.RangeDim(1, args.max_length)))
    
    # Convert to CoreML (using .mlmodel format)
    print("Converting to CoreML format (.mlmodel)...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)],
        convert_to="neuralnetwork",  # Force legacy format for .mlmodel
        compute_units=ct.ComputeUnit.ALL,
    )
    
    # Add metadata for better iOS integration
    mlmodel.user_defined_metadata["example_text"] = example_text
    mlmodel.user_defined_metadata["model_type"] = "causal_lm"
    mlmodel.user_defined_metadata["max_length"] = str(args.max_length)
    
    # Save the model
    print(f"Saving CoreML model to {args.coreml_model_path}")
    mlmodel.save(args.coreml_model_path)
    
    return mlmodel

def test_coreml_model(args):
    """Test the converted CoreML model with a sample input."""
    print("\n=== Testing CoreML Model ===")
    
    import coremltools as ct
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Just in case
    
    # Prepare input
    text = "The future belongs to those who"
    print(f"Input text: '{text}'")
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    
    # Load Core ML model
    print(f"Loading CoreML model from {args.coreml_model_path}")
    mlmodel = ct.models.MLModel(args.coreml_model_path)
    
    # Print model input description
    print("\nModel input description:")
    print(mlmodel.input_description)
    
    # Convert input to the right format (int32 numpy array)
    input_array = input_ids.numpy().astype(np.int32)
    print(f"Input shape: {input_array.shape}, dtype: {input_array.dtype}")
    
    # Run prediction
    print("Running prediction...")
    try:
        # Try with dictionary input
        coreml_input = {"input_ids": input_array}
        output = mlmodel.predict(coreml_input)
        print("Prediction successful with dictionary input!")
    except Exception as e:
        print(f"Dictionary input failed: {e}")
        try:
            # Try with direct array input
            output = mlmodel.predict(input_array)
            print("Prediction successful with direct array input!")
        except Exception as e:
            print(f"Direct array input failed: {e}")
            try:
                # Get the actual input name from the model
                input_name = list(mlmodel.input_description.keys())[0]
                print(f"Trying with model's input name: {input_name}")
                coreml_input = {input_name: input_array}
                output = mlmodel.predict(coreml_input)
                print(f"Prediction successful with input name '{input_name}'!")
            except Exception as e:
                print(f"Named input failed: {e}")
                print("\nModel conversion was successful, but runtime prediction failed.")
                print("You may need to use the model directly in your iOS app.")
                return
    
    # Process output if prediction was successful
    print("\nRaw output keys:")
    for key in output:
        print(f"  - {key}: shape {output[key].shape if hasattr(output[key], 'shape') else 'N/A'}")
    
    # Try to get the logits from the output
    logits_key = None
    for key in output:
        if isinstance(output[key], np.ndarray) and len(output[key].shape) == 3:
            logits_key = key
            break
    
    if logits_key:
        print(f"\nUsing output key: {logits_key}")
        logits = output[logits_key]
        
        # Get top 5 next tokens
        print("\nTop 5 next tokens:")
        last_token_logits = logits[0, -1, :]
        top_indices = np.argsort(last_token_logits)[-5:][::-1]
        
        for idx in top_indices:
            token = tokenizer.decode([idx])
            print(f"  - '{token}' (score: {last_token_logits[idx]:.2f})")
    else:
        print("\nCouldn't identify logits in the output. Model structure may be different than expected.")
    
    print("\nCoreML model test completed!\n")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer, device = load_and_prepare_model(args)
    
    # Load and prepare dataset
    train_dataset = load_dataset_and_prepare(tokenizer, args)
    
    # Apply LoRA and train
    model = apply_lora_and_train(model, train_dataset, args, device)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.merged_model_dir)
    
    # Merge LoRA weights
    merged_model = merge_lora_weights(model, args)
    
    # Convert to CoreML
    convert_to_coreml(merged_model, tokenizer, args, device)
    
    # Test the converted model
    test_coreml_model(args)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()