import torch
import coremltools as ct
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    model_name = input("Enter the Hugging Face model name (e.g. 'sarahai/your-inspiration'): ")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    sample_text = "Life is"
    sample_input = torch.tensor([tokenizer.encode(sample_text)])

    traced_model = torch.jit.trace(model, sample_input)

    input_shape = ct.Shape(shape=(1, ct.RangeDim(1, 64)))

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=torch.int32)],
        compute_units=ct.ComputeUnit.ALL,
    )

    output_filename = input("Enter the output Core ML model filename (without .mlmodel extension): ")
    output_filename = f"{output_filename}.mlmodel"
    
    mlmodel.save(output_filename)

    print(f"Core ML model converted and saved successfully as {output_filename}!")


if __name__ == "__main__":
    main()
