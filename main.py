import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class WrappedGPT2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits = self.model(input_ids=input_ids, return_dict=False)[0]
        return logits

def main():
    model_name = input("Enter the Hugging Face model name (e.g. 'sarahai/your-inspiration'): ")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    wrapped_model = WrappedGPT2(model)
    wrapped_model.eval()
    sample_input = torch.tensor([tokenizer.encode("The journey begins")])
    traced_model = torch.jit.trace(wrapped_model, sample_input)

    input_shape = ct.Shape(shape=(1, ct.RangeDim(1, 64)))

    file_extension = input("Enter the file extension (e.g. 'mlmodel' or 'mlpackage'): ")

    if file_extension == "mlpackage":
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)],
            compute_units=ct.ComputeUnit.ALL,
        )
    else:
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)],
            convert_to="neuralnetwork",  # 👈 force legacy format
            compute_units=ct.ComputeUnit.ALL,
        )

    output_filename = input("Enter the output Core ML model filename (without extension): ")
    output_filename = f"{output_filename}.{file_extension}"

    mlmodel.save(output_filename)

    print(f"Core ML model converted and saved successfully as {output_filename}!")


if __name__ == "__main__":
    main()
