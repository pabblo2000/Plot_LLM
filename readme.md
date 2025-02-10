# Attention Weights Visualizer

The **Attention Weights Visualizer** is a Python tool for extracting and visualizing the query projection weights of the attention layers from pre-trained language models available on [Hugging Face](https://huggingface.co). For faster execution, it is recommended to use only lightweight models such as:

- `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`
- `"gpt2"`

## Features

- **Multi-architecture Support:**  
  Supports models with different internal structures (e.g., those with `self_attn.q_proj` or `attn.c_attn`).

- **Customizable Visualization:**  
  - Option to apply gamma transformation and percentile adjustments for improved contrast.
  - Choose from several color palettes and adjust percentile ranges.

- **Automatic Saving:**  
  Save the generated heatmap images in a structured folder (`img/{model_name}`).

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/attention-weights-visualizer.git
   cd attention-weights-visualizer

## Example of image
![Example image](https://raw.githubusercontent.com/pabblo2000/Plot_LLM/master/img/gpt2/layer_0.png)

