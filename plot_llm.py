
"""
Attention Weights Visualizer

This script extracts and visualizes the query projection weights from the attention layer
of a pre-trained language model using Hugging Face's Transformers library. For speed,
only lightweight models are recommended. Currently, the recommended models are:

    - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    - "gpt2"


Author: Pablo Álvaro Hidalgo
License: MIT
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM

# ==================== CONFIGURATION ====================
# Pre-trained model:
# Recommended models for fast execution:
#   "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#   "gpt2"
MODEL_NAME = "gpt2"

# Layer index to visualize:
# Examples:
#   0  -> first layer
#   1  -> second layer
#   2  -> third layer
LAYER_INDEX = 6

# Whether to apply visual improvements (gamma transformation and percentile adjustment)
APPLY_IMPROVEMENTS = True

# Color palette for the heatmap:
# Examples: "viridis", "RdBu", "coolwarm"
COLOR_PALETTE = "viridis"

# Percentile range to adjust the heatmap color scale:
# Examples: (5, 95), (1, 99), (10, 90)
PERCENTILE_RANGE = (5, 95)

# Whether to save the generated plot in a folder named "img/{model_name}"
SAVE_IMAGE = True
# ========================================================

def visualize_attention_weights(model_name=MODEL_NAME,
                                layer_index=LAYER_INDEX,
                                apply_improvements=APPLY_IMPROVEMENTS,
                                color_palette=COLOR_PALETTE,
                                percentile_range=PERCENTILE_RANGE,
                                save_image=SAVE_IMAGE):
    """
    Visualizes the query projection weights from the attention layer of a pre-trained model.
    If direct access fails due to differing architectures, alternative methods are attempted.
    
    Parameters:
    - model_name (str): Name of the pre-trained model to load.
    - layer_index (int): Index of the attention layer to visualize.
    - apply_improvements (bool): If True, apply gamma transformation and percentile adjustment.
    - color_palette (str): Color palette for the heatmap.
    - percentile_range (tuple): Percentile range to adjust the color scale (vmin, vmax).
    - save_image (bool): If True, save the plot in "img/{model_name}".
    """
    
    # Load the pre-trained model
    print(f"Loading model '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Attempt to access the desired attention layer
    layer = None
    try:
        # Common route for some models (e.g., TinyLlama)
        layer = model.model.layers[layer_index]
    except AttributeError:
        try:
            # Alternative route used in models like GPT-2
            layer = model.transformer.h[layer_index]
        except AttributeError:
            print("Error: Could not access the layer using 'model.model.layers' or 'model.transformer.h'.")
            return
    except IndexError:
        print(f"Error: Layer index {layer_index} is out of range in 'model.model.layers'.")
        return

    # Extract the weights of the query projection from the attention layer.
    weights = None
    try:
        # Route 1: For models with self_attn.q_proj (e.g., TinyLlama)
        weights = layer.self_attn.q_proj.weight.detach().cpu().numpy()
        print("Weights extracted using 'self_attn.q_proj.weight'.")
    except AttributeError:
        try:
            # Route 2: For GPT-2, where a single Linear layer 'c_attn' is used
            full_weights = layer.attn.c_attn.weight.detach().cpu().numpy()
            n_embd = full_weights.shape[0]
            query_weights = full_weights[:, :n_embd]  # Extract query part (first n_embd columns)
            weights = query_weights.T  # Transpose to have rows as output dimensions
            print("Weights extracted using 'attn.c_attn.weight' for GPT-2.")
        except AttributeError:
            print("Error: Could not access known weight attributes in the selected layer.")
            return

    # Normalize the weights (subtract mean and divide by standard deviation)
    mean = np.mean(weights)
    std = np.std(weights)
    normalized_weights = (weights - mean) / std

    if apply_improvements:
        # Apply gamma transformation (gamma=0.5 corresponds to a square-root transformation)
        gamma = 0.5
        transformed_weights = np.sign(normalized_weights) * np.abs(normalized_weights) ** gamma
        vmin, vmax = np.percentile(transformed_weights, [percentile_range[0], percentile_range[1]])
        data_to_plot = transformed_weights
        print("Applied gamma transformation and percentile adjustment.")
    else:
        vmin, vmax = normalized_weights.min(), normalized_weights.max()
        data_to_plot = normalized_weights

    # Create the heatmap with labeled axes
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(data_to_plot, cmap=color_palette, vmin=vmin, vmax=vmax, center=0)
    ax.set_xlabel("Input Dimension")   # Each column represents an input feature
    ax.set_ylabel("Output Dimension")    # Each row represents an output neuron/component
    plt.title(f"Heatmap of Layer {layer_index} (query) of Attention\nModel: {model_name}")

    # Save the image if specified
    if save_image:
        sanitized_model_name = model_name.replace("/", "_").replace("\\", "_")
        output_dir = os.path.join("img", sanitized_model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f"layer_{layer_index}.png")
        plt.savefig(filename)
        print(f"Image saved at: {filename}")

    plt.show()
    print(f"Visualization of weights for '{model_name}' at layer {layer_index} is complete.")


if __name__ == '__main__':
    visualize_attention_weights()
