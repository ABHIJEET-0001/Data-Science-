# Mini GPT-2  Model 🚀

This project involves building and training a mini version of a GPT-2–style decoder-only language model entirely from scratch using PyTorch. 

Inspired by foundational concepts of transformer architectures, this project has been fully customized and upgraded to demonstrate a deep, hands-on understanding of Sequence Modeling, Tokenization, and Self-Attention.

## ✨ Project Highlights

While building the core transformer architecture, several modifications were made to push beyond the basics:

- **Custom Dataset**: Instead of standard instructional datasets, this model was trained on *Alice's Adventures in Wonderland*, giving it a unique literary flavor during text generation.
- **Architectural Modification (Bonus)**: The standard Feed-Forward neural network was upgraded. Instead of the standard `ReLU` activation function, this model implements **`GELU` (Gaussian Error Linear Unit)**, which is the exact activation function utilized in the real OpenAI GPT-2 model.
- **Original Architecture**: The code structure features custom classes (`AttentionHead`, `MultiHeadAttentionGroup`, `TransformerLayer`, `MiniGPT`) that logically separate the complex math into readable, modular PyTorch modules.
- **Training Visualization**: Includes a plotting section utilizing `matplotlib` to visualize the model's Training and Validation Loss history over time.
- **Safe Inference**: Implemented proper `@torch.no_grad()` scoping for text generation to prevent memory leaks and graph accumulation during inference.
- **Model Checkpointing**: Added functionality to save the model weights and optimizer state to a `.pth` file, allowing training to be paused and resumed.

## 🛠️ Tech Stack

- **Python 3**
- **PyTorch** (Deep Learning Framework)
- **Matplotlib** (Data Visualization)

## 🧠 Transformer Components Implemented

1. **Character-Level Tokenizer**: Maps text to integer vocabulary IDs.
2. **Positional & Token Embeddings**: Preserves the order of sequence inputs.
3. **Causal Masked Self-Attention**: Prevents the model from "looking into the future" during next-word prediction.
4. **Multi-Head Attention**: Allows the model to process different context dependencies in parallel.
5. **Layer Normalization & Residual Connections**: Prevents vanishing gradients and stabilizes deep network training.

## 🚀 How to Run

1. Open `Mini_GPT2_0.ipynb` in Google Colab, VS Code, or Jupyter Notebook.
2. Install required dependencies: `pip install torch matplotlib`
3. Run all cells sequentially.
4. The notebook will automatically download the dataset, initialize the model, train it for 3000 steps, plot the loss graph, and generate a sample text!
