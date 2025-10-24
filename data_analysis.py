import torch
from model import GPT, GPTConfig
import numpy as np
import matplotlib.pyplot as plt
#from caas_jupyter_tools import display_dataframe_to_user
from sklearn.decomposition import PCA

def load_model(checkpoint_path, device='cuda', override_args=None):
    """
    Load a pre-trained GPT model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on ('cuda' or 'cpu')
        override_args: Optional dict of model args to override (e.g., {'dropout': 0.0})
    
    Returns:
        model: Loaded GPT model
        checkpoint: Full checkpoint dict (contains iter_num, best_val_loss, etc.)
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    checkpoint_model_args = checkpoint['model_args']
    
    # Apply any overrides (useful for inference, e.g., setting dropout=0)
    if override_args:
        checkpoint_model_args.update(override_args)
    
    # Create the model
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    
    # Remove unwanted prefix if present (common with compiled models)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Set to eval mode by default (use model.train() if you want to continue training)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"- Iterations trained: {checkpoint.get('iter_num', 'unknown')}")
    print(f"- Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model, checkpoint



model, checkpoint = load_model('path/to/checkpoint.pt', device='cpu')

embedding_matrix = model.transformer.wte.weight

meta = checkpoint['meta']
stoi = meta['stoi']  # string to index mapping

chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

token_ids = [stoi[c] for c in chars]
digit_embeddings = embedding_matrix[token_ids].detach().cpu().numpy()

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(digit_embeddings)

plt.figure(figsize=(10, 8))
for i, char in enumerate(chars):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=100)
    plt.annotate(char, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                fontsize=12, ha='center')
plt.title('Digit Embeddings (PCA projection)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

