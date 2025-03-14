import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def extract_sequence_distributions_and_embeddings(vcsmc_model, taxa_N, data_NxSxA):
    """
    Extract the distribution over nucleotide sequences and embeddings in the Poincaré disk.
    
    Args:
        vcsmc_model: The trained VCSMC model
        taxa_N: List of taxa names
        data_NxSxA: Tensor of sequences (N, S, A)
        
    Returns:
        A tuple containing:
        - sequence_distributions: Tensor of shape (N, S, A) with probability distributions
        - all_embeddings: Tensor of shape (N + N-1, 2) with embeddings for all nodes
        - taxa_embeddings: Tensor of shape (N, 2) with embeddings just for taxa
        - internal_embeddings: Tensor of shape (N-1, 2) with embeddings for internal nodes
    """
    N = len(taxa_N)  # Number of taxa
    S = data_NxSxA.shape[1]  # Number of sites
    A = data_NxSxA.shape[2]  # Alphabet size (4 for A,C,G,T)
    
    # Step 1: Run the model once to get the tree structure and embeddings
    site_positions_SxSfull = torch.eye(S)  # One-hot encoding of site positions
    
    # Run VCSMC to get a result
    with torch.no_grad():
        result = vcsmc_model(taxa_N, data_NxSxA, data_NxSxA, site_positions_SxSfull)
    
    # Step 2: Extract the sequence distributions
    # Get leaf node embeddings
    leaf_embeddings_NxD = vcsmc_model.proposal.seq_encoder(data_NxSxA)
    
    # Get site positions
    site_positions_SxC = vcsmc_model.q_matrix_decoder.site_positions_encoder(site_positions_SxSfull)
    
    # Get sequence distributions (stationary probabilities) for leaf nodes
    leaf_distributions_NxSxA = vcsmc_model.q_matrix_decoder.stat_probs_VxSxA(
        leaf_embeddings_NxD, site_positions_SxC
    )
    
    # Step 3: Extract all embeddings (taxa + internal nodes)
    # Get the best tree
    best_tree_idx = torch.argmax(result["log_likelihood_K"])
    
    # Get leaf embeddings (these are the taxa embeddings)
    taxa_embeddings = leaf_embeddings_NxD
    
    # Get internal node embeddings from the best tree
    internal_embeddings = result["best_embeddings_N1xD"]
    
    # Combine all embeddings
    all_embeddings = torch.cat([taxa_embeddings, internal_embeddings], dim=0)
    
    return (
        leaf_distributions_NxSxA,
        all_embeddings,
        taxa_embeddings,
        internal_embeddings
    )

def visualize_poincare_disk(all_embeddings, taxa_names=None, internal_names=None):
    """
    Visualize embeddings in the Poincaré disk.
    
    Args:
        all_embeddings: Tensor of shape (N + N-1, 2) with embeddings
        taxa_names: List of taxa names (optional)
        internal_names: List of internal node names (optional)
    """
    # Convert to numpy for plotting
    embeddings_np = all_embeddings.detach().cpu().numpy()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw the Poincaré disk boundary
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(circle)
    
    # Separate taxa and internal nodes
    n_taxa = len(taxa_names) if taxa_names else embeddings_np.shape[0] // 2 + 1
    
    # Plot taxa points
    taxa_points = embeddings_np[:n_taxa]
    ax.scatter(taxa_points[:, 0], taxa_points[:, 1], c='blue', s=100, label='Taxa')
    
    # Plot internal nodes
    internal_points = embeddings_np[n_taxa:]
    ax.scatter(internal_points[:, 0], internal_points[:, 1], c='red', s=100, label='Internal Nodes')
    
    # Add labels if provided
    if taxa_names:
        for i, name in enumerate(taxa_names):
            ax.annotate(name, (taxa_points[i, 0], taxa_points[i, 1]), fontsize=10)
    
    if internal_names:
        for i, name in enumerate(internal_names):
            ax.annotate(name, (internal_points[i, 0], internal_points[i, 1]), fontsize=10)
    
    # Set limits and labels
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Embeddings in the Poincaré Disk')
    ax.legend()
    
    plt.tight_layout()
    return fig

def save_extracted_data(sequence_distributions, all_embeddings, taxa_embeddings, 
                       internal_embeddings, output_dir="extracted_data"):
    """
    Save extracted data to files.
    
    Args:
        sequence_distributions: Tensor of shape (N, S, A) with probability distributions
        all_embeddings: Tensor of shape (N + N-1, 2) with embeddings for all nodes
        taxa_embeddings: Tensor of shape (N, 2) with embeddings just for taxa
        internal_embeddings: Tensor of shape (N-1, 2) with embeddings for internal nodes
        output_dir: Directory to save files
    """
    # Create directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save tensors
    torch.save(sequence_distributions, Path(output_dir) / "sequence_distributions.pt")
    torch.save(all_embeddings, Path(output_dir) / "all_embeddings.pt")
    torch.save(taxa_embeddings, Path(output_dir) / "taxa_embeddings.pt")
    torch.save(internal_embeddings, Path(output_dir) / "internal_embeddings.pt")
    
    # Also save as numpy arrays for easier use in other tools
    np.save(Path(output_dir) / "sequence_distributions.npy", 
            sequence_distributions.detach().cpu().numpy())
    np.save(Path(output_dir) / "all_embeddings.npy", 
            all_embeddings.detach().cpu().numpy())
    np.save(Path(output_dir) / "taxa_embeddings.npy", 
            taxa_embeddings.detach().cpu().numpy())
    np.save(Path(output_dir) / "internal_embeddings.npy", 
            internal_embeddings.detach().cpu().numpy())
    
    print(f"Data saved to {output_dir}/")

# Example usage (to be integrated into your workflow):
"""
# Load a trained model
from vcsmc.phy import load_phy, A4_ALPHABET
from vcsmc.distances import Hyperbolic
from vcsmc.encoders import EmbeddingTableSequenceEncoder, HyperbolicGeodesicMidpointMergeEncoder
from vcsmc.proposals import EmbeddingProposal
from vcsmc.q_matrix_decoders import JC69QMatrixDecoder
from vcsmc.vcsmc import VCSMC

# 1. Load your data (replace with your file path)
N, S, A, data_NxSxA, taxa_N = load_phy("your_data.phy", A4_ALPHABET)

# 2. Load a trained model (or construct a new one if needed)
distance = Hyperbolic(initial_scale=0.1)
seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=2)
merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
q_matrix_decoder = JC69QMatrixDecoder(A=A)
proposal = EmbeddingProposal(distance, seq_encoder, merge_encoder, N=N, lookahead_merge=True)
vcsmc_model = VCSMC(q_matrix_decoder, proposal, N=N, K=10)

# 3. Load checkpoint if available
# checkpoint = torch.load("path/to/checkpoint.pt")
# vcsmc_model.load_state_dict(checkpoint['vcsmc'].state_dict())

# 4. Extract and visualize
sequence_distributions, all_embeddings, taxa_embeddings, internal_embeddings = \
    extract_sequence_distributions_and_embeddings(vcsmc_model, taxa_N, data_NxSxA)

# 5. Visualize
fig = visualize_poincare_disk(all_embeddings, taxa_names=taxa_N)
fig.savefig("poincare_embeddings.png")

# 6. Save data
save_extracted_data(sequence_distributions, all_embeddings, taxa_embeddings, internal_embeddings)
"""