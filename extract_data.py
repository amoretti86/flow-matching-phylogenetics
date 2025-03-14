import argparse
import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import from the repository
from vcsmc.phy import load_phy, A4_ALPHABET
from vcsmc.train import load_checkpoint
from vcsmc.utils.train_utils import find_most_recent_path

# Import the extraction utilities
from extraction_utils import (
    extract_sequence_distributions_and_embeddings,
    visualize_poincare_disk,
    save_extracted_data
)

def main():
    parser = argparse.ArgumentParser(description="Extract sequence distributions and embeddings from a trained model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", 
                        help="Directory containing the checkpoints")
    parser.add_argument("--output-dir", type=str, default="extracted_data",
                        help="Directory to save extracted data")
    parser.add_argument("--best", action="store_true", 
                        help="Use the best checkpoint instead of the latest")
    args = parser.parse_args()
    
    # Load the checkpoint
    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    checkpoint_args, checkpoint = load_checkpoint(
        search_dir=args.checkpoint_dir,
        start_epoch="best" if args.best else None
    )
    
    # Extract model components
    vcsmc_model = checkpoint["vcsmc"]
    data_NxSxA = checkpoint_args["data_NxSxA"]
    taxa_N = checkpoint_args["taxa_N"]
    
    # Set the model to evaluation mode
    vcsmc_model.eval()
    
    # Extract distributions and embeddings
    print("Extracting sequence distributions and embeddings...")
    with torch.no_grad():
        sequence_distributions, all_embeddings, taxa_embeddings, internal_embeddings = \
            extract_sequence_distributions_and_embeddings(vcsmc_model, taxa_N, data_NxSxA)
    
    # Print shape information
    print(f"Sequence distributions shape: {sequence_distributions.shape}")
    print(f"All embeddings shape: {all_embeddings.shape}")
    print(f"Taxa embeddings shape: {taxa_embeddings.shape}")
    print(f"Internal embeddings shape: {internal_embeddings.shape}")
    
    # Visualize embeddings in the Poincaré disk
    print("Visualizing embeddings...")
    fig = visualize_poincare_disk(all_embeddings, taxa_names=taxa_N)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save visualization
    fig.savefig(os.path.join(args.output_dir, "poincare_embeddings.png"))
    print(f"Visualization saved to {args.output_dir}/poincare_embeddings.png")
    
    # Save the extracted data
    print("Saving extracted data...")
    save_extracted_data(
        sequence_distributions, 
        all_embeddings, 
        taxa_embeddings, 
        internal_embeddings,
        output_dir=args.output_dir
    )
    
    print("Extraction completed successfully!")

if __name__ == "__main__":
    main()