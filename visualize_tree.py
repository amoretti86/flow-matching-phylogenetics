import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys
import os

# Import the visualization functions
from poincare_tree_visualization import (
    fix_embeddings, 
    visualize_phylogenetic_tree_in_poincare
)

def main():
    parser = argparse.ArgumentParser(description="Visualize phylogenetic tree in the Poincaré disk")
    parser.add_argument("--embeddings-file", type=str, default="extracted_data/all_embeddings.npy",
                        help="Path to embeddings numpy file")
    parser.add_argument("--results-dir", type=str, default="extracted_data",
                        help="Directory with extracted results")
    parser.add_argument("--taxa-file", type=str,
                        help="Optional file with taxa names, one per line")
    parser.add_argument("--output-file", type=str, default="phylogenetic_tree_poincare.png",
                        help="Output file for visualization")
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_file}")
    embeddings = np.load(args.embeddings_file)
    
    # Fix embeddings if needed
    embeddings = fix_embeddings(embeddings)
    
    # Load or create merge indexes from the results directory
    merge_indexes_file = os.path.join(args.results_dir, "merge_indexes.npy")
    try:
        print(f"Looking for merge indexes in {merge_indexes_file}")
        merge_indexes = np.load(merge_indexes_file)
        print(f"Loaded merge indexes with shape {merge_indexes.shape}")
    except FileNotFoundError:
        # Try loading from checkpoint
        checkpoint_file = os.path.join(args.results_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_file):
            print(f"Loading merge indexes from checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if 'best_merge_indexes_N1x2' in checkpoint:
                merge_indexes = checkpoint['best_merge_indexes_N1x2'].numpy()
                print(f"Extracted merge indexes with shape {merge_indexes.shape}")
            else:
                print("Checkpoint doesn't contain merge indexes, using dummy values")
                n_taxa = embeddings.shape[0] // 2 + 1
                n_internal = embeddings.shape[0] - n_taxa
                merge_indexes = np.zeros((n_internal, 2), dtype=int)
                
                # Create a simple sequential merge pattern
                for i in range(n_internal):
                    if i == 0:
                        merge_indexes[i] = [0, 1]
                    elif i == 1:
                        merge_indexes[i] = [2, 3]
                    else:
                        merge_indexes[i] = [n_taxa + i - 2, (i + 2) % n_taxa]
        else:
            print("Merge indexes file not found and no checkpoint available.")
            print("Creating dummy merge indexes (visualization won't be accurate).")
            n_taxa = embeddings.shape[0] // 2 + 1
            n_internal = embeddings.shape[0] - n_taxa
            merge_indexes = np.zeros((n_internal, 2), dtype=int)
            
            # Create a simple sequential merge pattern
            for i in range(n_internal):
                if i == 0:
                    merge_indexes[i] = [0, 1]
                elif i == 1:
                    merge_indexes[i] = [2, 3]
                else:
                    merge_indexes[i] = [n_taxa + i - 2, (i + 2) % n_taxa]
    
    # Load taxa names if provided, otherwise create dummy names
    if args.taxa_file and os.path.exists(args.taxa_file):
        print(f"Loading taxa names from {args.taxa_file}")
        with open(args.taxa_file, 'r') as f:
            taxa_names = [line.strip() for line in f.readlines()]
    else:
        print("No taxa names file provided, using dummy names")
        n_taxa = embeddings.shape[0] // 2 + 1
        taxa_names = [f"Taxon_{i+1}" for i in range(n_taxa)]
    
    print(f"Found {len(taxa_names)} taxa names")
    
    # Create the visualization
    print("Creating visualization...")
    fig, ax = visualize_phylogenetic_tree_in_poincare(
        embeddings, merge_indexes, taxa_names, 
        output_file=args.output_file
    )
    
    print(f"Visualization saved to {args.output_file}")

if __name__ == "__main__":
    main()