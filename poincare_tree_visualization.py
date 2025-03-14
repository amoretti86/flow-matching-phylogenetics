import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def fix_embeddings(embeddings, max_norm=0.99):
    """
    Ensure all embeddings are within the Poincaré disk by rescaling if necessary.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = np.array(embeddings)
    
    # Check if any points are outside the unit disk
    norms = np.linalg.norm(embeddings_np, axis=1)
    max_found_norm = np.max(norms)
    
    if max_found_norm >= 1.0:
        # Scale all points to ensure they're inside the disk
        scale_factor = max_norm / max_found_norm
        fixed_embeddings = embeddings_np * scale_factor
        print(f"Rescaled embeddings by factor {scale_factor}")
        return fixed_embeddings
    
    return embeddings_np

def cart_to_polar(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (r, theta).
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def create_complex_number(x, y):
    """
    Creates a complex number from real and imaginary parts.
    """
    return complex(x, y)

def find_inverse(p):
    """
    Finds the circular inverse of a point P with respect to the unit circle in the Poincare disk.
    """
    x, y = p
    norm_squared = x**2 + y**2
    if norm_squared < 1e-10:  # Handle point at origin
        return (1e10, 0)
    return (x / norm_squared, y / norm_squared)

def circle_from_3_points(z1, z2, z3):
    """
    Computes the center and radius of the circle passing through three points in the complex plane.
    """
    if (z1 == z2) or (z2 == z3) or (z3 == z1):
        # Special case: straight line through origin
        if abs(z1.imag * z2.real - z1.real * z2.imag) < 1e-10:
            return complex(1e10, 0), 1e10  # Very large radius

    w = (z3 - z1) / (z2 - z1)
    
    if abs(w.imag) <= 1e-10:
        # Points are collinear
        # For diameter case, return a circle with center at midpoint and appropriate radius
        if abs(abs(z1) + abs(z2) - abs(z1 - z2)) < 1e-10:
            center = (z1 + z2) / 2
            radius = abs(z1 - center)
            return center, radius
        return complex(1e10, 0), 1e10  # Very large radius
    
    # Center of the circle through three points
    c = (z2 - z1) * (w - abs(w) ** 2) / (2j * w.imag) + z1
    # Radius of the circle
    r = abs(z1 - c)
    
    return c, r

def compute_geodesic_arc_points(p1, p2, num_points=100):
    """
    Computes points along the geodesic arc between p1 and p2 in the Poincaré disk.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        num_points: Number of points to generate along the arc
        
    Returns:
        Array of points (x, y) along the geodesic
    """
    # Handle special case where one point is at the origin
    if np.linalg.norm(p1) < 1e-10:
        # Straight line from origin to p2
        t = np.linspace(0, 1, num_points)
        return np.array([(t_i*p2[0], t_i*p2[1]) for t_i in t])
    
    if np.linalg.norm(p2) < 1e-10:
        # Straight line from p1 to origin
        t = np.linspace(0, 1, num_points)
        return np.array([((1-t_i)*p1[0], (1-t_i)*p1[1]) for t_i in t])
    
    # Convert P and Q into complex numbers
    p1c = create_complex_number(*p1)
    p2c = create_complex_number(*p2)
    
    # Find inverse of P
    p1inv = find_inverse(p1)
    p1invc = create_complex_number(*p1inv)
    
    try:
        # Calculate center and radius of the circle passing through P, Q, and P inverse
        c, r = circle_from_3_points(p1c, p2c, p1invc)
        
        # Check if the center is very far away (almost straight line)
        if abs(c) > 1e6:
            # Straight line approximation
            t = np.linspace(0, 1, num_points)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            return np.column_stack((x, y))
        
        # Convert points P and Q to polar angles on the geodesic circle
        p_theta = np.arctan2(p1[1] - c.imag, p1[0] - c.real)
        q_theta = np.arctan2(p2[1] - c.imag, p2[0] - c.real)
        
        # Ensure we take the shorter arc
        if abs(p_theta - q_theta) > np.pi:
            if p_theta < q_theta:
                p_theta += 2*np.pi
            else:
                q_theta += 2*np.pi
                
        # Generate points along the arc
        theta = np.linspace(p_theta, q_theta, num_points)
        x = c.real + r * np.cos(theta)
        y = c.imag + r * np.sin(theta)
        
        return np.column_stack((x, y))
    
    except Exception as e:
        print(f"Error computing geodesic for {p1} and {p2}: {e}")
        # Fallback to straight line
        t = np.linspace(0, 1, num_points)
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        return np.column_stack((x, y))

def visualize_phylogenetic_tree_in_poincare(all_embeddings, merge_indexes, taxa_names=None, output_file=None):
    """
    Visualize the full phylogenetic tree inside the Poincaré disk with geodesic edges.
    
    Args:
        all_embeddings: Array or tensor of shape (N_taxa + N_internal, 2) with node embeddings
        merge_indexes: Array of shape (N_internal, 2) with merge indices 
                      (which nodes were merged to create each internal node)
        taxa_names: List of taxa names
        output_file: Path to save the visualization (optional)
    """
    # Ensure embeddings are in numpy format and properly within the disk
    embeddings = fix_embeddings(all_embeddings)
    
    # If merge_indexes is a torch tensor, convert to numpy
    if isinstance(merge_indexes, torch.Tensor):
        merge_indexes = merge_indexes.cpu().numpy()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw the Poincaré disk boundary
    theta = np.linspace(0, 2*np.pi, 100)
    boundary_x = np.cos(theta)
    boundary_y = np.sin(theta)
    ax.plot(boundary_x, boundary_y, 'b-', linewidth=1.5, alpha=0.5, label='Poincaré Disk')
    
    # Determine number of taxa
    n_taxa = len(taxa_names) if taxa_names else embeddings.shape[0] // 2 + 1
    n_total = embeddings.shape[0]
    n_internal = n_total - n_taxa
    
    # Separate embeddings for taxa and internal nodes
    taxa_embeddings = embeddings[:n_taxa]
    internal_embeddings = embeddings[n_taxa:]
    
    # Plot the nodes
    ax.scatter(taxa_embeddings[:, 0], taxa_embeddings[:, 1], 
              c='blue', s=100, label='Taxa', zorder=10)
    ax.scatter(internal_embeddings[:, 0], internal_embeddings[:, 1], 
              c='red', s=100, label='Internal Nodes', zorder=10)
    
    # Add labels for taxa
    if taxa_names:
        for i, name in enumerate(taxa_names):
            ax.annotate(name, 
                       (taxa_embeddings[i, 0], taxa_embeddings[i, 1]),
                       fontsize=10, 
                       xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7),
                       zorder=11)
    
    # Add labels for internal nodes
    for i in range(n_internal):
        ax.annotate(f"I{i+1}", 
                   (internal_embeddings[i, 0], internal_embeddings[i, 1]),
                   fontsize=8, 
                   xytext=(3, 3), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.2", fc="mistyrose", ec="lightcoral", alpha=0.7),
                   zorder=11)
    
    # Draw edges as geodesics
    for i in range(n_internal):
        internal_idx = n_taxa + i
        
        # Get the two child nodes that were merged to create this internal node
        child1_idx = merge_indexes[i, 0]
        child2_idx = merge_indexes[i, 1]
        
        # Get the coordinates
        parent_point = embeddings[internal_idx]
        child1_point = embeddings[child1_idx]
        child2_point = embeddings[child2_idx]
        
        # Draw geodesic from parent to child1
        arc_points1 = compute_geodesic_arc_points(parent_point, child1_point)
        ax.plot(arc_points1[:, 0], arc_points1[:, 1], 'g-', linewidth=1.5, alpha=0.7)
        
        # Draw geodesic from parent to child2
        arc_points2 = compute_geodesic_arc_points(parent_point, child2_point)
        ax.plot(arc_points2[:, 0], arc_points2[:, 1], 'g-', linewidth=1.5, alpha=0.7)
    
    # Set plot limits and properties
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Phylogenetic Tree in the Poincaré Disk', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    return fig, ax

# Example usage:
if __name__ == "__main__":
    # Load the fixed embeddings
    embeddings_file = "extracted_data/all_embeddings_fixed.npy"
    merge_indexes_file = "extracted_data/merge_indexes.npy"
    
    # Taxa names (replace with your actual taxa names)
    taxa_names = [f"Taxon_{i+1}" for i in range(12)]
    
    all_embeddings = np.load(embeddings_file)
    merge_indexes = np.load(merge_indexes_file)
    
    fig, ax = visualize_phylogenetic_tree_in_poincare(
        all_embeddings, merge_indexes, taxa_names, 
        output_file="phylogenetic_tree_poincare.png"
    )
    
    plt.show()