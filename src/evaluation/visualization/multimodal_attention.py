"""
Visualization tools for multimodal attention patterns in QLLM models.

This module provides specialized visualizations for attention maps between text and
visual modalities, helping to understand cross-modal integration in multimodal extensions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from PIL import Image
import torch
from typing import Dict, List, Any, Optional, Tuple, Union


def visualize_cross_modal_attention(
    image: Union[str, np.ndarray, torch.Tensor],
    text: str,
    attention_weights: np.ndarray,
    token_ids: Optional[List[int]] = None,
    tokenizer: Optional[Any] = None,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Visualize attention weights between text and image regions.
    
    Args:
        image: Image as file path, numpy array, or tensor
        text: Input text
        attention_weights: Attention weights [num_tokens, num_image_regions]
        token_ids: Optional token IDs corresponding to attention weights
        tokenizer: Optional tokenizer for token display
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    # Load image if needed
    if isinstance(image, str):
        # Load from file path
        img = np.array(Image.open(image).convert('RGB'))
    elif isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        img = image.cpu().numpy()
        if img.shape[0] in [1, 3]:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = img * 255
        img = img.astype(np.uint8)
    else:
        # Already numpy array
        img = image
    
    # Ensure attention weights are numpy array
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Get tokens if tokenizer is available
    tokens = []
    if tokenizer is not None and token_ids is not None:
        try:
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
        except:
            # Fallback if convert_ids_to_tokens not available
            tokens = [f"Token {i}" for i in range(len(attention_weights))]
    else:
        # Create token identifiers
        tokens = [f"Token {i}" for i in range(len(attention_weights))]
    
    # Create figure with gridspec for layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])
    
    # Subplot for image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_title("Image")
    ax_img.axis('off')
    
    # Subplot for text tokens
    ax_tokens = fig.add_subplot(gs[0, 1])
    ax_tokens.set_title("Text Tokens")
    ax_tokens.axis('off')
    
    # Display tokens
    max_tokens_display = 25  # Limit display to prevent crowding
    display_tokens = tokens[:max_tokens_display]
    token_positions = np.linspace(0, 1, len(display_tokens))
    
    for i, (token, pos) in enumerate(zip(display_tokens, token_positions)):
        token_str = token
        # Truncate long tokens
        if len(token_str) > 15:
            token_str = token_str[:12] + "..."
            
        ax_tokens.text(
            0.5, 1.0 - pos,
            token_str,
            ha='center',
            va='center',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5')
        )
    
    if len(tokens) > max_tokens_display:
        ax_tokens.text(
            0.5, 0.0,
            f"(+ {len(tokens) - max_tokens_display} more tokens)",
            ha='center',
            va='top',
            fontsize=10,
            style='italic'
        )
    
    # Subplot for attention heatmap
    ax_attn = fig.add_subplot(gs[1, :])
    
    # Determine image region layout
    # Assume square grid for simplicity
    num_regions = attention_weights.shape[1]
    grid_size = int(np.sqrt(num_regions))
    
    # If not perfect square, adjust
    if grid_size * grid_size != num_regions:
        grid_size = int(np.ceil(np.sqrt(num_regions)))
    
    # Reshape attention for visualization
    attn_display = attention_weights[:max_tokens_display, :]
    
    # Create colormap
    cmap = plt.cm.YlOrRd
    
    # Plot heatmap
    im = ax_attn.imshow(attn_display, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_attn)
    cbar.set_label('Attention Weight')
    
    # Set ticks and labels
    ax_attn.set_yticks(np.arange(len(display_tokens)))
    ax_attn.set_yticklabels(display_tokens)
    
    # X-axis will represent image regions
    ax_attn.set_xlabel('Image Regions')
    ax_attn.set_title('Cross-Modal Attention Weights')
    
    # Set x-ticks (simplify if too many regions)
    if num_regions <= 25:
        ax_attn.set_xticks(np.arange(num_regions))
        ax_attn.set_xticklabels([f"R{i}" for i in range(num_regions)])
    else:
        step = num_regions // 10
        ax_attn.set_xticks(np.arange(0, num_regions, step))
        ax_attn.set_xticklabels([f"R{i}" for i in range(0, num_regions, step)])
    
    # Rotate x-tick labels if needed
    plt.setp(ax_attn.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title to overall figure
    fig.suptitle(title or "Cross-Modal Attention Visualization", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cross-modal attention visualization saved to {output_file}")
    else:
        plt.show()


def visualize_attention_regions(
    image: Union[str, np.ndarray, torch.Tensor],
    text: str,
    attention_map: np.ndarray,
    box_coordinates: Optional[List[Tuple[int, int, int, int]]] = None,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    alpha: float = 0.5,
    highlight_top_k: int = 3
) -> None:
    """
    Visualize attention over specific regions of an image with text query.
    
    Args:
        image: Image as file path, numpy array, or tensor
        text: Input text or query
        attention_map: Spatial attention weights [height, width] or [1, height, width]
        box_coordinates: Optional bounding box coordinates for regions [(x1, y1, x2, y2), ...]
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
        alpha: Transparency for attention overlay
        highlight_top_k: Number of top regions to highlight
    """
    # Load image if needed
    if isinstance(image, str):
        # Load from file path
        img = np.array(Image.open(image).convert('RGB'))
    elif isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        img = image.cpu().numpy()
        if img.shape[0] in [1, 3]:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = img * 255
        img = img.astype(np.uint8)
    else:
        # Already numpy array
        img = image
    
    # Ensure attention map is correctly shaped
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach().cpu().numpy()
    
    if attention_map.ndim == 3 and attention_map.shape[0] == 1:
        attention_map = attention_map[0]  # Remove batch dimension
    
    # Resize attention map if needed
    if attention_map.shape[:2] != img.shape[:2]:
        # Simple nearest-neighbor resize
        h, w = img.shape[:2]
        resized_map = np.zeros((h, w))
        
        attn_h, attn_w = attention_map.shape
        
        for i in range(h):
            for j in range(w):
                # Map coordinates
                attn_i = min(int(i * attn_h / h), attn_h - 1)
                attn_j = min(int(j * attn_w / w), attn_w - 1)
                
                resized_map[i, j] = attention_map[attn_i, attn_j]
        
        attention_map = resized_map
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the image
    ax.imshow(img)
    
    # Create a heatmap overlay
    attention_map_normalized = attention_map / attention_map.max()
    
    # Create colormap with transparency
    cmap = plt.cm.hot
    overlay = cmap(attention_map_normalized)
    overlay[..., 3] = attention_map_normalized * alpha  # Set alpha channel
    
    # Display the attention overlay
    ax.imshow(overlay, alpha=alpha)
    
    # If box coordinates are provided, draw boxes
    if box_coordinates:
        # If we have attention values per box, find top-k boxes
        if len(box_coordinates) == attention_map.size:
            # Flatten attention map for comparison
            flat_attention = attention_map.flatten()
            
            # Get indices of top-k boxes
            top_indices = np.argsort(flat_attention)[-highlight_top_k:]
            
            # Highlight top boxes in a different color
            for i, box in enumerate(box_coordinates):
                if i in top_indices:
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1), 
                        x2 - x1, 
                        y2 - y1, 
                        linewidth=2,
                        edgecolor='g',
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add attention score
                    score = flat_attention[i]
                    ax.text(
                        x1, y1 - 5,
                        f"{score:.2f}",
                        color='white',
                        fontsize=10,
                        bbox=dict(facecolor='green', alpha=0.7)
                    )
        else:
            # Just draw all boxes
            for box in box_coordinates:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), 
                    x2 - x1, 
                    y2 - y1, 
                    linewidth=2,
                    edgecolor='blue',
                    facecolor='none'
                )
                ax.add_patch(rect)
    
    # Display query text
    text_box = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    ax.text(
        0.5, 0.05,
        f"Query: {text}",
        transform=ax.transAxes,
        fontsize=12,
        ha='center',
        va='bottom',
        bbox=text_box
    )
    
    # Set title and remove axis
    ax.set_title(title or "Attention Region Visualization")
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Attention region visualization saved to {output_file}")
    else:
        plt.show()


def visualize_multimodal_fusion(
    image: Union[str, np.ndarray, torch.Tensor],
    text: str,
    fusion_weights: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Visualize multimodal fusion mechanisms between visual and textual features.
    
    Args:
        image: Image as file path, numpy array, or tensor
        text: Input text
        fusion_weights: Fusion weights between visual and textual features
        feature_names: Names of features for axis labels
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    # Load image if needed
    if isinstance(image, str):
        # Load from file path
        img = np.array(Image.open(image).convert('RGB'))
    elif isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        img = image.cpu().numpy()
        if img.shape[0] in [1, 3]:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = img * 255
        img = img.astype(np.uint8)
    else:
        # Already numpy array
        img = image
    
    # Ensure fusion weights are numpy array
    if isinstance(fusion_weights, torch.Tensor):
        fusion_weights = fusion_weights.detach().cpu().numpy()
    
    # Create default feature names if not provided
    if feature_names is None:
        num_features = fusion_weights.shape[1] if fusion_weights.ndim > 1 else fusion_weights.shape[0]
        feature_names = [f"Feature {i+1}" for i in range(num_features)]
    
    # Create figure with gridspec for layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])
    
    # Subplot for image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_title("Image")
    ax_img.axis('off')
    
    # Subplot for text
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.text(
        0.5, 0.5,
        text,
        ha='center',
        va='center',
        fontsize=12,
        wrap=True,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    ax_text.set_title("Text Input")
    ax_text.axis('off')
    
    # Subplot for fusion visualization
    ax_fusion = fig.add_subplot(gs[1, :])
    
    # Determine visualization based on fusion weight dimensions
    if fusion_weights.ndim == 1:
        # 1D weights - bar chart
        ax_fusion.bar(
            feature_names,
            fusion_weights,
            color='purple'
        )
        ax_fusion.set_xlabel('Features')
        ax_fusion.set_ylabel('Fusion Weight')
        ax_fusion.set_title('Feature Fusion Weights')
        plt.setp(ax_fusion.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
    elif fusion_weights.ndim == 2:
        # 2D weights - heatmap
        im = ax_fusion.imshow(
            fusion_weights,
            cmap='viridis',
            aspect='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_fusion)
        cbar.set_label('Fusion Weight')
        
        # Set ticks and labels if dimensions aren't too large
        if fusion_weights.shape[0] <= 25 and fusion_weights.shape[1] <= 25:
            if fusion_weights.shape[0] <= len(feature_names):
                ax_fusion.set_yticks(np.arange(fusion_weights.shape[0]))
                ax_fusion.set_yticklabels(feature_names[:fusion_weights.shape[0]])
                
            if fusion_weights.shape[1] <= len(feature_names):
                ax_fusion.set_xticks(np.arange(fusion_weights.shape[1]))
                ax_fusion.set_xticklabels(feature_names[:fusion_weights.shape[1]])
                plt.setp(ax_fusion.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax_fusion.set_title('Feature Fusion Matrix')
    
    else:
        # Higher-dimensional weights - show top pairs
        ax_fusion.text(
            0.5, 0.5,
            "Fusion weights have >2 dimensions;\nshowing aggregated importance",
            ha='center',
            va='center',
            fontsize=14
        )
        ax_fusion.axis('off')
    
    # Add title to overall figure
    fig.suptitle(title or "Multimodal Fusion Visualization", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Multimodal fusion visualization saved to {output_file}")
    else:
        plt.show()


def visualize_modal_entanglement(
    image_features: np.ndarray,
    text_features: np.ndarray,
    entanglement_matrix: np.ndarray,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Visualize entanglement patterns between image and text feature spaces.
    
    Args:
        image_features: Image feature embeddings [batch_size, img_dim]
        text_features: Text feature embeddings [batch_size, text_dim]
        entanglement_matrix: Entanglement/correlation matrix [img_dim, text_dim]
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    # Ensure inputs are numpy arrays
    if isinstance(image_features, torch.Tensor):
        image_features = image_features.detach().cpu().numpy()
    if isinstance(text_features, torch.Tensor):
        text_features = text_features.detach().cpu().numpy()
    if isinstance(entanglement_matrix, torch.Tensor):
        entanglement_matrix = entanglement_matrix.detach().cpu().numpy()
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    
    # Panel 1: Entanglement matrix heatmap
    ax_matrix = fig.add_subplot(gs[0, :])
    im = ax_matrix.imshow(
        entanglement_matrix,
        cmap='coolwarm',
        aspect='auto',
        vmin=-1,
        vmax=1
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_matrix)
    cbar.set_label('Entanglement Strength')
    
    # Set labels
    ax_matrix.set_xlabel('Text Feature Dimensions')
    ax_matrix.set_ylabel('Image Feature Dimensions')
    ax_matrix.set_title('Modal Entanglement Matrix')
    
    # Panel 2: Feature space projection (PCA/t-SNE style visualization)
    ax_projection = fig.add_subplot(gs[1, 0])
    
    # Perform simple dimensionality reduction (PCA-like)
    # Just take the first two principal components for simplicity
    from sklearn.decomposition import PCA
    
    # Combine image and text features for joint projection
    # We'll use a simple trick - concatenate and then project
    combined_features = np.vstack([image_features, text_features])
    
    try:
        pca = PCA(n_components=2)
        projection = pca.fit_transform(combined_features)
        
        # Split back into image and text projections
        img_proj = projection[:len(image_features)]
        text_proj = projection[len(image_features):]
        
        # Plot projections
        ax_projection.scatter(
            img_proj[:, 0],
            img_proj[:, 1],
            c='blue',
            label='Image Features',
            alpha=0.6
        )
        
        ax_projection.scatter(
            text_proj[:, 0],
            text_proj[:, 1],
            c='red',
            label='Text Features',
            alpha=0.6
        )
        
        # Connect corresponding points to show alignment
        for i in range(min(len(img_proj), len(text_proj))):
            ax_projection.plot(
                [img_proj[i, 0], text_proj[i, 0]],
                [img_proj[i, 1], text_proj[i, 1]],
                'k-',
                alpha=0.2
            )
        
        ax_projection.set_title('Feature Space Projection')
        ax_projection.legend()
        ax_projection.grid(True, linestyle='--', alpha=0.7)
        
    except Exception as e:
        # Fallback if PCA fails
        ax_projection.text(
            0.5, 0.5,
            f"Projection failed: {str(e)}",
            ha='center',
            va='center'
        )
        ax_projection.set_title('Feature Space Projection (Failed)')
    
    # Panel 3: Entanglement statistics
    ax_stats = fig.add_subplot(gs[1, 1])
    
    # Calculate statistics
    avg_entanglement = np.mean(np.abs(entanglement_matrix))
    max_entanglement = np.max(np.abs(entanglement_matrix))
    
    # Find top entangled dimensions
    abs_matrix = np.abs(entanglement_matrix)
    top_indices = np.unravel_index(np.argsort(abs_matrix, axis=None)[-5:], abs_matrix.shape)
    top_img_dims = top_indices[0]
    top_text_dims = top_indices[1]
    top_values = entanglement_matrix[top_img_dims, top_text_dims]
    
    # Plot statistics
    stats = [
        f"Average Entanglement: {avg_entanglement:.4f}",
        f"Maximum Entanglement: {max_entanglement:.4f}",
        "\nTop Entangled Dimensions:"
    ]
    
    for i, (img_dim, text_dim, val) in enumerate(zip(top_img_dims, top_text_dims, top_values)):
        stats.append(f"Img Dim {img_dim} ↔ Text Dim {text_dim}: {val:.4f}")
    
    ax_stats.text(
        0.1, 0.9,
        '\n'.join(stats),
        va='top',
        ha='left',
        transform=ax_stats.transAxes,
        fontsize=12
    )
    
    ax_stats.set_title('Entanglement Statistics')
    ax_stats.axis('off')
    
    # Add overall title
    fig.suptitle(title or "Modal Entanglement Analysis", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Modal entanglement visualization saved to {output_file}")
    else:
        plt.show()