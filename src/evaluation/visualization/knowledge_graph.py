"""
Visualization tools for knowledge graph structures and traversals in QLLM models.

This module provides specialized visualizations for knowledge graphs, relationship
structures, and graph-based reasoning paths used in the memory extension.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Any, Optional, Tuple, Union


def visualize_knowledge_graph(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    layout: str = "spring",
    node_size_factor: float = 1000,
    max_nodes: int = 50
) -> None:
    """
    Visualize a knowledge graph with entities and relations.
    
    Args:
        entities: List of entity dictionaries with 'name', 'type', and optional 'metadata'
        relations: List of relation dictionaries with 'source', 'target', 'relation_type', and optional 'metadata'
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
        layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai', 'planar', 'random')
        node_size_factor: Factor to scale node sizes
        max_nodes: Maximum number of nodes to display (for large graphs)
    """
    if not entities or not relations:
        print("No entities or relations provided")
        return
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Limit nodes if too many
    if len(entities) > max_nodes:
        print(f"Too many entities ({len(entities)}), limiting to {max_nodes}")
        entities = entities[:max_nodes]
    
    # Add nodes (entities)
    entity_types = {}
    for entity in entities:
        name = entity.get("name", "")
        entity_type = entity.get("type", 0)
        metadata = entity.get("metadata", {})
        
        if not name:
            continue
            
        # Add node
        G.add_node(
            name,
            type=entity_type,
            metadata=metadata
        )
        
        # Track entity types for coloring
        entity_types[name] = entity_type
    
    # Add edges (relations)
    edge_types = {}
    for relation in relations:
        source = relation.get("source", "")
        target = relation.get("target", "")
        relation_type = relation.get("relation_type", 0)
        strength = relation.get("metadata", {}).get("strength", 0.5)
        
        # Skip if source or target is not in the graph
        if source not in G.nodes or target not in G.nodes:
            continue
            
        # Add edge
        G.add_edge(
            source,
            target,
            type=relation_type,
            weight=strength
        )
        
        # Track relation types
        edge_types[(source, target)] = relation_type
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=0.3, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "planar":
        # Try planar layout, fall back to spring if not planar
        try:
            pos = nx.planar_layout(G)
        except nx.NetworkXException:
            pos = nx.spring_layout(G, k=0.3, iterations=50)
    elif layout == "random":
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Get unique entity types for coloring
    unique_entity_types = sorted(set(entity_types.values()))
    entity_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_entity_types), 1)))
    
    # Create node color map
    node_colors = [entity_colors[unique_entity_types.index(entity_types[node])] for node in G.nodes]
    
    # Get unique relation types for coloring
    unique_relation_types = sorted(set(edge_types.values()))
    edge_color_map = plt.cm.Dark2(np.linspace(0, 1, max(len(unique_relation_types), 1)))
    
    # Create edge color map
    edge_colors = [edge_color_map[unique_relation_types.index(edge_types[edge])] for edge in G.edges]
    
    # Draw nodes with different sizes based on degree
    node_sizes = [node_size_factor * (1 + 0.5 * G.degree(node)) for node in G.nodes]
    
    # Draw the graph
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8
    )
    
    # Draw edges
    edge_widths = [1 + 2 * G[u][v].get("weight", 0.5) for u, v in G.edges]
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.7,
        connectionstyle='arc3,rad=0.1',  # Curved edges
        arrowsize=15
    )
    
    # Add labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_family='sans-serif'
    )
    
    # Create legend for entity types
    entity_patches = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=entity_colors[i], markersize=10, 
                              label=f'Entity Type {t}')
                  for i, t in enumerate(unique_entity_types)]
    
    # Create legend for relation types
    relation_patches = [plt.Line2D([0], [0], color=edge_color_map[i], 
                                lw=2, label=f'Relation Type {t}')
                    for i, t in enumerate(unique_relation_types)]
    
    # Add legends
    plt.legend(handles=entity_patches + relation_patches, 
              title="Entity & Relation Types",
              loc='upper right', 
              bbox_to_anchor=(1.15, 1))
    
    # Set title and remove axis
    plt.title(title or "Knowledge Graph Visualization")
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Knowledge graph visualization saved to {output_file}")
    else:
        plt.show()


def visualize_traversal_path(
    path: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Visualize a specific traversal path through the knowledge graph.
    
    Args:
        path: List of nodes and edges in the traversal path
        entities: List of all entity dictionaries
        relations: List of all relation dictionaries
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    if not path:
        print("No path provided")
        return
    
    # Create graph with all entities and relations
    full_graph = nx.DiGraph()
    
    # Add all entities as nodes
    for entity in entities:
        name = entity.get("name", "")
        if name:
            full_graph.add_node(name, **entity)
    
    # Add all relations as edges
    for relation in relations:
        source = relation.get("source", "")
        target = relation.get("target", "")
        if source and target and source in full_graph.nodes and target in full_graph.nodes:
            full_graph.add_edge(source, target, **relation)
    
    # Create subgraph with just the path
    path_nodes = []
    path_edges = []
    
    for step in path:
        if "node" in step:
            path_nodes.append(step["node"])
        elif "edge" in step:
            source = step["edge"].get("source", "")
            target = step["edge"].get("target", "")
            if source and target:
                path_edges.append((source, target))
    
    # Get all nodes involved in path
    all_path_nodes = set(path_nodes)
    for source, target in path_edges:
        all_path_nodes.add(source)
        all_path_nodes.add(target)
    
    # Create subgraph
    path_graph = full_graph.subgraph(all_path_nodes)
    
    # Prepare figure
    plt.figure(figsize=figsize)
    
    # Position nodes
    pos = nx.spring_layout(path_graph, k=0.3, seed=42)
    
    # Draw non-path nodes and edges (grayed out)
    non_path_nodes = [node for node in path_graph.nodes if node not in path_nodes]
    nx.draw_networkx_nodes(
        path_graph, pos,
        nodelist=non_path_nodes,
        node_color='lightgray',
        alpha=0.5,
        node_size=800
    )
    
    non_path_edges = [edge for edge in path_graph.edges if edge not in path_edges]
    nx.draw_networkx_edges(
        path_graph, pos,
        edgelist=non_path_edges,
        edge_color='lightgray',
        alpha=0.3,
        width=1,
        arrowsize=10
    )
    
    # Draw path nodes with special highlighting
    nx.draw_networkx_nodes(
        path_graph, pos,
        nodelist=path_nodes,
        node_color='red',
        alpha=0.8,
        node_size=1000
    )
    
    # Draw path edges with special highlighting
    nx.draw_networkx_edges(
        path_graph, pos,
        edgelist=path_edges,
        edge_color='red',
        alpha=0.8,
        width=2,
        arrowsize=15,
        connectionstyle='arc3,rad=0.1'
    )
    
    # Add node labels
    nx.draw_networkx_labels(
        path_graph, pos,
        font_size=10,
        font_weight='bold'
    )
    
    # Add title and remove axis
    plt.title(title or "Knowledge Graph Traversal Path")
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Traversal path visualization saved to {output_file}")
    else:
        plt.show()


def visualize_complex_structure(
    structure: Dict[str, Any],
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Visualize a complex knowledge structure with multiple components and relation types.
    
    Args:
        structure: Dictionary with 'concept', 'components', and 'relations' keys
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    if not structure:
        print("No structure provided")
        return
    
    concept = structure.get("concept", "Unknown")
    components = structure.get("components", [])
    relations = structure.get("relations", [])
    
    if not components or not relations:
        print("No components or relations in structure")
        return
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add the central concept node
    G.add_node(concept, type="concept")
    
    # Add component nodes
    for component in components:
        G.add_node(component, type="component")
        # Connect components to central concept
        G.add_edge(concept, component, type="has_component", style="dashed")
    
    # Add relation edges between components
    for relation in relations:
        source = relation.get("source", "")
        target = relation.get("target", "")
        rel_type = relation.get("type", "related")
        
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target, type=rel_type, style="solid")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Position nodes - concept at center, components in circle
    pos = {}
    
    # Central concept
    pos[concept] = np.array([0, 0])
    
    # Components in a circle around the concept
    if components:
        angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False)
        radius = 5
        for i, component in enumerate(components):
            pos[component] = np.array([
                radius * np.cos(angles[i]),
                radius * np.sin(angles[i])
            ])
    
    # Node colors by type
    node_colors = []
    for node in G.nodes:
        if node == concept:
            node_colors.append('gold')
        elif G.nodes[node].get("type") == "component":
            node_colors.append('skyblue')
        else:
            node_colors.append('lightgray')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        alpha=0.8,
        node_size=[1500 if node == concept else 1000 for node in G.nodes]
    )
    
    # Draw different edge types with different styles
    # Concept to component edges (dashed)
    concept_edges = [(u, v) for u, v in G.edges if G.edges[u, v].get("style") == "dashed"]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=concept_edges,
        edge_color='gray',
        style='dashed',
        alpha=0.5,
        width=1,
        arrowsize=10
    )
    
    # Component relation edges (solid, colored by type)
    relation_edges = [(u, v) for u, v in G.edges if G.edges[u, v].get("style") == "solid"]
    
    # Get unique relation types
    relation_types = {G.edges[edge].get("type", "related") for edge in relation_edges}
    relation_colors = plt.cm.tab20(np.linspace(0, 1, max(len(relation_types), 1)))
    relation_type_to_color = {t: relation_colors[i] for i, t in enumerate(relation_types)}
    
    # Group edges by relation type
    for rel_type, color in relation_type_to_color.items():
        type_edges = [(u, v) for u, v in relation_edges if G.edges[u, v].get("type") == rel_type]
        
        if type_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=type_edges,
                edge_color=[color],
                alpha=0.8,
                width=2,
                arrowsize=15,
                connectionstyle='arc3,rad=0.1'
            )
    
    # Add labels to all nodes
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold'
    )
    
    # Add edge labels for relation types
    edge_labels = {(u, v): G.edges[u, v].get("type", "") 
                   for u, v in relation_edges}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8
    )
    
    # Create legend for relation types
    relation_patches = [plt.Line2D([0], [0], color=color, 
                                 lw=2, label=rel_type)
                      for rel_type, color in relation_type_to_color.items()]
    
    # Add legend
    plt.legend(handles=relation_patches, 
               title="Relation Types",
               loc='upper right', 
               bbox_to_anchor=(1.15, 1))
    
    # Set title and remove axis
    plt.title(title or f"Complex Structure: {concept}")
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Complex structure visualization saved to {output_file}")
    else:
        plt.show()


def visualize_counterfactual_comparison(
    original_entities: List[Dict[str, Any]],
    original_relations: List[Dict[str, Any]],
    counterfactual_entities: List[Dict[str, Any]],
    counterfactual_relations: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 10)
) -> None:
    """
    Visualize a comparison between original and counterfactual knowledge graphs.
    
    Args:
        original_entities: List of entities in the original graph
        original_relations: List of relations in the original graph
        counterfactual_entities: List of entities in the counterfactual graph
        counterfactual_relations: List of relations in the counterfactual graph
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    if not original_entities or not original_relations or not counterfactual_entities or not counterfactual_relations:
        print("Missing data for counterfactual comparison")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Function to create and draw graph
    def create_and_draw_graph(ax, entities, relations, title_text):
        G = nx.DiGraph()
        
        # Add nodes (entities)
        entity_types = {}
        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("type", 0)
            
            if not name:
                continue
                
            G.add_node(name, type=entity_type)
            entity_types[name] = entity_type
        
        # Add edges (relations)
        for relation in relations:
            source = relation.get("source", "")
            target = relation.get("target", "")
            rel_type = relation.get("relation_type", 0)
            
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, type=rel_type)
        
        # Position nodes
        pos = nx.spring_layout(G, k=0.3, seed=42)
        
        # Node colors by type
        unique_types = sorted(set(entity_types.values()))
        color_map = plt.cm.tab10(np.linspace(0, 1, max(len(unique_types), 1)))
        node_colors = [color_map[unique_types.index(entity_types[node])] for node in G.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            node_color=node_colors,
            alpha=0.8,
            node_size=800
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            ax=ax,
            alpha=0.7,
            width=1.5,
            arrowsize=15,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Add labels
        nx.draw_networkx_labels(
            G, pos,
            ax=ax,
            font_size=9,
            font_weight='bold'
        )
        
        # Set title and remove axis
        ax.set_title(title_text)
        ax.axis('off')
        
        return G
    
    # Draw original graph
    original_graph = create_and_draw_graph(
        ax1, 
        original_entities, 
        original_relations, 
        "Original Knowledge Graph"
    )
    
    # Draw counterfactual graph
    counterfactual_graph = create_and_draw_graph(
        ax2, 
        counterfactual_entities, 
        counterfactual_relations, 
        "Counterfactual Knowledge Graph"
    )
    
    # Add overall title
    fig.suptitle(title or "Counterfactual Knowledge Graph Comparison", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Counterfactual comparison saved to {output_file}")
    else:
        plt.show()


def visualize_inductive_reasoning(
    initial_graph: Dict[str, List[Dict[str, Any]]],
    inferred_entities: List[Dict[str, Any]],
    inferred_relations: List[Dict[str, Any]],
    reasoning_steps: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Visualize the process of inductive reasoning through a knowledge graph.
    
    Args:
        initial_graph: Dictionary with 'entities' and 'relations' of the initial graph
        inferred_entities: List of entities inferred through reasoning
        inferred_relations: List of relations inferred through reasoning
        reasoning_steps: List of steps in the reasoning process
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    if not initial_graph or not reasoning_steps:
        print("Missing data for inductive reasoning visualization")
        return
    
    # Extract initial entities and relations
    initial_entities = initial_graph.get("entities", [])
    initial_relations = initial_graph.get("relations", [])
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    
    # Calculate grid size based on number of reasoning steps
    num_steps = len(reasoning_steps) + 1  # +1 for final result
    grid_size = int(np.ceil(np.sqrt(num_steps)))
    
    # Create subplot for initial graph
    ax_initial = plt.subplot(grid_size, grid_size, 1)
    
    # Function to create and draw a graph
    def draw_graph(ax, entities, relations, title_text):
        G = nx.DiGraph()
        
        # Add nodes (entities)
        for entity in entities:
            name = entity.get("name", "")
            if name:
                G.add_node(name, **entity)
        
        # Add edges (relations)
        for relation in relations:
            source = relation.get("source", "")
            target = relation.get("target", "")
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, **relation)
        
        # Position nodes
        pos = nx.spring_layout(G, k=0.3, seed=42)
        
        # Node colors: initial (blue), inferred (green)
        node_colors = []
        for node in G.nodes:
            if any(e.get("name") == node for e in initial_entities):
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')
        
        # Edge colors: initial (blue), inferred (green)
        edge_colors = []
        for u, v in G.edges:
            if any(r.get("source") == u and r.get("target") == v for r in initial_relations):
                edge_colors.append('royalblue')
            else:
                edge_colors.append('green')
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            alpha=0.8,
            node_size=700
        )
        
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            alpha=0.7,
            width=1.5,
            arrowsize=12,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Add labels
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=8
        )
        
        # Set title and remove axis
        ax.set_title(title_text)
        ax.axis('off')
        
        return G
    
    # Draw initial graph
    draw_graph(
        ax_initial,
        initial_entities,
        initial_relations,
        "Initial Knowledge Graph"
    )
    
    # Draw each reasoning step
    current_entities = list(initial_entities)
    current_relations = list(initial_relations)
    
    for i, step in enumerate(reasoning_steps):
        # Create subplot for this step
        ax_step = plt.subplot(grid_size, grid_size, i + 2)
        
        # Update entities and relations based on step
        step_entities = step.get("entities_added", [])
        step_relations = step.get("relations_added", [])
        
        current_entities.extend(step_entities)
        current_relations.extend(step_relations)
        
        # Draw graph for this step
        draw_graph(
            ax_step,
            current_entities,
            current_relations,
            f"Step {i+1}: {step.get('description', '')}"
        )
    
    # Draw final result with all inferences
    if len(reasoning_steps) < grid_size * grid_size:
        ax_final = plt.subplot(grid_size, grid_size, len(reasoning_steps) + 2)
        
        # Combine all entities and relations
        all_entities = list(initial_entities) + inferred_entities
        all_relations = list(initial_relations) + inferred_relations
        
        # Draw final graph
        draw_graph(
            ax_final,
            all_entities,
            all_relations,
            "Final Knowledge Graph with Inferences"
        )
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue',
                  markersize=10, label='Initial Entity'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                  markersize=10, label='Inferred Entity'),
        plt.Line2D([0], [0], color='royalblue', lw=2, label='Initial Relation'),
        plt.Line2D([0], [0], color='green', lw=2, label='Inferred Relation')
    ]
    
    # Add legend in a separate subplot
    ax_legend = plt.subplot(grid_size, grid_size, grid_size * grid_size)
    ax_legend.axis('off')
    ax_legend.legend(handles=legend_elements, loc='center')
    
    # Add overall title
    fig.suptitle(title or "Inductive Reasoning Process", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Inductive reasoning visualization saved to {output_file}")
    else:
        plt.show()