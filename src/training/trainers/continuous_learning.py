"""
Continuous Learning Framework for Semantic Resonance Model.

This module provides utilities for continuous model improvement through
ongoing feedback and adaptive training strategies.
"""

import os
import json
import time
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from transformers import get_scheduler

from src.data.dialogue_dataset import DialogueDataset
from src.model.semantic_resonance_model_with_extensions import SemanticResonanceModelWithExtensions
from src.model.extensions.memory.knowledge_graph_extension import KnowledgeGraphExtension
from src.model.extensions.extension_config import ExtensionConfig
from src.utils.logging import setup_logger


logger = setup_logger("continuous_learning")


@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning."""
    # General settings
    learning_mode: str = "adaptive"  # "adaptive", "scheduled", "feedback_driven"
    max_samples_per_update: int = 500
    min_samples_for_update: int = 50
    
    # Memory settings
    memory_buffer_size: int = 2000
    memory_priority: str = "recency"  # "recency", "importance", "diversity"
    
    # Learning thresholds
    improvement_threshold: float = 0.05  # Required improvement to accept update
    staleness_threshold: int = 5  # Number of updates without improvement before forced update
    
    # Validation settings
    validation_frequency: int = 1000  # Validate every N samples
    validation_samples: int = 100  # Number of samples to use for validation
    
    # Persistence settings
    save_history: bool = True
    history_dir: str = "learning_history"
    max_history_entries: int = 100
    
    # Learning rates and optimization
    base_learning_rate: float = 5e-5
    feedback_learning_rate: float = 1e-4
    adapt_learning_rate: bool = True
    
    # Mixture settings
    mix_with_base_data: bool = True
    base_data_ratio: float = 0.3  # Ratio of base data to include in update batches
    
    # Extension usage
    use_memory_extension: bool = True
    knowledge_persistence: bool = True
    quantum_adaptation: bool = True


class LearningEntry:
    """Represents a single learning interaction or feedback instance."""
    
    def __init__(self, entry_type: str, data: Dict, timestamp: Optional[float] = None,
                 importance: float = 1.0, metadata: Optional[Dict] = None):
        """
        Initialize a learning entry.
        
        Args:
            entry_type: Type of entry (feedback, conversation, correction, etc.)
            data: The actual data (conversation, feedback, etc.)
            timestamp: Time of entry creation (default: current time)
            importance: Importance score (1.0 = normal, higher = more important)
            metadata: Additional information about the entry
        """
        self.entry_type = entry_type
        self.data = data
        self.timestamp = timestamp or time.time()
        self.importance = importance
        self.metadata = metadata or {}
        self.used_count = 0
        self.performance_impact = None  # To be set after evaluation
    
    def mark_used(self):
        """Mark this entry as used in training."""
        self.used_count += 1
    
    def set_performance_impact(self, impact: float):
        """Set the performance impact of this entry."""
        self.performance_impact = impact
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "entry_type": self.entry_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "metadata": self.metadata,
            "used_count": self.used_count,
            "performance_impact": self.performance_impact
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create an entry from a dictionary."""
        entry = cls(
            entry_type=data["entry_type"],
            data=data["data"],
            timestamp=data["timestamp"],
            importance=data["importance"],
            metadata=data.get("metadata", {})
        )
        entry.used_count = data.get("used_count", 0)
        entry.performance_impact = data.get("performance_impact")
        return entry


class LearningHistory:
    """Manages the history of learning interactions and feedback."""
    
    def __init__(self, config: ContinuousLearningConfig):
        """
        Initialize the learning history.
        
        Args:
            config: Configuration for learning history management
        """
        self.config = config
        self.entries = deque(maxlen=config.memory_buffer_size)
        self.performance_history = []
        self.update_history = []
    
    def add_entry(self, entry: LearningEntry) -> None:
        """
        Add a new entry to the history.
        
        Args:
            entry: Learning entry to add
        """
        self.entries.append(entry)
    
    def get_entries(self, count: Optional[int] = None, 
                   entry_type: Optional[str] = None,
                   min_importance: float = 0.0,
                   sort_by: str = "recency") -> List[LearningEntry]:
        """
        Get entries from history based on criteria.
        
        Args:
            count: Maximum number of entries to return
            entry_type: Filter by entry type
            min_importance: Minimum importance score
            sort_by: How to sort entries ("recency", "importance", "impact")
            
        Returns:
            List of matching entries
        """
        # Filter entries
        filtered = list(self.entries)
        
        if entry_type:
            filtered = [e for e in filtered if e.entry_type == entry_type]
        
        if min_importance > 0:
            filtered = [e for e in filtered if e.importance >= min_importance]
        
        # Sort entries
        if sort_by == "recency":
            filtered.sort(key=lambda e: e.timestamp, reverse=True)
        elif sort_by == "importance":
            filtered.sort(key=lambda e: e.importance, reverse=True)
        elif sort_by == "impact":
            # Sort by performance impact if available, otherwise by importance
            filtered.sort(key=lambda e: e.performance_impact if e.performance_impact is not None 
                                                              else e.importance, 
                         reverse=True)
        elif sort_by == "least_used":
            filtered.sort(key=lambda e: e.used_count)
        
        # Return requested number of entries
        if count:
            return filtered[:count]
        return filtered
    
    def record_performance(self, metrics: Dict[str, float], 
                          update_id: Optional[int] = None) -> None:
        """
        Record model performance at a point in time.
        
        Args:
            metrics: Performance metrics
            update_id: Optional ID of the update that produced these metrics
        """
        entry = {
            "timestamp": time.time(),
            "metrics": metrics,
            "update_id": update_id
        }
        self.performance_history.append(entry)
    
    def record_update(self, update_info: Dict[str, Any]) -> int:
        """
        Record information about a model update.
        
        Args:
            update_info: Update information
            
        Returns:
            Update ID
        """
        update_id = len(self.update_history)
        entry = {
            "update_id": update_id,
            "timestamp": time.time(),
            **update_info
        }
        self.update_history.append(entry)
        return update_id
    
    def get_performance_trend(self, metric: str, 
                             window: Optional[int] = None) -> List[float]:
        """
        Get performance trend for a specific metric.
        
        Args:
            metric: Metric to track
            window: Number of most recent entries to include
            
        Returns:
            List of metric values
        """
        values = [entry["metrics"].get(metric) for entry in self.performance_history 
                 if metric in entry["metrics"]]
        
        if window and len(values) > window:
            return values[-window:]
        return values
    
    def detect_improvement(self, metric: str, window: int = 3) -> bool:
        """
        Detect if there has been improvement in a metric recently.
        
        Args:
            metric: Metric to check
            window: Number of entries to consider
            
        Returns:
            True if improvement detected, False otherwise
        """
        values = self.get_performance_trend(metric)
        
        if len(values) < window + 1:
            return False
        
        # Compare average of last 'window' values to previous value
        recent_avg = sum(values[-window:]) / window
        previous = values[-(window+1)]
        
        # For metrics where lower is better (like loss, perplexity)
        if metric in ["loss", "perplexity"]:
            return recent_avg < previous * (1 - self.config.improvement_threshold)
        
        # For metrics where higher is better (like accuracy)
        return recent_avg > previous * (1 + self.config.improvement_threshold)
    
    def save(self, directory: str) -> None:
        """
        Save learning history to disk.
        
        Args:
            directory: Directory to save history to
        """
        if not self.config.save_history:
            return
        
        os.makedirs(directory, exist_ok=True)
        
        # Save entries
        entries_data = [e.to_dict() for e in self.entries]
        with open(os.path.join(directory, "learning_entries.json"), "w") as f:
            json.dump(entries_data, f, indent=2)
        
        # Save performance history
        with open(os.path.join(directory, "performance_history.json"), "w") as f:
            json.dump(self.performance_history, f, indent=2)
        
        # Save update history
        with open(os.path.join(directory, "update_history.json"), "w") as f:
            json.dump(self.update_history, f, indent=2)
    
    def load(self, directory: str) -> bool:
        """
        Load learning history from disk.
        
        Args:
            directory: Directory to load history from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load entries
            entries_path = os.path.join(directory, "learning_entries.json")
            if os.path.exists(entries_path):
                with open(entries_path, "r") as f:
                    entries_data = json.load(f)
                self.entries = deque(
                    [LearningEntry.from_dict(e) for e in entries_data],
                    maxlen=self.config.memory_buffer_size
                )
            
            # Load performance history
            perf_path = os.path.join(directory, "performance_history.json")
            if os.path.exists(perf_path):
                with open(perf_path, "r") as f:
                    self.performance_history = json.load(f)
            
            # Load update history
            update_path = os.path.join(directory, "update_history.json")
            if os.path.exists(update_path):
                with open(update_path, "r") as f:
                    self.update_history = json.load(f)
                    
            return True
        except Exception as e:
            logger.error(f"Failed to load learning history: {str(e)}")
            return False


class ContinuousLearningManager:
    """
    Manages the continuous learning process for a model.
    
    This class is responsible for:
    1. Tracking learning progress
    2. Deciding when to update the model
    3. Managing the learning history
    4. Integrating user feedback
    5. Handling the ongoing improvement cycle
    """
    
    def __init__(self, model: SemanticResonanceModelWithExtensions,
                 tokenizer, base_dataloader: Optional[DataLoader] = None,
                 config: Optional[ContinuousLearningConfig] = None,
                 output_dir: str = "continuous_learning"):
        """
        Initialize the continuous learning manager.
        
        Args:
            model: The model to continuously improve
            tokenizer: Tokenizer for processing text
            base_dataloader: DataLoader for base training data
            config: Configuration for continuous learning
            output_dir: Directory for outputs
        """
        self.model = model
        self.tokenizer = tokenizer
        self.base_dataloader = base_dataloader
        self.config = config or ContinuousLearningConfig()
        self.output_dir = output_dir
        self.history = LearningHistory(self.config)
        self.current_update_id = 0
        self.last_update_time = None
        self.staleness_counter = 0
        self.best_metrics = {}
        self.optimizer = None
        self.scheduler = None
        
        # Create required directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize optimizer
        self._initialize_optimizer()
        
        # Save initial config
        self._save_config()
    
    def _initialize_optimizer(self):
        """Initialize or reinitialize the optimizer."""
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.base_learning_rate,
            weight_decay=0.01
        )
        
        # Create learning rate scheduler
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=100,
            num_training_steps=10000  # Will be updated during training
        )
    
    def _save_config(self):
        """Save the current configuration."""
        config_path = os.path.join(self.output_dir, "continuous_learning_config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def add_feedback(self, feedback_data: Dict, 
                    importance: float = 1.0,
                    metadata: Optional[Dict] = None) -> None:
        """
        Add user feedback for learning.
        
        Args:
            feedback_data: The feedback data (conversation with corrections)
            importance: Importance score for this feedback
            metadata: Additional information about the feedback
        """
        entry = LearningEntry(
            entry_type="feedback",
            data=feedback_data,
            importance=importance,
            metadata=metadata
        )
        self.history.add_entry(entry)
        
        # Check if we should update the model
        if (len(self.history.get_entries(entry_type="feedback")) >= 
                self.config.min_samples_for_update):
            self._check_for_update()
    
    def add_conversation(self, conversation_data: Dict,
                        importance: float = 1.0,
                        metadata: Optional[Dict] = None) -> None:
        """
        Add a conversation for learning.
        
        Args:
            conversation_data: The conversation data
            importance: Importance score for this conversation
            metadata: Additional information about the conversation
        """
        entry = LearningEntry(
            entry_type="conversation",
            data=conversation_data,
            importance=importance,
            metadata=metadata
        )
        self.history.add_entry(entry)
    
    def _check_for_update(self) -> bool:
        """
        Check if the model should be updated based on current status.
        
        Returns:
            True if update should be performed, False otherwise
        """
        # Get feedback entries
        feedback_entries = self.history.get_entries(entry_type="feedback")
        
        if len(feedback_entries) < self.config.min_samples_for_update:
            return False
        
        if self.config.learning_mode == "adaptive":
            # Update if we have enough samples or if we've been stale for too long
            if (len(feedback_entries) >= self.config.max_samples_per_update or
                    self.staleness_counter >= self.config.staleness_threshold):
                return True
            return False
            
        elif self.config.learning_mode == "scheduled":
            # Check if enough time has passed since last update
            if not self.last_update_time:
                return True
                
            time_since_last = time.time() - self.last_update_time
            # Update daily (86400 seconds)
            return time_since_last >= 86400
            
        elif self.config.learning_mode == "feedback_driven":
            # Update immediately with each feedback
            return True
            
        return False
    
    def update_model(self, validation_dataloader: Optional[DataLoader] = None,
                    max_steps: int = 1000) -> Dict[str, float]:
        """
        Update the model based on accumulated feedback and learning history.
        
        Args:
            validation_dataloader: DataLoader for validation
            max_steps: Maximum training steps
            
        Returns:
            Dictionary of metrics from the update
        """
        # Prepare data for update
        update_dataset = self._prepare_update_dataset()
        if not update_dataset:
            logger.warning("No data available for update")
            return {}
        
        # Create dataloader
        update_dataloader = DataLoader(
            update_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=lambda x: {
                "input_ids": torch.stack([item["input_ids"] for item in x]),
                "attention_mask": torch.stack([item["attention_mask"] for item in x]),
                "labels": torch.stack([item["labels"] for item in x])
            }
        )
        
        # Update the model
        update_info = self._train_update(
            update_dataloader, 
            validation_dataloader,
            max_steps=max_steps
        )
        
        # Record the update
        self.current_update_id = self.history.record_update(update_info)
        self.last_update_time = time.time()
        
        # Reset staleness counter if improvement detected
        if update_info.get("improved", False):
            self.staleness_counter = 0
        else:
            self.staleness_counter += 1
        
        # Mark used entries
        for entry_id in update_info.get("entries_used", []):
            self.history.entries[entry_id].mark_used()
        
        # Save history
        self.history.save(os.path.join(self.output_dir, self.config.history_dir))
        
        # Propagate knowledge to the model's memory extension if available
        if (self.config.use_memory_extension and 
                self.config.knowledge_persistence and
                self.model.memory_extension is not None):
            self._update_memory_extension(update_info.get("entries_used", []))
        
        return update_info
    
    def _prepare_update_dataset(self) -> Optional[torch.utils.data.Dataset]:
        """
        Prepare a dataset for model update based on learning history.
        
        Returns:
            Dataset for update or None if no data is available
        """
        # Get entries for update
        feedback_entries = self.history.get_entries(
            count=self.config.max_samples_per_update,
            entry_type="feedback",
            sort_by=self.config.memory_priority
        )
        
        if not feedback_entries:
            return None
        
        # Convert entries to format expected by DialogueDataset
        learning_samples = []
        for entry in feedback_entries:
            # Handle different feedback formats
            if "conversations" in entry.data:
                # ShareGPT format
                learning_samples.append(entry.data["conversations"])
            elif "messages" in entry.data:
                # ChatML format
                messages = entry.data["messages"]
                conversation = []
                for msg in messages:
                    conversation.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
                learning_samples.append(conversation)
            elif isinstance(entry.data, list):
                # Direct conversation list
                learning_samples.append(entry.data)
        
        # Create a dialogue dataset
        update_dataset = DialogueDataset(
            tokenizer=self.tokenizer,
            max_length=1024,
            learning_samples=learning_samples,
            return_tensors=True
        )
        
        # Mix with base data if configured
        if self.config.mix_with_base_data and self.base_dataloader:
            # Sample a portion of base data
            base_samples = []
            sample_size = int(len(update_dataset) * self.config.base_data_ratio)
            
            if sample_size > 0:
                # Extract samples from base dataloader
                for i, batch in enumerate(self.base_dataloader):
                    if i * self.base_dataloader.batch_size >= sample_size:
                        break
                    
                    # Unpack batch to individual samples
                    for j in range(min(len(batch["input_ids"]), 
                                      sample_size - i * self.base_dataloader.batch_size)):
                        base_samples.append({
                            "input_ids": batch["input_ids"][j],
                            "attention_mask": batch["attention_mask"][j],
                            "labels": batch["labels"][j]
                        })
                
                # Create dataset for base samples
                base_dataset = torch.utils.data.TensorDataset(
                    torch.stack([s["input_ids"] for s in base_samples]),
                    torch.stack([s["attention_mask"] for s in base_samples]),
                    torch.stack([s["labels"] for s in base_samples])
                )
                
                # Combine datasets
                combined_dataset = ConcatDataset([update_dataset, base_dataset])
                return combined_dataset
        
        return update_dataset
    
    def _train_update(self, train_dataloader: DataLoader,
                     val_dataloader: Optional[DataLoader] = None,
                     max_steps: int = 1000) -> Dict[str, Any]:
        """
        Train the model for an update.
        
        Args:
            train_dataloader: DataLoader for training
            val_dataloader: DataLoader for validation
            max_steps: Maximum training steps
            
        Returns:
            Dictionary with update information
        """
        device = next(self.model.parameters()).device
        self.model.train()
        
        # Training hyperparameters
        num_steps = min(max_steps, len(train_dataloader))
        
        # Update learning rate scheduler for number of steps
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=max(10, num_steps // 10),
            num_training_steps=num_steps
        )
        
        # Training loop
        total_loss = 0
        steps = 0
        entries_used = list(range(len(self.history.entries)))
        
        for batch in train_dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            steps += 1
            if steps >= max_steps:
                break
        
        # Calculate average loss
        avg_loss = total_loss / steps if steps > 0 else float('inf')
        
        # Validation
        val_metrics = {}
        if val_dataloader:
            val_metrics = self.evaluate(val_dataloader)
            
            # Record performance
            metrics = {"loss": avg_loss, **val_metrics}
            self.history.record_performance(metrics, update_id=self.current_update_id)
            
            # Check for improvement
            improved = False
            if "perplexity" in val_metrics:
                current_perplexity = val_metrics["perplexity"]
                if ("perplexity" not in self.best_metrics or 
                        current_perplexity < self.best_metrics["perplexity"] * 
                        (1 - self.config.improvement_threshold)):
                    improved = True
                    self.best_metrics["perplexity"] = current_perplexity
        
        # Create update info
        update_info = {
            "loss": avg_loss,
            "steps": steps,
            "entries_used": entries_used,
            "validation_metrics": val_metrics,
            "improved": improved if val_dataloader else None,
            "timestamp": time.time()
        }
        
        return update_info
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Dictionary of metrics
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
    
    def _update_memory_extension(self, entry_indices: List[int]) -> None:
        """
        Update the model's memory extension with knowledge from entries.
        
        Args:
            entry_indices: Indices of entries to use for updating memory
        """
        if not self.model.memory_extension:
            return
        
        memory_ext = self.model.memory_extension
        
        # Process entries to extract knowledge
        for idx in entry_indices:
            if idx >= len(self.history.entries):
                continue
                
            entry = self.history.entries[idx]
            
            # Skip non-conversation entries
            if entry.entry_type != "conversation" and entry.entry_type != "feedback":
                continue
            
            # Process conversation to extract entities and relations
            conversation = None
            if "conversations" in entry.data:
                conversation = entry.data["conversations"]
            elif "messages" in entry.data:
                conversation = entry.data["messages"]
            elif isinstance(entry.data, list):
                conversation = entry.data
            
            if not conversation:
                continue
            
            # Extract text from conversation
            full_text = ""
            for msg in conversation:
                if isinstance(msg, dict) and "content" in msg:
                    full_text += msg["content"] + " "
                elif isinstance(msg, str):
                    full_text += msg + " "
            
            # Use simple heuristics to extract entities and relations
            # In a real system, this would use NER and relation extraction
            self._extract_knowledge_from_text(full_text, memory_ext)
    
    def _extract_knowledge_from_text(self, text: str, memory_ext: KnowledgeGraphExtension) -> None:
        """
        Extract knowledge from text and add to memory extension.
        
        This is a simplified implementation. In a real system, you would use
        NER and relation extraction models.
        
        Args:
            text: Text to extract knowledge from
            memory_ext: Memory extension to update
        """
        # This is a placeholder for a more sophisticated extraction system
        # In a real implementation, you'd use NLP techniques to extract entities and relations
        
        # For demonstration purposes, we'll just add the text as an entity
        timestamp = int(time.time())
        
        try:
            # Add text as a document entity
            entity_id = memory_ext.add_entity(
                name=f"document_{timestamp}",
                entity_type=1,  # Document type
                attributes={
                    "text": text[:1000],  # Limit text length
                    "timestamp": timestamp,
                    "source": "continuous_learning"
                }
            )
            
            # For simplicity, we're not extracting relations
            # In a real system, you would parse the text for entities and relationships
            
            logger.info(f"Added document entity to memory: {entity_id}")
            
        except Exception as e:
            logger.error(f"Failed to add knowledge to memory: {str(e)}")
    
    def save_state(self, path: Optional[str] = None) -> None:
        """
        Save the continuous learning state.
        
        Args:
            path: Directory to save state to (default: output_dir)
        """
        save_dir = path or self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Save history
        history_dir = os.path.join(save_dir, self.config.history_dir)
        self.history.save(history_dir)
        
        # Save optimizer and scheduler states
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best_metrics": self.best_metrics,
            "current_update_id": self.current_update_id,
            "last_update_time": self.last_update_time,
            "staleness_counter": self.staleness_counter
        }, os.path.join(save_dir, "continuous_learning_state.pt"))
        
        # Save configuration
        self._save_config()
        
        logger.info(f"Saved continuous learning state to {save_dir}")
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """
        Load the continuous learning state.
        
        Args:
            path: Directory to load state from (default: output_dir)
            
        Returns:
            True if successful, False otherwise
        """
        load_dir = path or self.output_dir
        
        try:
            # Load history
            history_dir = os.path.join(load_dir, self.config.history_dir)
            if os.path.exists(history_dir):
                self.history.load(history_dir)
            
            # Load optimizer and scheduler states
            state_path = os.path.join(load_dir, "continuous_learning_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path)
                self.optimizer.load_state_dict(state["optimizer"])
                
                if state.get("scheduler") and self.scheduler:
                    self.scheduler.load_state_dict(state["scheduler"])
                
                self.best_metrics = state.get("best_metrics", {})
                self.current_update_id = state.get("current_update_id", 0)
                self.last_update_time = state.get("last_update_time")
                self.staleness_counter = state.get("staleness_counter", 0)
            
            logger.info(f"Loaded continuous learning state from {load_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load continuous learning state: {str(e)}")
            return False
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the learning process.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "entries_count": len(self.history.entries),
            "feedback_count": len(self.history.get_entries(entry_type="feedback")),
            "conversation_count": len(self.history.get_entries(entry_type="conversation")),
            "updates_count": len(self.history.update_history),
            "last_update_time": self.last_update_time,
            "staleness_counter": self.staleness_counter,
            "best_metrics": self.best_metrics
        }
        
        # Add performance trends if available
        if self.history.performance_history:
            stats["loss_trend"] = self.history.get_performance_trend("loss")
            stats["perplexity_trend"] = self.history.get_performance_trend("perplexity")
        
        return stats