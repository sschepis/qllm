"""
Metrics logging system for the enhanced training system.

This module provides utilities for collecting, storing, and visualizing
training metrics across different stages of model training.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Callable

import torch
import numpy as np


# Get logger
logger = logging.getLogger("quantum_resonance")


class MetricsLogger:
    """
    Logger for training metrics.
    
    This class handles collection, storage, and visualization of metrics
    during model training, with support for TensorBoard and other logging
    backends.
    """
    
    def __init__(
        self,
        log_dir: str = "runs/quantum_resonance",
        logging_steps: int = 10,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        log_to_file: bool = True
    ):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory for storing logs
            logging_steps: Steps between logging
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            log_to_file: Whether to log metrics to file
        """
        self.log_dir = log_dir
        self.logging_steps = logging_steps
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.log_to_file = log_to_file
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics history
        self.metrics_history = {}
        
        # Initialize TensorBoard writer if requested
        self.writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging enabled at {log_dir}")
            except ImportError:
                logger.warning("TensorBoard requested but not available. Install with 'pip install tensorboard'")
                self.use_tensorboard = False
        
        # Initialize Weights & Biases if requested
        if self.use_wandb:
            try:
                import wandb
                if not wandb.api.api_key:
                    logger.warning("Weights & Biases API key not found. Login with 'wandb login'")
                    self.use_wandb = False
                else:
                    wandb_project = os.path.basename(os.path.normpath(log_dir))
                    wandb.init(project=wandb_project, dir=log_dir)
                    logger.info(f"Weights & Biases logging enabled for project {wandb_project}")
            except ImportError:
                logger.warning("Weights & Biases requested but not available. Install with 'pip install wandb'")
                self.use_wandb = False
        
        # Initialize metrics file if requested
        self.metrics_file = None
        if self.log_to_file:
            self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
            logger.info(f"Metrics will be logged to {self.metrics_file}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        epoch: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step
            epoch: Current epoch (optional)
            prefix: Prefix for metric names (e.g., 'train', 'val', 'test')
        """
        if not metrics:
            return
        
        # Format metrics for logging
        formatted_metrics = self._format_metrics(metrics)
        
        # Add epoch info if provided
        if epoch is not None:
            formatted_metrics["epoch"] = epoch
        
        # Add timestamp
        formatted_metrics["timestamp"] = time.time()
        
        # Store in metrics history
        if prefix not in self.metrics_history:
            self.metrics_history[prefix] = []
        
        self.metrics_history[prefix].append({
            "step": step,
            "epoch": epoch,
            "timestamp": formatted_metrics["timestamp"],
            "metrics": {k: v for k, v in formatted_metrics.items() 
                      if k not in ["step", "epoch", "timestamp"]}
        })
        
        # Log to console
        if step % self.logging_steps == 0 or prefix in ["val", "test"]:
            log_str = f"{prefix} step {step}"
            if epoch is not None:
                log_str += f" (epoch {epoch+1})"
            log_str += ": " + ", ".join(f"{k}: {v:.6g}" for k, v in formatted_metrics.items() 
                                     if k not in ["epoch", "timestamp"])
            logger.info(log_str)
        
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            for key, value in formatted_metrics.items():
                if key not in ["epoch", "timestamp"]:
                    tag = f"{prefix}/{key}" if prefix else key
                    self.writer.add_scalar(tag, value, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            import wandb
            wandb_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in formatted_metrics.items()}
            wandb_metrics["step"] = step
            if epoch is not None:
                wandb_metrics["epoch"] = epoch
            wandb.log(wandb_metrics)
        
        # Log to file
        if self.log_to_file and self.metrics_file is not None:
            try:
                with open(self.metrics_file, "a") as f:
                    json_metrics = {
                        "step": step,
                        "prefix": prefix,
                        **{k: v for k, v in formatted_metrics.items()}
                    }
                    f.write(json.dumps(json_metrics) + "\n")
            except Exception as e:
                logger.warning(f"Error writing metrics to file: {e}")
    
    def log_model_predictions(
        self,
        inputs: Union[List[str], List[Dict[str, Any]]],
        predictions: List[str],
        labels: Optional[List[str]] = None,
        step: int = 0,
        prefix: str = "eval"
    ) -> None:
        """
        Log model predictions for analysis.
        
        Args:
            inputs: Input texts or dictionaries
            predictions: Model predictions
            labels: Ground truth labels (optional)
            step: Current step
            prefix: Prefix for logging (e.g., 'eval', 'test')
        """
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            # Create a formatted text table
            table_rows = []
            
            # Add header
            if labels is not None:
                table_rows.append("| Input | Prediction | Label |")
                table_rows.append("| --- | --- | --- |")
            else:
                table_rows.append("| Input | Prediction |")
                table_rows.append("| --- | --- |")
            
            # Add data rows (limit to 10 examples to avoid excessive logging)
            for i in range(min(10, len(inputs))):
                input_text = inputs[i]
                if isinstance(input_text, dict):
                    input_text = input_text.get("text", str(input_text))
                
                input_text = str(input_text).replace("\n", " ")[:100] + "..." if len(str(input_text)) > 100 else str(input_text)
                pred_text = predictions[i].replace("\n", " ")[:100] + "..." if len(predictions[i]) > 100 else predictions[i]
                
                if labels is not None:
                    label_text = labels[i].replace("\n", " ")[:100] + "..." if len(labels[i]) > 100 else labels[i]
                    table_rows.append(f"| {input_text} | {pred_text} | {label_text} |")
                else:
                    table_rows.append(f"| {input_text} | {pred_text} |")
            
            # Log the table
            table_text = "\n".join(table_rows)
            self.writer.add_text(f"{prefix}/predictions", table_text, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            import wandb
            if labels is not None:
                columns = ["input", "prediction", "label"]
                data = [[str(inputs[i]), predictions[i], labels[i]] for i in range(min(20, len(inputs)))]
            else:
                columns = ["input", "prediction"]
                data = [[str(inputs[i]), predictions[i]] for i in range(min(20, len(inputs)))]
            
            wandb.log({f"{prefix}/predictions": wandb.Table(columns=columns, data=data)})
    
    def log_confusion_matrix(
        self,
        confusion_matrix: Union[List[List[int]], np.ndarray],
        class_names: List[str],
        step: int = 0,
        prefix: str = "eval"
    ) -> None:
        """
        Log a confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix as a 2D list or numpy array
            class_names: Names of the classes
            step: Current step
            prefix: Prefix for logging (e.g., 'eval', 'test')
        """
        # Convert to numpy array if needed
        if not isinstance(confusion_matrix, np.ndarray):
            confusion_matrix = np.array(confusion_matrix)
        
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib.colors import LinearSegmentedColormap
                
                fig, ax = plt.figure(figsize=(10, 10)), plt.subplot(111)
                cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#4285f4'])
                
                ax.imshow(confusion_matrix, cmap=cmap)
                
                # Set labels
                ax.set_xticks(np.arange(len(class_names)))
                ax.set_yticks(np.arange(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha="right")
                ax.set_yticklabels(class_names)
                
                # Loop over data and create text annotations
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        text_color = "black" if confusion_matrix[i, j] < confusion_matrix.max() / 2 else "white"
                        ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color=text_color)
                
                ax.set_xlabel('Predicted label')
                ax.set_ylabel('True label')
                ax.set_title('Confusion matrix')
                
                self.writer.add_figure(f"{prefix}/confusion_matrix", fig, step)
            except Exception as e:
                logger.warning(f"Error creating confusion matrix visualization: {e}")
        
        # Log to Weights & Biases
        if self.use_wandb:
            import wandb
            try:
                wandb.log({f"{prefix}/confusion_matrix": wandb.plots.HeatMap(
                    x_labels=class_names,
                    y_labels=class_names,
                    matrix_values=confusion_matrix,
                    show_text=True
                )})
            except Exception as e:
                logger.warning(f"Error logging confusion matrix to wandb: {e}")
    
    def log_learning_rate(
        self,
        lr: float,
        step: int,
        epoch: Optional[int] = None
    ) -> None:
        """
        Log learning rate.
        
        Args:
            lr: Learning rate
            step: Current step
            epoch: Current epoch (optional)
        """
        self.log_metrics({"learning_rate": lr}, step, epoch, prefix="train")
    
    def log_image(
        self,
        image: Union[torch.Tensor, np.ndarray],
        caption: Optional[str] = None,
        step: int = 0,
        prefix: str = "images"
    ) -> None:
        """
        Log an image.
        
        Args:
            image: Image to log
            caption: Image caption
            step: Current step
            prefix: Prefix for logging
        """
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            try:
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                
                # Handle different image formats
                if image.ndim == 4:
                    # Batch of images, log first image
                    image = image[0]
                
                if image.ndim == 3:
                    # Convert image to proper format for TensorBoard
                    if image.shape[0] in [1, 3]:
                        # Image is in CHW format, TensorBoard expects it
                        pass
                    elif image.shape[2] in [1, 3]:
                        # Image is in HWC format, convert to CHW
                        image = np.transpose(image, (2, 0, 1))
                    else:
                        # Unusual format, try to handle gracefully
                        if image.shape[0] > 4:
                            # Likely HWC without proper channel dimension
                            image = np.transpose(image, (2, 0, 1))
                
                # Handle different data types and ranges
                if image.dtype == np.uint8:
                    # Already in 0-255 range
                    pass
                elif image.max() <= 1.0:
                    # Scale from [0, 1] to [0, 255]
                    image = (image * 255).astype(np.uint8)
                
                self.writer.add_image(f"{prefix}/{caption or 'image'}", image, step)
            except Exception as e:
                logger.warning(f"Error logging image: {e}")
        
        # Log to Weights & Biases
        if self.use_wandb:
            import wandb
            try:
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                
                # Handle different image formats for wandb
                if image.ndim == 4:
                    # Batch of images, log first image
                    image = image[0]
                
                # wandb expects HWC format
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    # Image is in CHW format, convert to HWC
                    image = np.transpose(image, (1, 2, 0))
                
                # Handle different data types and ranges
                if image.dtype == np.uint8:
                    # Already in 0-255 range
                    pass
                elif image.max() <= 1.0:
                    # Scale from [0, 1] to [0, 255]
                    image = (image * 255).astype(np.uint8)
                
                wandb.log({f"{prefix}/{caption or 'image'}": wandb.Image(image)})
            except Exception as e:
                logger.warning(f"Error logging image to wandb: {e}")
    
    def get_latest_metric(
        self,
        prefix: str,
        metric_name: str
    ) -> Optional[float]:
        """
        Get the latest value of a specific metric.
        
        Args:
            prefix: Metric prefix (e.g., 'train', 'val', 'test')
            metric_name: Name of the metric
            
        Returns:
            Latest metric value or None if not found
        """
        if prefix not in self.metrics_history or not self.metrics_history[prefix]:
            return None
        
        latest_entry = self.metrics_history[prefix][-1]
        return latest_entry["metrics"].get(metric_name)
    
    def get_metrics_history(
        self,
        prefix: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the history of all metrics.
        
        Args:
            prefix: Optional prefix to filter by
            
        Returns:
            Dictionary of metrics history
        """
        if prefix is not None:
            return {prefix: self.metrics_history.get(prefix, [])}
        return self.metrics_history
    
    def save_metrics_to_file(
        self,
        filepath: Optional[str] = None
    ) -> None:
        """
        Save all metrics history to a file.
        
        Args:
            filepath: Path to save metrics to (default: log_dir/metrics_history.json)
        """
        if filepath is None:
            filepath = os.path.join(self.log_dir, "metrics_history.json")
        
        try:
            with open(filepath, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Metrics history saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics history: {e}")
    
    def close(self) -> None:
        """
        Close all logging resources.
        """
        # Close TensorBoard writer
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()
        
        # Finish wandb run
        if self.use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        
        # Save metrics history
        self.save_metrics_to_file()
    
    def __del__(self) -> None:
        """
        Clean up when object is destroyed.
        """
        try:
            self.close()
        except:
            pass
    
    def _format_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Format metrics for logging.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary of formatted metrics
        """
        formatted = {}
        
        for key, value in metrics.items():
            # Handle various numeric types
            if isinstance(value, (int, float)):
                formatted[key] = float(value)
            elif isinstance(value, torch.Tensor):
                # Convert tensor to float
                try:
                    formatted[key] = float(value.detach().cpu().item())
                except:
                    try:
                        formatted[key] = float(value.detach().cpu().float().mean().item())
                    except:
                        logger.warning(f"Could not convert tensor to float for metric {key}")
                        continue
            elif isinstance(value, np.ndarray):
                # Convert numpy array to float
                try:
                    formatted[key] = float(value.mean())
                except:
                    logger.warning(f"Could not convert numpy array to float for metric {key}")
                    continue
            else:
                # Try to convert to float, skip if not possible
                try:
                    formatted[key] = float(value)
                except:
                    logger.warning(f"Skipping non-numeric metric: {key}")
                    continue
        
        return formatted