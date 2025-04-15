import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # Set level to DEBUG
logger = logging.getLogger(__name__)

# Import the summary iterator
from tensorflow.python.summary.summary_iterator import summary_iterator

# --- Helper Functions ---

def load_tensorboard_logs(log_dir):
    """
    Loads scalar data from TensorBoard event files using summary_iterator.
    """
    scalar_data = defaultdict(lambda: defaultdict(list)) # {run: {tag: [(step, value)]}}
    runs = {} # {run_name: path}

    # Find subdirectories (train, validation) which represent different runs/phases
    try:
        for item_name in os.listdir(log_dir):
            item_path = os.path.join(log_dir, item_name)
            if os.path.isdir(item_path):
                # Check if this directory contains event files
                event_files_in_subdir = [f for f in os.listdir(item_path) if f.startswith("events.out.tfevents")]
                if event_files_in_subdir:
                    runs[item_name] = item_path # Treat subdir as a run
            elif item_name.startswith("events.out.tfevents"):
                 # Found event file directly in log_dir, treat as 'root' run
                 if 'root' not in runs:
                     runs['root'] = log_dir

        if not runs:
             logger.error(f"No event files found in {log_dir} or its immediate subdirectories containing event files.")
             return None

        logger.info(f"Found runs to process: {list(runs.keys())}")

        for run_name, run_path in runs.items():
            event_files = [os.path.join(run_path, f) for f in os.listdir(run_path) if f.startswith("events.out.tfevents")]
            if not event_files:
                 logger.warning(f"No event files found in run directory: {run_path} (should not happen based on initial check)")
                 continue

            logger.info(f"Processing run '{run_name}' from {len(event_files)} event file(s)...")

            processed_tags = set() # Keep track of tags processed in this run

            # Iterate through all event files in the run directory
            for event_file in event_files:
                logger.debug(f"Processing event file: {event_file}")
                try:
                    record_count = 0
                    scalar_count = 0
                    for event in summary_iterator(event_file):
                        record_count += 1
                        # logger.debug(f"Read event: step={event.step}, wall_time={event.wall_time}") # Log event details
                        if event.summary:
                            # logger.debug(f"  Event has summary with {len(event.summary.value)} values.")
                            for value in event.summary.value:
                                # logger.debug(f"    Value tag: {value.tag}, metadata: {value.metadata.summary_description}") # Log tag and description
                                if value.HasField('simple_value'):
                                    scalar_count += 1
                                    # logger.info(f"    Found scalar: tag='{value.tag}', step={event.step}, value={value.simple_value}") # Log success
                                    tag = value.tag
                                    step = event.step
                                    val = value.simple_value
                                    scalar_data[run_name][tag].append((step, val))
                                    processed_tags.add(tag)
                                # else:
                                #      # Log what kind of value it *is* if not simple_value
                                #      value_type = value.WhichOneof('value')
                                #      logger.debug(f"    Skipping non-scalar value: tag='{value.tag}', type='{value_type}'")
                        # else:
                        #     logger.debug("  Event has no summary.")
                    logger.debug(f"Finished processing {event_file}. Read {record_count} records, found {scalar_count} scalars.")
                except Exception as e:
                    logger.error(f"Error processing event file {event_file}: {e}", exc_info=True) # Log full traceback on error

            if not processed_tags:
                 logger.warning(f"No scalar data found in any event files for run '{run_name}' in {run_path}")
                 continue # Skip sorting if no data

            # Sort data by step for each tag after processing all files for the run
            logger.info(f"Sorting data for run '{run_name}'...")
            for tag in scalar_data[run_name]:
                # Deduplicate steps just in case (though unlikely with single writer per run)
                unique_steps = {}
                for step, val in scalar_data[run_name][tag]:
                    if step not in unique_steps:
                         unique_steps[step] = val
                # Sort by step
                sorted_data = sorted(unique_steps.items())
                scalar_data[run_name][tag] = sorted_data # Replace with sorted, unique list of tuples

        if not scalar_data:
             logger.warning("No scalar data extracted from any runs.")
             return None

        return scalar_data

    except FileNotFoundError:
        logger.error(f"Log directory not found: {log_dir}")
        return None
    except Exception as e:
        logger.error(f"Error loading TensorBoard logs from {log_dir}: {e}", exc_info=True)
        return None


def convert_to_dataframe(scalar_data):
    """
    Converts the loaded scalar data into a Pandas DataFrame.
    """
    if not scalar_data:
        return None

    all_dfs = []
    for run_name, tags_data in scalar_data.items():
        for tag, values in tags_data.items():
            if values: # Ensure there is data
                steps, vals = zip(*values)
                df = pd.DataFrame({'step': steps, 'value': vals})
                df['tag'] = tag
                df['run'] = run_name
                all_dfs.append(df)

    if not all_dfs:
        logger.warning("No scalar data found to convert to DataFrame.")
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Created DataFrame with {len(full_df)} rows and columns: {full_df.columns.tolist()}")
    return full_df

# --- Plotting Functions ---

def plot_metric_evolution(df, metrics_to_plot, output_dir='analysis_plots'):
    """
    Plots the evolution of specified metrics over steps for different runs.
    """
    if df is None or df.empty:
        logger.warning("DataFrame is empty, skipping plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        metric_df = df[df['tag'] == metric]

        if metric_df.empty:
            logger.warning(f"No data found for metric '{metric}', skipping plot.")
            continue

        sns.lineplot(data=metric_df, x='step', y='value', hue='run', marker='o', markersize=4)
        plt.title(f'Evolution of {metric}')
        plt.xlabel('Global Step')
        plt.ylabel('Value')
        plt.legend(title='Run')
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{metric.replace('/', '_')}_evolution.png")
        plt.savefig(plot_filename)
        plt.close()
        logger.info(f"Saved plot: {plot_filename}")

# --- Correlation Analysis Functions (Placeholder) ---

def analyze_correlations(df, output_dir='analysis_plots'):
    """
    Calculates and potentially visualizes correlations between metrics.
    (Placeholder implementation)
    """
    if df is None or df.empty:
        logger.warning("DataFrame is empty, skipping correlation analysis.")
        return

    # Pivot table to get metrics as columns (might need alignment by step)
    # This requires careful handling of steps and runs.
    # Example (simplified, assumes steps align across metrics/runs):
    try:
        pivot_df = df.pivot_table(index='step', columns=['run', 'tag'], values='value')
        # Flatten multi-index columns if desired: pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

        correlation_matrix = pivot_df.corr()

        plt.figure(figsize=(15, 12))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f") # Annot=True can be too crowded
        plt.title('Metric Correlation Matrix (Approximate - Step Aligned)')
        plt.tight_layout()
        corr_filename = os.path.join(output_dir, "metric_correlations.png")
        plt.savefig(corr_filename)
        plt.close()
        logger.info(f"Saved correlation heatmap: {corr_filename}")
        # print("\nCorrelation Matrix:")
        # print(correlation_matrix)

    except Exception as e:
         logger.error(f"Could not generate correlation matrix (might be due to misaligned steps or data issues): {e}")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TensorBoard logs for RKM.")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to the specific TensorBoard log directory (e.g., logs/rkm_baseline/20250415-040941)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_plots",
        help="Directory to save generated plots."
    )
    args = parser.parse_args()

    logger.info(f"Loading logs from: {args.log_dir}")
    scalar_data = load_tensorboard_logs(args.log_dir)

    if scalar_data:
        logger.info("Converting logs to DataFrame...")
        df = convert_to_dataframe(scalar_data)

        if df is not None and not df.empty:
            # Define which metrics to plot
            # Adjust based on the actual tags found in your logs
            metrics_to_plot = [
                'loss/batch_internal',
                'loss/batch_monad',
                'loss/batch_total_observed',
                'metrics/ce_loss',
                'metrics/entropy_penalty',
                'metrics/dispersion_penalty',
                'metrics/alignment_penalty',
                'metrics/observer_alignment_mean',
                'metrics/gamma_mean',
                'metrics/observer_state_norm',
                'metrics/prime_emb_norm',
                'metrics/phase_emb_norm',
                'monad/symbolic_entropy',
                'monad/resonance',
                'monad/parity_even',
                'monad/is_collapsed'
                # Add epoch losses/metrics if logged and desired
                # 'epoch_loss_internal', 'epoch_loss_total_observed', etc.
            ]

            logger.info("Generating metric evolution plots...")
            plot_metric_evolution(df, metrics_to_plot, args.output_dir)

            logger.info("Performing correlation analysis...")
            analyze_correlations(df, args.output_dir)

            logger.info(f"Analysis complete. Plots saved to: {args.output_dir}")
        else:
            logger.error("Failed to create DataFrame from logs.")
    else:
        logger.error("Failed to load scalar data from logs.")