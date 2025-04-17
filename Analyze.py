#!/usr/bin/env python3
# results_analysis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Define colors for consistent visualization
COLORS = {
    'minADE': '#1f77b4',  # blue
    'minADEdiv': '#ff7f0e',  # orange
    'GMM_NLL': '#2ca02c',  # green
}

def load_training_logs(results_dir='results'):
    """Load all training log files into DataFrames."""
    logs = {}
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} directory not found.")
        return logs
    
    # Look for training log files with the expected naming pattern
    log_files = list(Path(results_dir).glob('*_training_log.csv'))
    
    if not log_files:
        print(f"No training log files found in {results_dir} directory.")
        return logs
    
    # Load each log file
    for log_file in log_files:
        try:
            # Extract loss function name from filename
            loss_name = log_file.stem.split('_training_log')[0]
            
            # Load CSV into DataFrame
            df = pd.read_csv(log_file)
            
            # Store in dictionary
            logs[loss_name] = df
            print(f"Loaded {log_file} with {len(df)} records")
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return logs

def plot_min_metrics(logs, output_dir='results'):
    """Create plots comparing min_ade and min_fde across epochs for all loss functions."""
    if not logs:
        print("No logs available to plot min metrics.")
        return
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot min_ade for each loss function
    for loss_name, df in logs.items():
        axes[0].plot(df['epoch'], df['min_ade'], label=loss_name, 
                   color=COLORS.get(loss_name, None), linewidth=2)
    
    axes[0].set_title('Minimum ADE Over Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('min_ade')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot min_fde for each loss function
    for loss_name, df in logs.items():
        axes[1].plot(df['epoch'], df['min_fde'], label=loss_name, 
                   color=COLORS.get(loss_name, None), linewidth=2)
    
    axes[1].set_title('Minimum FDE Over Training')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('min_fde')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/min_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved min metrics plot to {output_dir}/min_metrics_comparison.png")

def plot_total_loss(logs, output_dir='results'):
    """Create a plot comparing total_loss across epochs for all loss functions."""
    if not logs:
        print("No logs available to plot total loss.")
        return
    
    plt.figure(figsize=(10, 6))
    
    for loss_name, df in logs.items():
        plt.plot(df['epoch'], df['total_loss'], label=loss_name, 
                 color=COLORS.get(loss_name, None), linewidth=2)
    
    plt.title('Total Loss Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_loss_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved total loss plot to {output_dir}/total_loss_comparison.png")

def plot_current_metrics(logs, output_dir='results'):
    """Create plots comparing current ade and fde across epochs for all loss functions."""
    if not logs:
        print("No logs available to plot current metrics.")
        return
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot ade for each loss function
    for loss_name, df in logs.items():
        axes[0].plot(df['epoch'], df['ade'], label=loss_name, 
                   color=COLORS.get(loss_name, None), linewidth=2)
    
    axes[0].set_title('ADE Per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('ADE')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot fde for each loss function
    for loss_name, df in logs.items():
        axes[1].plot(df['epoch'], df['fde'], label=loss_name, 
                   color=COLORS.get(loss_name, None), linewidth=2)
    
    axes[1].set_title('FDE Per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('FDE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/current_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved current metrics plot to {output_dir}/current_metrics_comparison.png")

def analyze_convergence(logs, output_dir='results'):
    """Analyze convergence rate and create visualization."""
    if not logs:
        print("No logs available to analyze convergence.")
        return
    
    # Define convergence as percentage improvement from first epoch
    convergence_data = []
    
    for loss_name, df in logs.items():
        # Get first epoch values for normalization
        first_ade = df['ade'].iloc[0]
        first_fde = df['fde'].iloc[0]
        
        # Calculate relative improvement for each epoch
        for idx, row in df.iterrows():
            rel_ade_improv = 100 * (first_ade - row['ade']) / first_ade
            rel_fde_improv = 100 * (first_fde - row['fde']) / first_fde
            
            convergence_data.append({
                'epoch': row['epoch'],
                'loss_function': loss_name,
                'rel_ade_improvement': rel_ade_improv,
                'rel_fde_improvement': rel_fde_improv
            })
    
    # Convert to DataFrame
    conv_df = pd.DataFrame(convergence_data)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    for loss_name in logs.keys():
        loss_data = conv_df[conv_df['loss_function'] == loss_name]
        plt.plot(loss_data['epoch'], loss_data['rel_ade_improvement'], 
                 label=f"{loss_name} (ADE)", 
                 color=COLORS.get(loss_name, None), 
                 linewidth=2)
        plt.plot(loss_data['epoch'], loss_data['rel_fde_improvement'], 
                 label=f"{loss_name} (FDE)", 
                 color=COLORS.get(loss_name, None), 
                 linestyle='--',
                 linewidth=2)
    
    plt.title('Convergence Rate: Relative Improvement Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Relative Improvement (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence analysis plot to {output_dir}/convergence_analysis.png")

def load_best_results(results_dir='results'):
    """Load all best results files into DataFrames."""
    results = {}
    
    # Look for best results files with the expected naming pattern
    result_files = list(Path(results_dir).glob('*_best_results.csv'))
    
    if not result_files:
        print(f"No best results files found in {results_dir} directory.")
        return results
    
    # Load each results file
    for result_file in result_files:
        try:
            # Extract loss function name from filename
            loss_name = result_file.stem.split('_best_results')[0]
            
            # Load CSV into DataFrame
            df = pd.read_csv(result_file)
            
            # Store in dictionary
            results[loss_name] = df
            print(f"Loaded {result_file}")
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return results

def create_comparative_barplot(results, output_dir='results'):
    """Create a comparative bar plot of the best results."""
    if not results:
        print("No results available to create comparative bar plot.")
        return
    
    # Prepare data for plotting
    metrics = ['Best Average minADE', 'Best Average minFDE', 
               'Best median minADE', 'Best median minFDE']
    
    # Define specific order for loss functions
    ordered_loss_functions = ['minADE', 'minADEdiv', 'GMM_NLL']
    
    # Create a dict to store values for each metric and loss function
    data = {metric: [] for metric in metrics}
    available_loss_functions = []
    
    # Process results in the desired order
    for loss_name in ordered_loss_functions:
        if loss_name in results:
            available_loss_functions.append(loss_name)
            df = results[loss_name]
            for metric in metrics:
                # Extract value for this metric
                value = df[df['metric'] == metric]['value'].values
                data[metric].append(value[0] if len(value) > 0 else np.nan)
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(available_loss_functions))
    width = 0.2
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        rects = ax.bar(x + offset, data[metric], width, label=metric)
        multiplier += 1
    
    # Add labels, title and legend
    ax.set_xlabel('Loss Function')
    ax.set_ylabel('Value (lower is better)')
    ax.set_title('Comparison of Best Results Across Loss Functions')
    ax.set_xticks(x + width, available_loss_functions)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/best_results_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved best results comparison plot to {output_dir}/best_results_comparison.png")



def main():
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created 'results' directory")
    
    # Load training logs
    logs = load_training_logs()
    
    # Generate plots if logs were loaded successfully
    if logs:
        # Plot min_ade and min_fde
        plot_min_metrics(logs)
        
        # Plot total_loss
        plot_total_loss(logs)
        
        # Plot current ade and fde
        plot_current_metrics(logs)
        
        # Analyze convergence
        analyze_convergence(logs)
        
        # Load best results
        best_results = load_best_results()
        
        # Create comparative bar plot
        if best_results:
            create_comparative_barplot(best_results)
    else:
        print("No logs found to analyze")

if __name__ == "__main__":
    main()