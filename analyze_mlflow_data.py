#!/usr/bin/env python3
"""
Analyze MLflow JSON data to understand training dynamics across generations.
Focus on epoch analysis, training duration, overfitting patterns, and performance curves.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_mlflow_data(file_path):
    """Load MLflow data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['runs']

def extract_run_info(run):
    """Extract key information from a single run."""
    info = run['info']
    data = run['data']
    
    # Basic info
    run_info = {
        'run_id': info['run_uuid'],
        'run_name': info['run_name'],
        'experiment_id': info['experiment_id'],
        'status': info['status'],
        'start_time': info['start_time'],
        'end_time': info['end_time'],
        'duration_seconds': (info['end_time'] - info['start_time']) / 1000,
        'duration_hours': (info['end_time'] - info['start_time']) / (1000 * 60 * 60)
    }
    
    # Extract generation from run name
    if 'gen1' in run_info['run_name'] or 'gen2' in run_info['run_name']:
        run_info['generation'] = 'Gen1-2'
    elif 'gen3' in run_info['run_name'] or 'gen4' in run_info['run_name']:
        run_info['generation'] = 'Gen3-4'
    elif 'gen5' in run_info['run_name']:
        run_info['generation'] = 'Gen5'
    else:
        run_info['generation'] = 'Unknown'
    
    # Extract task type
    if 'clean_ice' in run_info['run_name'] or 'ci' in run_info['run_name']:
        run_info['task'] = 'clean_ice'
    elif 'debris_ice' in run_info['run_name'] or 'di' in run_info['run_name']:
        run_info['task'] = 'debris_ice'
    elif 'multi' in run_info['run_name']:
        run_info['task'] = 'multiclass'
    else:
        run_info['task'] = 'unknown'
    
    # Extract metrics
    metrics = {}
    for metric in data.get('metrics', []):
        key = metric['key']
        value = metric['value']
        step = metric.get('step', 0)
        
        if key not in metrics:
            metrics[key] = []
        metrics[key].append((step, value))
    
    # Get final epoch
    if 'epoch' in metrics:
        epochs = [v for s, v in metrics['epoch']]
        run_info['final_epoch'] = max(epochs) if epochs else None
    else:
        run_info['final_epoch'] = None
    
    # Get final metrics
    final_metrics = {}
    for key, values in metrics.items():
        if values:
            final_metrics[f'final_{key}'] = values[-1][1]  # Last value
    
    # Get best metrics (for validation metrics)
    for key, values in metrics.items():
        if key.startswith('val_') and values:
            if 'iou' in key or 'accuracy' in key or 'precision' in key or 'recall' in key or 'f1' in key:
                # Higher is better
                best_val = max(v for s, v in values)
                best_step = max(s for s, v in values if v == best_val)
                final_metrics[f'best_{key}'] = best_val
                final_metrics[f'best_{key}_epoch'] = best_step
            elif 'loss' in key:
                # Lower is better
                best_val = min(v for s, v in values)
                best_step = max(s for s, v in values if v == best_val)
                final_metrics[f'best_{key}'] = best_val
                final_metrics[f'best_{key}_epoch'] = best_step
    
    run_info.update(final_metrics)
    
    # Extract configuration parameters if available
    params = {}
    for param in data.get('params', []):
        params[param['key']] = param['value']
    
    # Key configuration parameters
    config_keys = ['max_epochs', 'early_stopping_patience', 'batch_size', 'lr', 'dataset_name']
    for key in config_keys:
        if key in params:
            run_info[f'config_{key}'] = params[key]
    
    return run_info

def analyze_epochs(df):
    """Analyze epoch patterns across generations."""
    print("\n" + "="*60)
    print("EPOCH ANALYSIS")
    print("="*60)
    
    # Epoch statistics by generation
    print("\nFinal Epoch Statistics by Generation:")
    epoch_stats = df.groupby('generation')['final_epoch'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(1)
    print(epoch_stats)
    
    # Epoch statistics by task
    print("\nFinal Epoch Statistics by Task:")
    task_epoch_stats = df.groupby('task')['final_epoch'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(1)
    print(task_epoch_stats)
    
    # Best performance epoch analysis
    if 'best_val_CleanIce_iou_epoch' in df.columns:
        print("\nBest Val IoU Epoch Statistics:")
        best_epoch_stats = df.groupby('generation')['best_val_CleanIce_iou_epoch'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(1)
        print(best_epoch_stats)
    
    # Convergence analysis (when did 90% of final performance occur?)
    convergence_data = []
    for _, run in df.iterrows():
        if pd.notna(run['final_epoch']) and run['final_epoch'] > 10:
            # Estimate 90% convergence at 80% of final epoch (heuristic)
            convergence_epoch = run['final_epoch'] * 0.8
            convergence_data.append({
                'generation': run['generation'],
                'task': run['task'],
                'final_epoch': run['final_epoch'],
                'estimated_90pct_epoch': convergence_epoch
            })
    
    if convergence_data:
        conv_df = pd.DataFrame(convergence_data)
        print("\nEstimated 90% Performance Epoch:")
        conv_stats = conv_df.groupby('generation')['estimated_90pct_epoch'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(1)
        print(conv_stats)

def analyze_training_duration(df):
    """Analyze training duration patterns."""
    print("\n" + "="*60)
    print("TRAINING DURATION ANALYSIS")
    print("="*60)
    
    # Duration statistics by generation
    print("\nTraining Duration (hours) by Generation:")
    duration_stats = df.groupby('generation')['duration_hours'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    print(duration_stats)
    
    # Time per epoch
    df['time_per_epoch'] = df['duration_hours'] / df['final_epoch']
    print("\nTime per Epoch (hours) by Generation:")
    time_per_epoch_stats = df.groupby('generation')['time_per_epoch'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    print(time_per_epoch_stats)
    
    # Duration by task
    print("\nTraining Duration (hours) by Task:")
    task_duration_stats = df.groupby('task')['duration_hours'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    print(task_duration_stats)

def analyze_overfitting(df):
    """Analyze overfitting patterns."""
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    
    # Train-val gap analysis (if both metrics available)
    if 'final_train_loss' in df.columns and 'final_val_loss' in df.columns:
        df['train_val_gap'] = df['final_val_loss'] - df['final_train_loss']
        print("\nTrain-Val Loss Gap by Generation:")
        gap_stats = df.groupby('generation')['train_val_gap'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(4)
        print(gap_stats)
        
        # Overfitting threshold (gap > 0.1)
        overfit_threshold = 0.1
        df['is_overfitting'] = df['train_val_gap'] > overfit_threshold
        print(f"\nOverfitting Rate (gap > {overfit_threshold}):")
        overfit_rates = df.groupby('generation')['is_overfitting'].mean().round(3)
        print(overfit_rates)
    
    # Performance degradation analysis
    if 'best_val_CleanIce_iou' in df.columns and 'final_val_CleanIce_iou' in df.columns:
        df['performance_degradation'] = df['best_val_CleanIce_iou'] - df['final_val_CleanIce_iou']
        print("\nPerformance Degradation (Best - Final IoU):")
        degrad_stats = df.groupby('generation')['performance_degradation'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(4)
        print(degrad_stats)

def analyze_performance_curves(df):
    """Analyze performance vs epoch patterns."""
    print("\n" + "="*60)
    print("PERFORMANCE CURVE ANALYSIS")
    print("="*60)
    
    # Final performance by generation
    if 'final_val_CleanIce_iou' in df.columns:
        print("\nFinal Validation IoU by Generation:")
        perf_stats = df.groupby('generation')['final_val_CleanIce_iou'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        print(perf_stats)
    
    # Best performance by generation
    if 'best_val_CleanIce_iou' in df.columns:
        print("\nBest Validation IoU by Generation:")
        best_perf_stats = df.groupby('generation')['best_val_CleanIce_iou'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        print(best_perf_stats)
    
    # Performance consistency (std/mean ratio)
    if 'best_val_CleanIce_iou' in df.columns:
        consistency = df.groupby('generation')['best_val_CleanIce_iou'].agg(['mean', 'std'])
        consistency['cv'] = consistency['std'] / consistency['mean']  # Coefficient of variation
        print("\nPerformance Consistency (CV = std/mean, lower is better):")
        print(consistency[['mean', 'cv']].round(3))

def analyze_config_patterns(df):
    """Analyze configuration-specific patterns."""
    print("\n" + "="*60)
    print("CONFIGURATION PATTERN ANALYSIS")
    print("="*60)
    
    # Configuration parameters analysis
    config_cols = [col for col in df.columns if col.startswith('config_')]
    
    for col in config_cols:
        if df[col].notna().sum() > 0:  # Only analyze if we have data
            print(f"\n{col} Distribution:")
            value_counts = df[col].value_counts().head(10)
            print(value_counts)
            
            # Impact on final performance
            if 'final_val_CleanIce_iou' in df.columns:
                print(f"\nImpact of {col} on Final IoU:")
                impact = df.groupby(col)['final_val_CleanIce_iou'].agg(['count', 'mean', 'std']).round(3)
                print(impact.head(10))

def generate_recommendations(df):
    """Generate recommendations for Gen6."""
    print("\n" + "="*60)
    print("GEN6 RECOMMENDATIONS")
    print("="*60)
    
    recommendations = {}
    
    # Epoch recommendations
    if 'final_epoch' in df.columns:
        epoch_stats = df['final_epoch'].describe()
        # Use 75th percentile + 1 std as safe max_epochs
        recommended_max_epochs = int(epoch_stats['75%'] + epoch_stats['std'])
        recommendations['max_epochs'] = recommended_max_epochs
        
        # Early stopping patience (10% of max_epochs)
        recommended_patience = max(10, int(recommended_max_epochs * 0.1))
        recommendations['early_stopping_patience'] = recommended_patience
    
    # Task-specific recommendations
    task_recommendations = {}
    for task in df['task'].unique():
        if task != 'unknown':
            task_data = df[df['task'] == task]
            if 'final_epoch' in task_data.columns:
                task_epochs = task_data['final_epoch'].describe()
                task_max = int(task_epochs['75%'] + task_epochs['std'])
                task_patience = max(10, int(task_max * 0.1))
                task_recommendations[task] = {
                    'max_epochs': task_max,
                    'early_stopping_patience': task_patience
                }
    
    recommendations['task_specific'] = task_recommendations
    
    print("\nRecommended Configuration for Gen6:")
    print(f"Global max_epochs: {recommendations.get('max_epochs', 'N/A')}")
    print(f"Global early_stopping_patience: {recommendations.get('early_stopping_patience', 'N/A')}")
    
    print("\nTask-Specific Recommendations:")
    for task, config in recommendations.get('task_specific', {}).items():
        print(f"  {task}:")
        print(f"    max_epochs: {config['max_epochs']}")
        print(f"    early_stopping_patience: {config['early_stopping_patience']}")
    
    # Justification
    print("\nJustification:")
    print("- max_epochs based on 75th percentile + 1 std of historical convergence")
    print("- early_stopping_patience set to 10% of max_epochs (minimum 10)")
    print("- Task-specific adjustments account for different convergence patterns")
    
    return recommendations

def main():
    """Main analysis function."""
    print("MLflow Training Dynamics Analysis")
    print("="*60)
    
    # Load data from all generation files
    data_files = [
        'gen1_2_fresh.json',
        'gen3_4_fresh.json', 
        'gen5_fresh.json'
    ]
    
    all_runs = []
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"Loading {file_path}...")
            runs = load_mlflow_data(file_path)
            all_runs.extend(runs)
        else:
            print(f"Warning: {file_path} not found")
    
    if not all_runs:
        print("No data found!")
        return
    
    print(f"Total runs loaded: {len(all_runs)}")
    
    # Extract information from all runs
    print("Extracting run information...")
    run_infos = []
    for run in all_runs:
        try:
            run_info = extract_run_info(run)
            run_infos.append(run_info)
        except Exception as e:
            print(f"Error processing run {run.get('info', {}).get('run_name', 'unknown')}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(run_infos)
    print(f"Successfully processed {len(df)} runs")
    
    # Filter out runs with critical missing data
    df = df[df['final_epoch'].notna()]
    print(f"Runs with epoch data: {len(df)}")
    
    # Print generation distribution
    print("\nRun Distribution by Generation:")
    print(df['generation'].value_counts())
    
    print("\nRun Distribution by Task:")
    print(df['task'].value_counts())
    
    # Run analyses
    analyze_epochs(df)
    analyze_training_duration(df)
    analyze_overfitting(df)
    analyze_performance_curves(df)
    analyze_config_patterns(df)
    recommendations = generate_recommendations(df)
    
    # Save detailed results
    output_file = 'mlflow_analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    return df, recommendations

if __name__ == "__main__":
    df, recommendations = main()
