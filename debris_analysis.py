#!/usr/bin/env python3
"""
Comprehensive debris detection analysis comparing gen5 vs gen6
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def load_generation_data(filepath):
    """Load generation data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_debris_metrics(runs_data):
    """Extract debris metrics from runs."""
    debris_metrics = []
    
    for run in runs_data:
        run_name = run.get('run_name', 'unknown')
        config = run.get('configuration', {})
        performance = run.get('performance', {})
        
        # Get debris metrics
        all_metrics = performance.get('all_metrics', {})
        best_metrics = performance.get('best_metrics', {})
        
        debris_data = {
            'run_name': run_name,
            'task': config.get('task', 'unknown'),
            'config_type': config.get('config_type', 'unknown'),
            'server': config.get('server', 'unknown'),
            'status': run.get('status', 'unknown'),
            
            # Debris IoU metrics
            'val_Debris_iou': all_metrics.get('val_Debris_iou', None),
            'test_Debris_iou': all_metrics.get('test_Debris_iou', None),
            'best_test_Debris_iou': best_metrics.get('best_test_Debris_iou', None),
            'train_Debris_iou': all_metrics.get('train_Debris_iou', None),
            
            # Debris precision metrics
            'val_Debris_precision': all_metrics.get('val_Debris_precision', None),
            'test_Debris_precision': all_metrics.get('test_Debris_precision', None),
            'best_test_Debris_precision': best_metrics.get('best_test_Debris_precision', None),
            'train_Debris_precision': all_metrics.get('train_Debris_precision', None),
            
            # Debris recall metrics
            'val_Debris_recall': all_metrics.get('val_Debris_recall', None),
            'test_Debris_recall': all_metrics.get('test_Debris_recall', None),
            'best_test_Debris_recall': best_metrics.get('best_test_Debris_recall', None),
            'train_Debris_recall': all_metrics.get('train_Debris_recall', None),
        }
        
        debris_metrics.append(debris_data)
    
    return pd.DataFrame(debris_metrics)

def analyze_debris_performance(gen5_df, gen6_df):
    """Analyze and compare debris performance between generations."""
    
    print("=" * 80)
    print("DEBRIS DETECTION PERFORMANCE ANALYSIS: GEN5 vs GEN6")
    print("=" * 80)
    
    # Filter finished runs only
    gen5_finished = gen5_df[gen5_df['status'] == 'FINISHED'].copy()
    gen6_finished = gen6_df[gen6_df['status'] == 'FINISHED'].copy()
    
    print(f"\nSUMMARY:")
    print(f"Gen5 finished runs: {len(gen5_finished)}")
    print(f"Gen6 finished runs: {len(gen6_finished)}")
    
    # Key metrics comparison
    metrics_to_compare = [
        'best_test_Debris_iou',
        'best_test_Debris_precision', 
        'best_test_Debris_recall',
        'val_Debris_iou',
        'test_Debris_iou'
    ]
    
    print(f"\nKEY METRICS COMPARISON:")
    print("-" * 60)
    print(f"{'Metric':<25} {'Gen5 Mean':<12} {'Gen6 Mean':<12} {'Difference':<12}")
    print("-" * 60)
    
    for metric in metrics_to_compare:
        gen5_vals = gen5_finished[metric].dropna()
        gen6_vals = gen6_finished[metric].dropna()
        
        if len(gen5_vals) > 0 and len(gen6_vals) > 0:
            gen5_mean = gen5_vals.mean()
            gen6_mean = gen6_vals.mean()
            diff = gen6_mean - gen5_mean
            pct_change = (diff / gen5_mean) * 100 if gen5_mean != 0 else float('inf')
            
            print(f"{metric:<25} {gen5_mean:<12.6f} {gen6_mean:<12.6f} {diff:+.6f} ({pct_change:+.1f}%)")
        else:
            print(f"{metric:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    # Zero/near-zero analysis
    print(f"\nNEAR-ZERO DEBRIS IOU ANALYSIS:")
    print("-" * 60)
    
    threshold = 0.001  # Near-zero threshold
    
    gen5_near_zero = gen5_finished[gen5_finished['best_test_Debris_iou'] < threshold]
    gen6_near_zero = gen6_finished[gen6_finished['best_test_Debris_iou'] < threshold]
    
    print(f"Gen5 runs with IoU < {threshold}: {len(gen5_near_zero)}/{len(gen5_finished)} ({len(gen5_near_zero)/len(gen5_finished)*100:.1f}%)")
    print(f"Gen6 runs with IoU < {threshold}: {len(gen6_near_zero)}/{len(gen6_finished)} ({len(gen6_near_zero)/len(gen6_finished)*100:.1f}%)")
    
    # Configuration-specific analysis
    print(f"\nCONFIGURATION-SPECIFIC ANALYSIS:")
    print("-" * 60)
    
    for config in ['baseline', 'physics', 'velocity', 'synthesis']:
        gen5_config = gen5_finished[gen5_finished['config_type'] == config]
        gen6_config = gen6_finished[gen6_finished['config_type'] == config]
        
        if len(gen5_config) > 0 and len(gen6_config) > 0:
            gen5_iou = gen5_config['best_test_Debris_iou'].mean()
            gen6_iou = gen6_config['best_test_Debris_iou'].mean()
            
            print(f"{config:12} Gen5: {gen5_iou:.6f}, Gen6: {gen6_iou:.6f}, Change: {gen6_iou-gen5_iou:+.6f}")
    
    # Task-specific analysis
    print(f"\nTASK-SPECIFIC ANALYSIS:")
    print("-" * 60)
    
    for task in ['multiclass', 'debris_ice', 'clean_ice']:
        gen5_task = gen5_finished[gen5_finished['task'] == task]
        gen6_task = gen6_finished[gen6_finished['task'] == task]
        
        if len(gen5_task) > 0 and len(gen6_task) > 0:
            gen5_iou = gen5_task['best_test_Debris_iou'].mean()
            gen6_iou = gen6_task['best_test_Debris_iou'].mean()
            
            print(f"{task:12} Gen5: {gen5_iou:.6f}, Gen6: {gen6_iou:.6f}, Change: {gen6_iou-gen5_iou:+.6f}")
    
    # Worst affected runs
    print(f"\nGEN6 RUNS WITH WORST DEBRIS PERFORMANCE:")
    print("-" * 60)
    
    worst_gen6 = gen6_finished.nsmallest(10, 'best_test_Debris_iou')[['run_name', 'config_type', 'best_test_Debris_iou']]
    for _, row in worst_gen6.iterrows():
        print(f"{row['run_name']:<40} {row['config_type']:<12} {row['best_test_Debris_iou']:.8f}")
    
    # Check for systematic patterns
    print(f"\nSYSTEMATIC PATTERN ANALYSIS:")
    print("-" * 60)
    
    # Check if all gen6 runs have low debris metrics
    gen6_all_low = all(gen6_finished['best_test_Debris_iou'] < 0.01)
    gen5_some_good = any(gen5_finished['best_test_Debris_iou'] > 0.1)
    
    print(f"All Gen6 runs have IoU < 0.01: {gen6_all_low}")
    print(f"Any Gen5 runs have IoU > 0.1: {gen5_some_good}")
    
    if gen6_all_low and gen5_some_good:
        print("⚠️  STRONG EVIDENCE OF SYSTEMATIC BUG IN GEN6 DEBRIS DETECTION!")
    
    return gen5_finished, gen6_finished

def main():
    """Main analysis function."""
    
    # Load data
    print("Loading generation data...")
    gen5_data = load_generation_data('gen5_fresh.json')
    gen6_data = load_generation_data('archive/gen6/gen6_all_data.json')
    
    # Extract debris metrics
    gen5_df = extract_debris_metrics(gen5_data['runs'])
    gen6_df = extract_debris_metrics(gen6_data['runs'])
    
    # Analyze
    gen5_finished, gen6_finished = analyze_debris_performance(gen5_df, gen6_df)
    
    # Save detailed results
    results = {
        'gen5_summary': {
            'total_runs': len(gen5_df),
            'finished_runs': len(gen5_finished),
            'mean_best_debris_iou': gen5_finished['best_test_Debris_iou'].mean(),
            'median_best_debris_iou': gen5_finished['best_test_Debris_iou'].median(),
            'max_best_debris_iou': gen5_finished['best_test_Debris_iou'].max(),
        },
        'gen6_summary': {
            'total_runs': len(gen6_df),
            'finished_runs': len(gen6_finished),
            'mean_best_debris_iou': gen6_finished['best_test_Debris_iou'].mean(),
            'median_best_debris_iou': gen6_finished['best_test_Debris_iou'].median(),
            'max_best_debris_iou': gen6_finished['best_test_Debris_iou'].max(),
        }
    }
    
    with open('debris_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to debris_analysis_results.json")

if __name__ == "__main__":
    main()
