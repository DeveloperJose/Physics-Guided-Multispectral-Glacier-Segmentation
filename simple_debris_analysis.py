#!/usr/bin/env python3
"""
Simple debris detection analysis comparing gen5 vs gen6
"""

import json
import numpy as np
import pandas as pd

def extract_gen5_debris_metrics():
    """Extract debris metrics from gen5 data."""
    with open('gen5_fresh.json', 'r') as f:
        gen5_data = json.load(f)
    
    debris_metrics = []
    
    for run in gen5_data['runs']:
        run_name = run['info']['run_name']
        status = run['info']['status']
        
        if status != 'FINISHED':
            continue
            
        # Extract metrics from the metrics list
        metrics = {}
        for metric in run['data']['metrics']:
            key = metric['key']
            value = metric['value']
            metrics[key] = value
        
        debris_data = {
            'run_name': run_name,
            'status': status,
            'best_test_Debris_iou': metrics.get('best_test_Debris_iou', None),
            'best_test_Debris_precision': metrics.get('best_test_Debris_precision', None),
            'best_test_Debris_recall': metrics.get('best_test_Debris_recall', None),
            'test_Debris_iou': metrics.get('test_Debris_iou', None),
            'test_Debris_precision': metrics.get('test_Debris_precision', None),
            'test_Debris_recall': metrics.get('test_Debris_recall', None),
            'val_Debris_iou': metrics.get('val_Debris_iou', None),
            'val_Debris_precision': metrics.get('val_Debris_precision', None),
            'val_Debris_recall': metrics.get('val_Debris_recall', None),
        }
        
        debris_metrics.append(debris_data)
    
    return pd.DataFrame(debris_metrics)

def extract_gen6_debris_metrics():
    """Extract debris metrics from gen6 data."""
    with open('archive/gen6/gen6_all_data.json', 'r') as f:
        gen6_data = json.load(f)
    
    debris_metrics = []
    
    for run in gen6_data['runs']:
        run_name = run['run_name']
        status = run['status']
        
        if status != 'FINISHED':
            continue
            
        performance = run.get('performance', {})
        all_metrics = performance.get('all_metrics', {})
        best_metrics = performance.get('best_metrics', {})
        
        debris_data = {
            'run_name': run_name,
            'status': status,
            'best_test_Debris_iou': best_metrics.get('best_test_Debris_iou', None),
            'best_test_Debris_precision': best_metrics.get('best_test_Debris_precision', None),
            'best_test_Debris_recall': best_metrics.get('best_test_Debris_recall', None),
            'test_Debris_iou': all_metrics.get('test_Debris_iou', None),
            'test_Debris_precision': all_metrics.get('test_Debris_precision', None),
            'test_Debris_recall': all_metrics.get('test_Debris_recall', None),
            'val_Debris_iou': all_metrics.get('val_Debris_iou', None),
            'val_Debris_precision': all_metrics.get('val_Debris_precision', None),
            'val_Debris_recall': all_metrics.get('val_Debris_recall', None),
        }
        
        debris_metrics.append(debris_data)
    
    return pd.DataFrame(debris_metrics)

def main():
    """Main analysis function."""
    
    print("=" * 80)
    print("DEBRIS DETECTION PERFORMANCE ANALYSIS: GEN5 vs GEN6")
    print("=" * 80)
    
    # Extract metrics
    print("Extracting debris metrics...")
    gen5_df = extract_gen5_debris_metrics()
    gen6_df = extract_gen6_debris_metrics()
    
    print(f"Gen5 finished runs with debris data: {len(gen5_df)}")
    print(f"Gen6 finished runs with debris data: {len(gen6_df)}")
    
    # Key metrics comparison
    metrics_to_compare = [
        'best_test_Debris_iou',
        'best_test_Debris_precision', 
        'best_test_Debris_recall',
        'test_Debris_iou',
        'val_Debris_iou'
    ]
    
    print(f"\nKEY METRICS COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<25} {'Gen5 Mean':<12} {'Gen6 Mean':<12} {'Gen5 Max':<12} {'Gen6 Max':<12}")
    print("-" * 80)
    
    for metric in metrics_to_compare:
        gen5_vals = gen5_df[metric].dropna()
        gen6_vals = gen6_df[metric].dropna()
        
        gen5_mean = gen5_vals.mean() if len(gen5_vals) > 0 else 0
        gen6_mean = gen6_vals.mean() if len(gen6_vals) > 0 else 0
        gen5_max = gen5_vals.max() if len(gen5_vals) > 0 else 0
        gen6_max = gen6_vals.max() if len(gen6_vals) > 0 else 0
        
        diff = gen6_mean - gen5_mean
        pct_change = (diff / gen5_mean * 100) if gen5_mean != 0 else float('inf')
        
        print(f"{metric:<25} {gen5_mean:<12.6f} {gen6_mean:<12.6f} {gen5_max:<12.6f} {gen6_max:<12.6f}")
        
        if gen5_mean > 0.01 and gen6_mean < 0.01:
            print(f"  ⚠️  MAJOR DEGRADATION: {pct_change:+.1f}%")
    
    # Near-zero analysis
    print(f"\nNEAR-ZERO DEBRIS IOU ANALYSIS:")
    print("-" * 50)
    
    threshold = 0.001
    
    gen5_near_zero = gen5_df[gen5_df['best_test_Debris_iou'] < threshold] if len(gen5_df) > 0 else pd.DataFrame()
    gen6_near_zero = gen6_df[gen6_df['best_test_Debris_iou'] < threshold] if len(gen6_df) > 0 else pd.DataFrame()
    
    if len(gen5_df) > 0:
        print(f"Gen5 runs with IoU < {threshold}: {len(gen5_near_zero)}/{len(gen5_df)} ({len(gen5_near_zero)/len(gen5_df)*100:.1f}%)")
    if len(gen6_df) > 0:
        print(f"Gen6 runs with IoU < {threshold}: {len(gen6_near_zero)}/{len(gen6_df)} ({len(gen6_near_zero)/len(gen6_df)*100:.1f}%)")
    
    # Show worst performing runs
    print(f"\nWORST PERFORMING RUNS (Best Test Debris IoU):")
    print("-" * 80)
    
    if len(gen5_df) > 0:
        print("GEN5 BOTTOM 5:")
        worst_gen5 = gen5_df.nsmallest(5, 'best_test_Debris_iou')[['run_name', 'best_test_Debris_iou']]
        for _, row in worst_gen5.iterrows():
            print(f"  {row['run_name']:<50} {row['best_test_Debris_iou']:.8f}")
    
    if len(gen6_df) > 0:
        print("\nGEN6 BOTTOM 5:")
        worst_gen6 = gen6_df.nsmallest(5, 'best_test_Debris_iou')[['run_name', 'best_test_Debris_iou']]
        for _, row in worst_gen6.iterrows():
            print(f"  {row['run_name']:<50} {row['best_test_Debris_iou']:.8f}")
    
    # Show best performing runs
    print(f"\nBEST PERFORMING RUNS (Best Test Debris IoU):")
    print("-" * 80)
    
    if len(gen5_df) > 0:
        print("GEN5 TOP 5:")
        best_gen5 = gen5_df.nlargest(5, 'best_test_Debris_iou')[['run_name', 'best_test_Debris_iou']]
        for _, row in best_gen5.iterrows():
            print(f"  {row['run_name']:<50} {row['best_test_Debris_iou']:.8f}")
    
    if len(gen6_df) > 0:
        print("\nGEN6 TOP 5:")
        best_gen6 = gen6_df.nlargest(5, 'best_test_Debris_iou')[['run_name', 'best_test_Debris_iou']]
        for _, row in best_gen6.iterrows():
            print(f"  {row['run_name']:<50} {row['best_test_Debris_iou']:.8f}")
    
    # Systematic bug analysis
    print(f"\nSYSTEMATIC BUG ANALYSIS:")
    print("-" * 50)
    
    if len(gen5_df) > 0 and len(gen6_df) > 0:
        gen5_good = len(gen5_df[gen5_df['best_test_Debris_iou'] > 0.1])
        gen6_good = len(gen6_df[gen6_df['best_test_Debris_iou'] > 0.1])
        gen5_any = len(gen5_df[gen5_df['best_test_Debris_iou'] > 0.01])
        gen6_any = len(gen6_df[gen6_df['best_test_Debris_iou'] > 0.01])
        
        print(f"Gen5 runs with IoU > 0.1: {gen5_good}/{len(gen5_df)} ({gen5_good/len(gen5_df)*100:.1f}%)")
        print(f"Gen6 runs with IoU > 0.1: {gen6_good}/{len(gen6_df)} ({gen6_good/len(gen6_df)*100:.1f}%)")
        print(f"Gen5 runs with IoU > 0.01: {gen5_any}/{len(gen5_df)} ({gen5_any/len(gen5_df)*100:.1f}%)")
        print(f"Gen6 runs with IoU > 0.01: {gen6_any}/{len(gen6_df)} ({gen6_any/len(gen6_df)*100:.1f}%)")
        
        if gen5_good > 0 and gen6_good == 0:
            print("\n🚨 STRONG EVIDENCE OF SYSTEMATIC BUG IN GEN6!")
            print("   - Gen5 had runs with good debris performance (>0.1 IoU)")
            print("   - Gen6 has NO runs with good debris performance")
            print("   - This suggests a systematic issue in gen6 debris detection")
        
        if gen5_any > 0 and gen6_any == 0:
            print("\n⚠️  EVIDENCE OF SEVERE DEGRADATION IN GEN6!")
            print("   - Gen5 had runs with measurable debris performance (>0.01 IoU)")
            print("   - Gen6 has NO runs with measurable debris performance")
    
    # Save results
    results = {
        'gen5_summary': {
            'total_runs': len(gen5_df),
            'mean_best_debris_iou': gen5_df['best_test_Debris_iou'].mean() if len(gen5_df) > 0 else 0,
            'max_best_debris_iou': gen5_df['best_test_Debris_iou'].max() if len(gen5_df) > 0 else 0,
            'runs_with_iou_gt_01': len(gen5_df[gen5_df['best_test_Debris_iou'] > 0.1]) if len(gen5_df) > 0 else 0,
        },
        'gen6_summary': {
            'total_runs': len(gen6_df),
            'mean_best_debris_iou': gen6_df['best_test_Debris_iou'].mean() if len(gen6_df) > 0 else 0,
            'max_best_debris_iou': gen6_df['best_test_Debris_iou'].max() if len(gen6_df) > 0 else 0,
            'runs_with_iou_gt_01': len(gen6_df[gen6_df['best_test_Debris_iou'] > 0.1]) if len(gen6_df) > 0 else 0,
        }
    }
    
    with open('debris_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to debris_analysis_results.json")

if __name__ == "__main__":
    main()
