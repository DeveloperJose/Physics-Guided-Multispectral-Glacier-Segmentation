#!/usr/bin/env python3
"""
Debug debris detection analysis comparing gen5 vs gen6
"""

import json
import numpy as np
import pandas as pd

def extract_gen5_debris_metrics():
    """Extract debris metrics from gen5 data."""
    with open('gen5_fresh.json', 'r') as f:
        gen5_data = json.load(f)
    
    debris_metrics = []
    
    for i, run in enumerate(gen5_data['runs']):
        try:
            run_name = run['info']['run_name']
            status = run['info']['status']
            
            if status != 'FINISHED':
                continue
                
            # Extract metrics from metrics list
            metrics = {}
            if 'data' in run and 'metrics' in run['data']:
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
            
        except Exception as e:
            print(f"Error processing gen5 run {i}: {e}")
            continue
    
    return pd.DataFrame(debris_metrics)

def extract_gen6_debris_metrics():
    """Extract debris metrics from gen6 data."""
    with open('archive/gen6/gen6_all_data.json', 'r') as f:
        gen6_data = json.load(f)
    
    debris_metrics = []
    
    for i, run in enumerate(gen6_data['runs']):
        try:
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
            
        except Exception as e:
            print(f"Error processing gen6 run {i}: {e}")
            continue
    
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
    
    if len(gen5_df) == 0:
        print("No gen5 data extracted!")
        return
    
    if len(gen6_df) == 0:
        print("No gen6 data extracted!")
        return
    
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
    
    # Show actual values for inspection
    print(f"\nGEN5 SAMPLE DEBRIS IOU VALUES:")
    print("-" * 50)
    gen5_iou_values = gen5_df['best_test_Debris_iou'].dropna()
    print(f"Values: {gen5_iou_values.values[:10]}")
    print(f"Mean: {gen5_iou_values.mean():.6f}")
    print(f"Max: {gen5_iou_values.max():.6f}")
    
    print(f"\nGEN6 SAMPLE DEBRIS IOU VALUES:")
    print("-" * 50)
    gen6_iou_values = gen6_df['best_test_Debris_iou'].dropna()
    print(f"Values: {gen6_iou_values.values[:10]}")
    print(f"Mean: {gen6_iou_values.mean():.6f}")
    print(f"Max: {gen6_iou_values.max():.6f}")
    
    # Systematic bug analysis
    print(f"\nSYSTEMATIC BUG ANALYSIS:")
    print("-" * 50)
    
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

if __name__ == "__main__":
    main()
