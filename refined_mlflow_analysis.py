#!/usr/bin/env python3
"""
Refined MLflow analysis focusing on practical Gen6 recommendations.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean MLflow data for analysis."""
    
    data_files = ['gen1_2_fresh.json', 'gen3_4_fresh.json', 'gen5_fresh.json']
    all_runs = []
    
    for file_path in data_files:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_runs.extend(data['runs'])
    
    cleaned_runs = []
    
    for run in all_runs:
        try:
            info = run['info']
            data = run['data']
            
            # Skip runs without proper timing
            if 'end_time' not in info or info['end_time'] is None:
                continue
                
            # Skip failed runs
            if info.get('status') != 'FINISHED':
                continue
                
            run_info = {
                'run_id': info['run_uuid'],
                'run_name': info['run_name'],
                'start_time': info['start_time'],
                'end_time': info['end_time'],
                'duration_hours': (info['end_time'] - info['start_time']) / (1000 * 60 * 60)
            }
            
            # Extract generation and task
            name = run_info['run_name'].lower()
            if 'gen5' in name:
                run_info['generation'] = 'Gen5'
            elif 'gen3' in name or 'gen4' in name:
                run_info['generation'] = 'Gen3-4'
            elif 'gen1' in name or 'gen2' in name:
                run_info['generation'] = 'Gen1-2'
            else:
                run_info['generation'] = 'Other'
            
            if 'clean_ice' in name or 'ci' in name:
                run_info['task'] = 'clean_ice'
            elif 'debris_ice' in name or 'di' in name:
                run_info['task'] = 'debris_ice'
            elif 'multi' in name:
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
            
            # Get performance metrics
            for key in ['val_loss', 'val_CleanIce_iou', 'train_loss']:
                if key in metrics and metrics[key]:
                    run_info[f'final_{key}'] = metrics[key][-1][1]
            
            # Get best performance
            for key in metrics:
                if key.startswith('val_') and metrics[key]:
                    if 'iou' in key:
                        best_val = max(v for s, v in metrics[key])
                        best_step = max(s for s, v in metrics[key] if v == best_val)
                        run_info[f'best_{key}'] = best_val
                        run_info[f'best_{key}_epoch'] = best_step
                    elif 'loss' in key:
                        best_val = min(v for s, v in metrics[key])
                        best_step = max(s for s, v in metrics[key] if v == best_val)
                        run_info[f'best_{key}'] = best_val
                        run_info[f'best_{key}_epoch'] = best_step
            
            # Extract configuration
            params = {param['key']: param['value'] for param in data.get('params', [])}
            for key in ['max_epochs', 'early_stopping_patience', 'batch_size', 'dataset_name']:
                if key in params:
                    run_info[f'config_{key}'] = params[key]
            
            cleaned_runs.append(run_info)
            
        except Exception as e:
            continue
    
    return pd.DataFrame(cleaned_runs)

def analyze_meaningful_runs(df):
    """Focus on runs that provide meaningful insights."""
    
    # Filter for meaningful runs
    meaningful = df[
        (df['final_epoch'].notna()) & 
        (df['final_epoch'] > 5) &  # At least 5 epochs
        (df['duration_hours'] > 0.01) &  # At least 36 seconds
        (df['duration_hours'] < 50)  # Less than 2 days
    ].copy()
    
    print(f"Meaningful runs: {len(meaningful)} out of {len(df)} total")
    
    # Focus on recent generations (Gen3-4 and Gen5)
    recent = meaningful[meaningful['generation'].isin(['Gen3-4', 'Gen5'])].copy()
    print(f"Recent generations (Gen3-4, Gen5): {len(recent)} runs")
    
    return meaningful, recent

def analyze_convergence_patterns(meaningful, recent):
    """Analyze when models converge and stop improving."""
    
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    
    # Overall statistics
    print(f"\nOverall Epoch Statistics:")
    stats = meaningful['final_epoch'].describe()
    for stat, value in stats.items():
        print(f"  {stat}: {value:.1f}")
    
    # Recent generations
    print(f"\nRecent Generations (Gen3-4, Gen5):")
    recent_stats = recent['final_epoch'].describe()
    for stat, value in recent_stats.items():
        print(f"  {stat}: {value:.1f}")
    
    # Task-specific analysis
    print(f"\nTask-Specific Epoch Statistics:")
    for task in ['clean_ice', 'multiclass']:
        task_data = recent[recent['task'] == task]['final_epoch']
        if len(task_data) > 0:
            print(f"\n{task}:")
            stats = task_data.describe()
            for stat, value in stats.items():
                print(f"  {stat}: {value:.1f}")
    
    # Identify convergence patterns
    print(f"\nConvergence Insights:")
    
    # Most common final epochs
    epoch_counts = recent['final_epoch'].value_counts().head(10)
    print(f"Most common final epochs:")
    for epoch, count in epoch_counts.items():
        print(f"  {epoch}: {count} runs")
    
    # Early convergence (<= 50 epochs)
    early_converge = recent[recent['final_epoch'] <= 50]
    print(f"\nEarly convergence (<= 50 epochs): {len(early_converge)} runs ({len(early_converge)/len(recent)*100:.1f}%)")
    
    # Long training (> 100 epochs)
    long_training = recent[recent['final_epoch'] > 100]
    print(f"Long training (> 100 epochs): {len(long_training)} runs ({len(long_training)/len(recent)*100:.1f}%)")
    
    return recent

def analyze_performance_timing(recent):
    """Analyze when peak performance occurs."""
    
    print("\n" + "="*60)
    print("PERFORMANCE TIMING ANALYSIS")
    print("="*60)
    
    # Best performance epoch analysis
    if 'best_val_CleanIce_iou_epoch' in recent.columns:
        perf_data = recent[recent['best_val_CleanIce_iou_epoch'].notna()].copy()
        
        print(f"\nBest Performance Epoch Analysis ({len(perf_data)} runs):")
        
        # Convert step to epoch approximation (assuming ~66 steps per epoch)
        perf_data['best_epoch_approx'] = perf_data['best_val_CleanIce_iou_epoch'] / 66
        
        # Statistics
        stats = perf_data['best_epoch_approx'].describe()
        for stat, value in stats.items():
            print(f"  {stat}: {value:.1f}")
        
        # When does 90% of final performance occur?
        perf_data['pct_of_final'] = perf_data['best_epoch_approx'] / perf_data['final_epoch']
        early_achievers = perf_data[perf_data['pct_of_final'] <= 0.8]
        
        print(f"\nEarly achievers (best performance by 80% of training): {len(early_achievers)} runs")
        if len(early_achievers) > 0:
            print(f"  Average epoch for best performance: {early_achievers['best_epoch_approx'].mean():.1f}")
            print(f"  This was {early_achievers['pct_of_final'].mean()*100:.1f}% of their total training")
    
    return recent

def analyze_training_efficiency(recent):
    """Analyze training time efficiency."""
    
    print("\n" + "="*60)
    print("TRAINING EFFICIENCY ANALYSIS")
    print("="*60)
    
    # Time per epoch
    recent['time_per_epoch'] = recent['duration_hours'] / recent['final_epoch']
    
    print(f"\nTime per Epoch (hours):")
    stats = recent['time_per_epoch'].describe()
    for stat, value in stats.items():
        print(f"  {stat}: {value:.4f}")
    
    # By generation
    print(f"\nTime per Epoch by Generation:")
    for gen in ['Gen3-4', 'Gen5']:
        gen_data = recent[recent['generation'] == gen]
        if len(gen_data) > 0:
            stats = gen_data['time_per_epoch'].describe()
            print(f"\n{gen}:")
            for stat, value in stats.items():
                print(f"  {stat}: {value:.4f}")
    
    # By task
    print(f"\nTime per Epoch by Task:")
    for task in ['clean_ice', 'multiclass']:
        task_data = recent[recent['task'] == task]
        if len(task_data) > 0:
            stats = task_data['time_per_epoch'].describe()
            print(f"\n{task}:")
            for stat, value in stats.items():
                print(f"  {stat}: {value:.4f}")
    
    return recent

def generate_practical_recommendations(recent):
    """Generate practical recommendations for Gen6."""
    
    print("\n" + "="*60)
    print("PRACTICAL GEN6 RECOMMENDATIONS")
    print("="*60)
    
    # Focus on recent, successful runs
    successful = recent[
        (recent['final_epoch'] >= 10) & 
        (recent['final_epoch'] <= 200)  # Reasonable range
    ].copy()
    
    if len(successful) == 0:
        print("No successful runs in reasonable range found!")
        return
    
    print(f"Based on {len(successful)} successful recent runs")
    
    # Calculate recommendations
    epochs = successful['final_epoch']
    
    # Conservative but practical approach
    p50 = np.percentile(epochs, 50)  # Median
    p75 = np.percentile(epochs, 75)  # 75th percentile
    p90 = np.percentile(epochs, 90)  # 90th percentile
    
    print(f"\nEpoch Distribution:")
    print(f"  50th percentile (median): {p50:.0f}")
    print(f"  75th percentile: {p75:.0f}")
    print(f"  90th percentile: {p90:.0f}")
    
    # Recommendations
    recommendations = {}
    
    # Conservative recommendation (75th percentile)
    recommendations['conservative'] = {
        'max_epochs': int(p75),
        'early_stopping_patience': max(15, int(p75 * 0.1)),
        'rationale': 'Covers 75% of historical runs, allows full convergence'
    }
    
    # Balanced recommendation (between 75th and 90th)
    balanced_epochs = int((p75 + p90) / 2)
    recommendations['balanced'] = {
        'max_epochs': balanced_epochs,
        'early_stopping_patience': max(20, int(balanced_epochs * 0.1)),
        'rationale': 'Balanced approach covering most cases without excessive training'
    }
    
    # Task-specific recommendations
    task_recs = {}
    for task in ['clean_ice', 'multiclass']:
        task_data = successful[successful['task'] == task]['final_epoch']
        if len(task_data) >= 3:
            task_p75 = np.percentile(task_data, 75)
            task_epochs = int(task_p75)
            task_patience = max(10, int(task_epochs * 0.1))
            
            task_recs[task] = {
                'max_epochs': task_epochs,
                'early_stopping_patience': task_patience,
                'sample_size': len(task_data),
                'median': task_data.median()
            }
    
    recommendations['task_specific'] = task_recs
    
    # Print recommendations
    print(f"\nRECOMMENDATION OPTIONS:")
    
    print(f"\n1. CONSERVATIVE (Recommended for production):")
    rec = recommendations['conservative']
    print(f"   max_epochs: {rec['max_epochs']}")
    print(f"   early_stopping_patience: {rec['early_stopping_patience']}")
    print(f"   Rationale: {rec['rationale']}")
    
    print(f"\n2. BALANCED (For experimentation):")
    rec = recommendations['balanced']
    print(f"   max_epochs: {rec['max_epochs']}")
    print(f"   early_stopping_patience: {rec['early_stopping_patience']}")
    print(f"   Rationale: {rec['rationale']}")
    
    print(f"\n3. TASK-SPECIFIC:")
    for task, rec in task_recs.items():
        print(f"   {task}:")
        print(f"     max_epochs: {rec['max_epochs']}")
        print(f"     early_stopping_patience: {rec['early_stopping_patience']}")
        print(f"     Based on {rec['sample_size']} runs (median: {rec['median']:.0f})")
    
    # Time estimates
    median_time_per_epoch = successful['time_per_epoch'].median()
    
    print(f"\nTIME ESTIMATES (based on {median_time_per_epoch:.4f} hrs/epoch):")
    for name, rec in [('Conservative', recommendations['conservative']), 
                     ('Balanced', recommendations['balanced'])]:
        total_time = rec['max_epochs'] * median_time_per_epoch
        print(f"  {name}: {total_time:.1f} hours ({total_time/24:.1f} days)")
    
    # Final recommendation
    print(f"\nFINAL RECOMMENDATION FOR GEN6:")
    final = recommendations['conservative']
    print(f"  max_epochs: {final['max_epochs']}")
    print(f"  early_stopping_patience: {final['early_stopping_patience']}")
    print(f"  Expected training time: {final['max_epochs'] * median_time_per_epoch:.1f} hours")
    
    # Additional insights
    print(f"\nKEY INSIGHTS:")
    
    # Check for overfitting evidence
    if 'final_val_CleanIce_iou' in successful.columns and 'best_val_CleanIce_iou' in successful.columns:
        perf_drop = successful['best_val_CleanIce_iou'] - successful['final_val_CleanIce_iou']
        significant_drop = (perf_drop > 0.01).sum()
        print(f"  - {significant_drop} runs showed >1% performance drop from peak to final")
        if significant_drop > 0:
            print(f"  - Early stopping could prevent this overfitting")
    
    # Convergence patterns
    early_finishers = (successful['final_epoch'] <= 50).sum()
    print(f"  - {early_finishers} runs finished early (≤50 epochs)")
    print(f"  - Consider progressive patience: start with 15, increase to 25 after epoch 50")
    
    return recommendations

def main():
    """Main analysis function."""
    print("Refined MLflow Analysis for Gen6 Configuration")
    print("="*60)
    
    # Load and clean data
    df = load_and_clean_data()
    print(f"Loaded {len(df)} cleaned runs")
    
    # Analyze meaningful runs
    meaningful, recent = analyze_meaningful_runs(df)
    
    # Analyze convergence patterns
    recent = analyze_convergence_patterns(meaningful, recent)
    
    # Analyze performance timing
    recent = analyze_performance_timing(recent)
    
    # Analyze training efficiency
    recent = analyze_training_efficiency(recent)
    
    # Generate recommendations
    recommendations = generate_practical_recommendations(recent)
    
    return df, recommendations

if __name__ == "__main__":
    df, recommendations = main()
