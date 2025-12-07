#!/usr/bin/env python3
"""
Focused MLflow analysis for Gen6 epoch and early stopping recommendations.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data():
    """Load and process MLflow data with focus on configurations."""
    
    # Load all generation files
    data_files = ['gen1_2_fresh.json', 'gen3_4_fresh.json', 'gen5_fresh.json']
    all_runs = []
    
    for file_path in data_files:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_runs.extend(data['runs'])
    
    processed_runs = []
    
    for run in all_runs:
        try:
            info = run['info']
            data = run['data']
            
            # Skip runs without end_time
            if 'end_time' not in info or info['end_time'] is None:
                continue
                
            run_info = {
                'run_id': info['run_uuid'],
                'run_name': info['run_name'],
                'status': info['status'],
                'start_time': info['start_time'],
                'end_time': info['end_time'],
                'duration_seconds': (info['end_time'] - info['start_time']) / 1000,
                'duration_hours': (info['end_time'] - info['start_time']) / (1000 * 60 * 60)
            }
            
            # Extract generation
            if 'gen1' in run_info['run_name'] or 'gen2' in run_info['run_name']:
                run_info['generation'] = 'Gen1-2'
            elif 'gen3' in run_info['run_name'] or 'gen4' in run_info['run_name']:
                run_info['generation'] = 'Gen3-4'
            elif 'gen5' in run_info['run_name']:
                run_info['generation'] = 'Gen5'
            else:
                run_info['generation'] = 'Unknown'
            
            # Extract task
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
            
            # Get final performance metrics
            for key in ['val_loss', 'val_CleanIce_iou', 'val_DebrisIce_iou', 'train_loss']:
                if key in metrics and metrics[key]:
                    run_info[f'final_{key}'] = metrics[key][-1][1]
            
            # Get best performance metrics
            for key in metrics:
                if key.startswith('val_') and metrics[key]:
                    if 'iou' in key or 'accuracy' in key:
                        best_val = max(v for s, v in metrics[key])
                        best_step = max(s for s, v in metrics[key] if v == best_val)
                        run_info[f'best_{key}'] = best_val
                        run_info[f'best_{key}_epoch'] = best_step
                    elif 'loss' in key:
                        best_val = min(v for s, v in metrics[key])
                        best_step = max(s for s, v in metrics[key] if v == best_val)
                        run_info[f'best_{key}'] = best_val
                        run_info[f'best_{key}_epoch'] = best_step
            
            # Extract configuration parameters
            params = {}
            for param in data.get('params', []):
                params[param['key']] = param['value']
            
            # Key config parameters
            config_keys = ['max_epochs', 'early_stopping_patience', 'batch_size', 'lr', 'dataset_name']
            for key in config_keys:
                if key in params:
                    run_info[f'config_{key}'] = params[key]
            
            processed_runs.append(run_info)
            
        except Exception as e:
            continue  # Skip problematic runs
    
    return pd.DataFrame(processed_runs)

def analyze_config_patterns(df):
    """Analyze configuration patterns and their impact."""
    
    print("="*80)
    print("CONFIGURATION PATTERN ANALYSIS")
    print("="*80)
    
    # Configuration parameters found
    config_cols = [col for col in df.columns if col.startswith('config_')]
    print(f"\nConfiguration parameters found: {[col.replace('config_', '') for col in config_cols]}")
    
    # max_epochs analysis
    if 'config_max_epochs' in df.columns:
        print(f"\nmax_epochs distribution:")
        epoch_dist = df['config_max_epochs'].value_counts().sort_index()
        print(epoch_dist)
        
        print(f"\nmax_epochs by generation:")
        epoch_by_gen = df.groupby('generation')['config_max_epochs'].value_counts().unstack(fill_value=0)
        print(epoch_by_gen)
    
    # early_stopping_patience analysis
    if 'config_early_stopping_patience' in df.columns:
        print(f"\nearly_stopping_patience distribution:")
        patience_dist = df['config_early_stopping_patience'].value_counts().sort_index()
        print(patience_dist)
        
        print(f"\nearly_stopping_patience by generation:")
        patience_by_gen = df.groupby('generation')['config_early_stopping_patience'].value_counts().unstack(fill_value=0)
        print(patience_by_gen)
    
    # Dataset analysis
    if 'config_dataset_name' in df.columns:
        print(f"\nDataset distribution:")
        dataset_dist = df['config_dataset_name'].value_counts().head(10)
        print(dataset_dist)

def analyze_convergence_patterns(df):
    """Analyze convergence patterns by generation and task."""
    
    print("\n" + "="*80)
    print("CONVERGENCE PATTERN ANALYSIS")
    print("="*80)
    
    # Filter to runs with meaningful epoch data
    meaningful_runs = df[
        (df['final_epoch'].notna()) & 
        (df['final_epoch'] > 0) & 
        (df['status'] == 'FINISHED')
    ].copy()
    
    print(f"\nAnalyzing {len(meaningful_runs)} meaningful runs")
    
    # Epoch statistics by generation
    print(f"\nFinal epoch statistics by generation:")
    epoch_stats = meaningful_runs.groupby('generation')['final_epoch'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(1)
    print(epoch_stats)
    
    # Add quartiles manually
    for gen in meaningful_runs['generation'].unique():
        gen_data = meaningful_runs[meaningful_runs['generation'] == gen]['final_epoch']
        if len(gen_data) > 0:
            q25 = gen_data.quantile(0.25)
            q75 = gen_data.quantile(0.75)
            print(f"{gen}: Q25={q25:.1f}, Q75={q75:.1f}")
    
    # Epoch statistics by task
    print(f"\nFinal epoch statistics by task:")
    task_epoch_stats = meaningful_runs.groupby('task')['final_epoch'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(1)
    print(task_epoch_stats)
    
    # Add quartiles for tasks
    for task in meaningful_runs['task'].unique():
        if task != 'unknown':
            task_data = meaningful_runs[meaningful_runs['task'] == task]['final_epoch']
            if len(task_data) > 0:
                q25 = task_data.quantile(0.25)
                q75 = task_data.quantile(0.75)
                print(f"{task}: Q25={q25:.1f}, Q75={q75:.1f}")
    
    # Combined generation-task analysis
    print(f"\nFinal epoch statistics by generation-task combination:")
    combo_stats = meaningful_runs.groupby(['generation', 'task'])['final_epoch'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(1)
    print(combo_stats)
    
    # Training time analysis
    meaningful_runs['time_per_epoch'] = meaningful_runs['duration_hours'] / meaningful_runs['final_epoch']
    
    print(f"\nTime per epoch (hours) by generation:")
    time_stats = meaningful_runs.groupby('generation')['time_per_epoch'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    print(time_stats)
    
    return meaningful_runs

def analyze_performance_vs_epochs(df):
    """Analyze when peak performance occurs."""
    
    print("\n" + "="*80)
    print("PERFORMANCE VS EPOCH ANALYSIS")
    print("="*80)
    
    # Best performance epoch analysis
    if 'best_val_CleanIce_iou_epoch' in df.columns:
        perf_data = df[df['best_val_CleanIce_iou_epoch'].notna()].copy()
        
        print(f"\nBest Val IoU epoch statistics:")
        best_epoch_stats = perf_data.groupby('generation')['best_val_CleanIce_iou_epoch'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(1)
        print(best_epoch_stats)
        
        # Performance at different epoch milestones
        print(f"\nPerformance analysis by epoch milestones:")
        
        # Calculate when 90% of final performance is reached
        for gen in perf_data['generation'].unique():
            gen_data = perf_data[perf_data['generation'] == gen]
            if len(gen_data) > 5:  # Only analyze if sufficient data
                # Estimate 90% convergence epoch
                gen_data['estimated_90pct_epoch'] = gen_data['final_epoch'] * 0.8
                print(f"\n{gen} - 90% performance epoch:")
                print(f"  Mean: {gen_data['estimated_90pct_epoch'].mean():.1f}")
                print(f"  Median: {gen_data['estimated_90pct_epoch'].median():.1f}")
                print(f"  Std: {gen_data['estimated_90pct_epoch'].std():.1f}")

def generate_gen6_recommendations(df):
    """Generate specific recommendations for Gen6."""
    
    print("\n" + "="*80)
    print("GEN6 RECOMMENDATIONS")
    print("="*80)
    
    # Filter to successful runs with meaningful data
    successful_runs = df[
        (df['status'] == 'FINISHED') & 
        (df['final_epoch'].notna()) & 
        (df['final_epoch'] > 10)
    ].copy()
    
    if len(successful_runs) == 0:
        print("No successful runs with meaningful epoch data found!")
        return
    
    recommendations = {}
    
    # Global recommendations based on all successful runs
    global_epochs = successful_runs['final_epoch']
    
    # Conservative approach: 75th percentile + 0.5*std
    p75 = np.percentile(global_epochs, 75)
    recommended_max_epochs = int(p75 + 0.5 * global_epochs.std())
    # Conservative approach: 10% of max_epochs or 15, whichever is larger
    recommended_patience = max(15, int(recommended_max_epochs * 0.1))
    
    recommendations['global'] = {
        'max_epochs': recommended_max_epochs,
        'early_stopping_patience': recommended_patience,
        'justification': f"Based on 75th percentile ({p75:.0f}) + 0.5*std ({global_epochs.std():.0f}) of {len(global_epochs)} successful runs"
    }
    
    # Task-specific recommendations
    task_recommendations = {}
    for task in ['clean_ice', 'debris_ice', 'multiclass']:
        task_data = successful_runs[successful_runs['task'] == task]
        if len(task_data) >= 3:  # Only recommend if sufficient data
            task_epochs = task_data['final_epoch']
            task_p75 = np.percentile(task_epochs, 75)
            task_max = int(task_p75 + 0.5 * task_epochs.std())
            task_patience = max(10, int(task_max * 0.1))
            
            task_recommendations[task] = {
                'max_epochs': task_max,
                'early_stopping_patience': task_patience,
                'sample_size': len(task_data),
                'median_epoch': task_epochs.median(),
                'std_epoch': task_epochs.std(),
                'p75_epoch': task_p75
            }
    
    recommendations['task_specific'] = task_recommendations
    
    # Generation-based insights
    gen_insights = {}
    for gen in ['Gen3-4', 'Gen5']:
        gen_data = successful_runs[successful_runs['generation'] == gen]
        if len(gen_data) >= 3:
            gen_insights[gen] = {
                'sample_size': len(gen_data),
                'median_epoch': gen_data['final_epoch'].median(),
                'mean_epoch': gen_data['final_epoch'].mean(),
                'std_epoch': gen_data['final_epoch'].std(),
                'p75_epoch': np.percentile(gen_data['final_epoch'], 75),
                'median_time_per_epoch': (gen_data['duration_hours'] / gen_data['final_epoch']).median()
            }
    
    recommendations['generation_insights'] = gen_insights
    
    # Print recommendations
    print(f"\nGLOBAL RECOMMENDATIONS FOR GEN6:")
    print(f"  max_epochs: {recommendations['global']['max_epochs']}")
    print(f"  early_stopping_patience: {recommendations['global']['early_stopping_patience']}")
    print(f"  Justification: {recommendations['global']['justification']}")
    
    print(f"\nTASK-SPECIFIC RECOMMENDATIONS:")
    for task, config in recommendations['task_specific'].items():
        print(f"  {task}:")
        print(f"    max_epochs: {config['max_epochs']}")
        print(f"    early_stopping_patience: {config['early_stopping_patience']}")
        print(f"    Based on {config['sample_size']} runs (median: {config['median_epoch']:.0f}±{config['std_epoch']:.0f}, P75: {config['p75_epoch']:.0f})")
    
    print(f"\nGENERATION INSIGHTS:")
    for gen, insights in recommendations['generation_insights'].items():
        print(f"  {gen}:")
        print(f"    {insights['sample_size']} runs, median {insights['median_epoch']:.0f}±{insights['std_epoch']:.0f} epochs (P75: {insights['p75_epoch']:.0f})")
        print(f"    {insights['median_time_per_epoch']:.3f} hours per epoch")
    
    # Final recommendation summary
    print(f"\nFINAL GEN6 CONFIGURATION RECOMMENDATIONS:")
    
    # Choose the most conservative (highest) recommendation
    if task_recommendations:
        conservative_max = max(config['max_epochs'] for config in task_recommendations.values())
        conservative_patience = max(config['early_stopping_patience'] for config in task_recommendations.values())
        
        final_max = max(recommendations['global']['max_epochs'], conservative_max)
        final_patience = max(recommendations['global']['early_stopping_patience'], conservative_patience)
    else:
        final_max = recommendations['global']['max_epochs']
        final_patience = recommendations['global']['early_stopping_patience']
    
    print(f"  RECOMMENDED max_epochs: {final_max}")
    print(f"  RECOMMENDED early_stopping_patience: {final_patience}")
    
    # Additional insights
    print(f"\nADDITIONAL INSIGHTS:")
    
    # Check for overfitting patterns
    if 'final_val_CleanIce_iou' in df.columns and 'best_val_CleanIce_iou' in df.columns:
        overfit_data = df[
            (df['final_val_CleanIce_iou'].notna()) & 
            (df['best_val_CleanIce_iou'].notna())
        ].copy()
        overfit_data['performance_drop'] = overfit_data['best_val_CleanIce_iou'] - overfit_data['final_val_CleanIce_iou']
        
        significant_overfit = overfit_data[overfit_data['performance_drop'] > 0.02]
        if len(significant_overfit) > 0:
            print(f"  - {len(significant_overfit)} runs showed >2% performance drop from best to final")
            print(f"  - Average drop: {significant_overfit['performance_drop'].mean():.3f}")
            print(f"  - This suggests early stopping could prevent overfitting")
    
    # Time efficiency analysis
    if len(successful_runs) > 0:
        avg_time_per_epoch = (successful_runs['duration_hours'] / successful_runs['final_epoch']).median()
        total_estimated_time = final_max * avg_time_per_epoch
        print(f"  - Estimated training time: {total_estimated_time:.1f} hours ({total_estimated_time/24:.1f} days)")
        print(f"  - Based on {avg_time_per_epoch:.3f} hours per epoch median")
    
    recommendations['final'] = {
        'max_epochs': final_max,
        'early_stopping_patience': final_patience
    }
    
    return recommendations

def main():
    """Main analysis function."""
    print("Focused MLflow Analysis for Gen6 Configuration")
    print("="*80)
    
    # Load and process data
    df = load_and_process_data()
    print(f"Loaded {len(df)} runs")
    
    # Analyze configuration patterns
    analyze_config_patterns(df)
    
    # Analyze convergence patterns
    meaningful_df = analyze_convergence_patterns(df)
    
    # Analyze performance vs epochs
    analyze_performance_vs_epochs(meaningful_df)
    
    # Generate recommendations
    recommendations = generate_gen6_recommendations(meaningful_df)
    
    return df, recommendations

if __name__ == "__main__":
    df, recommendations = main()
