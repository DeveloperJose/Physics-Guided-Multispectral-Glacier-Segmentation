# Gen6 Experiment Goals and Rationale

## Primary Objectives

### 1. Resolve the Velocity Breakthrough Mystery
**Critical Question**: Can we replicate Gen2's +92% debris ice improvement (0.24→0.46 IoU)?

**Background**:
- Gen2: Massive +92% breakthrough with velocity-only dataset
- Gen3: Complete failure due to dataset preprocessing bug
- Gen4: Switched to physics+velocity to avoid the bug
- Gen5: Avoided velocity entirely

**Gen6 Strategy**: 
- Create comprehensive dataset with all channels
- Test velocity-only vs baseline vs physics vs synthesis
- Use consistent preprocessing to eliminate bugs

### 2. Optimize Channel Combinations Per Task
**Historical Performance Insights**:
```
Clean Ice:    Baseline (0.79) → Physics+0.025 (0.81) → Velocity (unknown)
Debris Ice:   Baseline (0.65) → Physics+0.03 (0.68) → Velocity+0.22 (0.46) 
Multi-class:  Baseline (0.53) → Physics+0.01 (0.54) → Velocity (unknown)
```

**Gen6 Matrix**:
- **Baseline**: Landsat+DEM+Spectral+HSV (16 channels)
- **Physics**: +Physics channels (20 channels) 
- **Velocity**: +Velocity channels (20 channels)
- **Synthesis**: +Physics+Velocity (24 channels)

### 3. Resolve Window Size Contradiction
**Gen1 vs Gen3 Conflict**:
```
Gen1: w512 (0.79) > w256 (0.78) for clean ice
Gen3: w256 (0.73) > w512 (0.70) for clean ice
```

**Gen6 Approach**: Test both w256 and w512 to definitively resolve optimal window size.

### 4. Standardize on Physics Scale (phys64_s1)
**Gen1 Physics Optimization**:
```
Clean Ice: phys64_s075 (0.81) > phys64_s1 (0.80) > phys32_s1 (0.78)
Debris Ice: phys64_s1 (0.68) > phys64_s075 (0.65) > phys128_s1 (0.66)
```

**Choice**: phys64_s1 (64m resolution, scale 1.0) - optimal for debris ice, excellent for clean ice.

## Technical Innovations

### 1. Comprehensive Dataset Strategy
**Single Dataset, Channel Selection**:
- One preprocessing run per window size
- All channels available for any experiment  
- Consistent data splits across all experiments
- Storage efficient vs multiple dataset variants

### 2. Training Pipeline Optimization
**Based on MLflow Analysis**:
- Reduced evaluations: 50% fewer test evaluations (5%→10% threshold)
- Fewer visualizations: 12→6 panels per evaluation
- Later baseline: epoch 15→30 for first test evaluation
- Expected speedup: 2-3x faster training

### 3. Hardware-Optimized Execution
**Server-Specific Strategy**:
```
Desktop (3060 Ti):   w256, quick validation, channel testing
Frodo (4x 2080 Ti): w512, parallel A/B testing, velocity breakthrough validation  
Bilbo (1x 4090):   w512, synthesis experiments, large batch sizes
```

## Success Metrics

### Minimum Requirements
- ✅ All channel combinations work without preprocessing bugs
- ✅ Velocity breakthrough validated (replicated or debunked)
- ✅ Optimal configurations identified per task
- ✅ Window size contradiction resolved

### Stretch Goals  
- 🎯 Replicate +92% velocity improvement for debris ice
- 🎯 Achieve >0.50 debris IoU with any configuration
- 🎯 Complete server-specific optimization strategies
- 🎯 2-3x training speedup while maintaining quality

## Critical Questions Gen6 Will Answer

### Primary Questions
1. **Velocity Breakthrough**: Real or fluke? Can we replicate +92% improvement?
2. **Optimal Channels**: Which combination works best for each task?
3. **Window Size**: Is w256 actually better than w512 for clean ice?
4. **Physics Scale**: Is phys64_s1 optimal across all tasks?

### Secondary Questions
5. **Task-Specific Optimal**: Different channel combos for clean vs debris vs multi-class?
6. **Server Performance**: How do different GPUs handle the workload?
7. **Training Efficiency**: Can we maintain quality with 2-3x speedup?

## Historical Context

### What Led to Gen6
**Gen1 (269 runs)**: Established baselines, discovered physics benefits
**Gen2 (Breakthrough)**: +92% velocity improvement for debris ice
**Gen3 (160 runs)**: Velocity dataset bug, physics success in multi-class
**Gen4 (Failed)**: Attempted physics+velocity synthesis
**Gen5 (Successful)**: Conservative baseline approach, reliable results

### Gen6 Synthesis
Combine the breakthrough potential of Gen2 with the reliability of Gen5, using the technical lessons from Gen3-4 to create a definitive, optimized experiment suite.

## Future Foundation

Gen6 will establish the definitive channel combinations, window sizes, and training strategies for production glacier mapping models, providing a robust foundation for Gen7 and beyond.

---

**Archive Date**: December 2025  
**Total Expected Runs**: ~50-70  
**Servers**: 3 (Desktop, Frodo, Bilbo)  
**Datasets**: 2 (w256, w512 comprehensive)  
**Channel Combinations**: 4 (baseline, physics, velocity, synthesis)