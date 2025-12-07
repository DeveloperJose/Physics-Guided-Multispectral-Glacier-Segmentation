# Gen6 Configuration Summary

## ✅ **Configuration Updates Complete**

### **Optimized Test Evaluation Parameters**
- **Baseline Epoch**: 35 (skip first 35 epochs)
- **Aggressive Threshold**: 15% (require significant improvement in phase 2)  
- **Transition Epoch**: 130 (switch to phase 3 late)
- **Expected Test Evaluations**: ~3-5 per run (vs current 1)

### **Configuration Mismatch Fixed**
- Updated `train.py` defaults to match `train.yaml` values
- No more unintended conservative defaults (15, 0.05, 50)

## 🏗️ **Gen6 Configuration Structure Created**

### **Desktop (3060 Ti) - Quick Validation**
```
configs/desktop/clean_ice/
├── base_w256_gen6.yaml      # 256px baseline breakthrough
├── physics_w256_gen6.yaml    # Physics + 256px
└── velocity_w256_gen6.yaml   # Velocity mystery test

configs/desktop/debris_ice/
└── base_w256_gen6.yaml      # Quick debris baseline
```

### **Frodo (4x 2080 Ti) - Parallel Breakthrough Testing**
```
configs/frodo/clean_ice/
├── base_w512_gen6_gpu0.yaml      # GPU 0: CI baseline 512px
├── velocity_w512_gen6_gpu0.yaml   # GPU 0: ⭐ VELOCITY BREAKTHROUGH
├── physics_w512_gen6_gpu0.yaml    # GPU 0: Physics optimization
├── synthesis_w512_gen6_gpu0.yaml  # GPU 0: Physics+Velocity
└── base_w256_gen6_gpu0.yaml      # GPU 0: CI baseline 256px

configs/frodo/debris_ice/
├── base_w512_gen6_gpu1.yaml      # GPU 1: DCI baseline  
├── velocity_w512_gen6_gpu1.yaml   # GPU 1: ⭐ VELOCITY BREAKTHROUGH
├── physics_w512_gen6_gpu1.yaml    # GPU 1: Physics optimization
└── synthesis_w512_gen6_gpu1.yaml  # GPU 1: Physics+Velocity

configs/frodo/multiclass/
├── base_w512_gen6_gpu2.yaml      # GPU 2: Multi baseline
├── velocity_w512_gen6_gpu2.yaml   # GPU 2: ⭐ VELOCITY BREAKTHROUGH  
├── physics_w512_gen6_gpu2.yaml    # GPU 2: Physics optimization
└── synthesis_w512_gen6_gpu2.yaml  # GPU 2: Physics+Velocity

configs/frodo/clean_ice/
├── base_w256_gen6_gpu3.yaml      # GPU 3: CI baseline 256px
├── dci_base_w256_gen6_gpu3.yaml   # GPU 3: DCI baseline 256px
└── multi_base_w256_gen6_gpu3.yaml  # GPU 3: Multi baseline 256px
```

### **Bilbo (1x 4090) - Large Batch Synthesis**
```
configs/bilbo/debris_ice/
└── physics_w512_gen6_bilbo.yaml  # Large batch physics

configs/bilbo/multiclass/
└── synthesis_w512_gen6_bilbo.yaml # Large batch synthesis
```

## 🎯 **Gen6 Experimental Strategy**

### **Critical Experiments**
1. **Velocity Mystery Resolution** (GPU 1 & 2)
   - Replicate or debunk Gen2's +92% breakthrough (0.24→0.46 IoU)
   - Test on clean ice, debris ice, and multi-class

2. **256px Breakthrough Validation** (Desktop + GPU 3)
   - Replicate Gen3's 0.73+ IoU for clean ice
   - Test efficiency vs 512px

3. **Physics Optimization** (All servers)
   - Build on Gen5's success with physics channels
   - Test physics+velocity synthesis

4. **Synthesis Experiments** (GPU 1, 2, Bilbo)
   - Combine physics + velocity channels
   - Test if combination stabilizes and enhances performance

### **Server-Specific Workflow**

**Desktop**: ⚠️ **DEPRECATED** - Moved to Frodo GPU 0 (faster 2080 Ti)
**Frodo GPU 0**: Clean ice full suite (baseline + velocity + physics + synthesis + 256px)
**Frodo GPU 1**: Debris ice experiments (velocity priority)
**Frodo GPU 2**: Multi-class experiments  
**Frodo GPU 3**: 256px validation + baselines
**Bilbo**: Large batch synthesis experiments

### **📊 GPU Performance Rationale**
- **RTX 2080 Ti**: 5,342 CUDA cores, 11GB VRAM, 616 GB/s bandwidth
- **RTX 3060 Ti**: 4,864 CUDA cores, 8GB VRAM, 448 GB/s bandwidth  
- **Performance**: 2080 Ti is ~15-25% faster for deep learning
- **Decision**: Move desktop configs to Frodo GPU 0 for better utilization

## 📊 **Expected Performance Improvements**

### **Training Efficiency**
- **Test Evaluations**: ~3-5 per run (vs current 1)
- **Training Time**: ~22% faster with optimized parameters
- **GPU Utilization**: Better parallel processing on Frodo

### **Success Targets**
- **Clean Ice**: Replicate 0.73+ IoU (256px breakthrough)
- **Debris Ice**: Resolve velocity mystery (target: 0.46+ IoU)
- **Multi-class**: Stable physics+velocity synthesis
- **Efficiency**: Maintain quality with 22% speedup

## 🚀 **Ready for Execution**

### **Phase 1: Desktop Validation**
```bash
./run_sequential_training.sh desktop --tasks ci,dci --priority ci,dci
```

### **Phase 2: Frodo Parallel Testing**  
```bash
# GPU 0 (Clean Ice - Full Suite)
./run_sequential_training.sh frodo --gpu 0 --tasks ci --priority ci

# GPU 1 (Debris Ice - Velocity Priority)
./run_sequential_training.sh frodo --gpu 1 --tasks dci --priority dci

# GPU 2 (Multi-class)
./run_sequential_training.sh frodo --gpu 2 --tasks multi --priority multi

# GPU 3 (256px Validation + Baselines)
./run_sequential_training.sh frodo --gpu 3 --tasks ci,dci,multi --priority ci,dci,multi
```

### **Phase 3: Bilbo Synthesis**
```bash
./run_sequential_training.sh bilbo --gpu 1 --tasks dci,multi --priority dci,multi
```

## 🔧 **Technical Implementation**

- **Dataset**: Comprehensive `bibek_*_comprehensive_phys64_s1` with channel selection
- **Channels**: Baseline (16), Physics (+4), Velocity (+4), Synthesis (+8)
- **Window Sizes**: 256px (efficiency), 512px (standard)
- **Test Strategy**: 3-phase with optimized parameters for ~5 evaluations/run

---
**Status**: ✅ Gen6 configurations complete and ready for execution
**Total Configs**: 20 core experiments across 3 servers (4 added to Frodo GPU 0)
**Priority**: Velocity mystery resolution, 256px validation, physics optimization