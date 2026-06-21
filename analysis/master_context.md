# HKH Glacier Mapping — Master Context

Consolidated general findings as of 2026-06-20.

Repo:
- `/home/devj/local-arch/code/glacier-mapping`

Core external references:
- LILA HKH dataset page: https://lila.science/datasets/hkh-glacier-mapping/
- LILA tarball source: https://storage.googleapis.com/public-datasets-lila/icimod-glacier-mapping/hkh_patches.tar.gz
- Original benchmark repo referenced by LILA page: https://github.com/krisrs1128/glacier_mapping
- ICIMOD metadata page for HKH glacier inventory: https://rds.icimod.org/metadata/f319036a-64ab-4583-a1bc-5df7b67119a7
- ICIMOD report landing page cited by LILA: http://rds.icimod.org/Home/DataDetail?metadataId=9359&searchlist=True
- ICIMOD report citation target: http://lib.icimod.org/record/9419

---

## 1. Big decisions

Current working strategy has 2 dataset tracks:
1. **Track A — Benchmark-faithful rebuild**: Replicate LILA released dataset's scene-by-ID approach from Python EE. Use the same 35 scene IDs (from `Aryal007/GEE_landsat_7_query_tiles/ids.js` analysis_image_ids_2). Reproduce scene-by-scene with gapfill or median compositing. Labels from LILA `clean.shp` + `debris.shp`.
2. **Track B — Improved rebuild**: Same scene IDs but with better compositing (median instead of single-scene) and optional extra channels (terrain features from ch.3, velocity from ch.4).

### Decision rationale

Two GEE codebases exist with different strategies:

**`Aryal007/GEE_landsat_7_query_tiles`** (LILA paper source)
- **Strategy**: Hardcoded scene IDs → download each scene with gapfill for SLC-off → add indices → full-scene export → tile offline
- Scene groups: `analysis_image_ids_2` (35 scenes = the 35 LILA GeoTIFFs), `analysis_image_ids_1` (45 scenes), `correction_image_ids` (23 scenes for gapfill)
- Uses `users/bibekaryal7/fall2020:gapfill.js` and `users/bibekaryal7/fall2020:calculate_indices.js`
- Reproducible: scene IDs are fixed, but collection is C01 (needs migration to C02)

**`krisrs1128/glacier_mapping/scripts/ee_code/`** (Bibek boundary-aware paper)
- **Strategy**: Dynamic GEE query by date/cloud bounds → mosaic all labeling scenes into single raster → clip to fishnet → tile
- Per the paper: "We then created a **mosaic** of all Landsat 7 images used for labeling into a single raster and clipped the raster mosaic to country boundaries"
- This mosaic approach is what causes the qualityMosaic streaking artifacts we observed
- Less reproducible: depends on GEE collection state at query time

**Recommendation**: Track A should anchor to `Aryal007/GEE_landsat_7_query_tiles` scene-by-ID approach. It's more deterministic, matches released LILA artifacts, and provides a cleaner baseline for improvements.

### LILA dataset performance with current code

Converting the released LILA format → repo processed format and running with our improved architecture (SMP Unet, better loss, etc.) gives strong baseline numbers on the LILA test set:
- **CI IoU**: 0.7226
- **DCI IoU**: 0.4984

These exceed Bibek's boundary-aware paper (CI 68.17%, DCI 35.94%) and approach dissertation chapter 3 results (CI 71.22%, DCI 45.92%) — with only 8 Landsat bands, no terrain/velocity features. The architecture/code improvements account for this gap.

### Other key decisions
- Keep **Python GEE exporter** as main rebuild path.
- Do **not** full-export all 202 tiles yet; current sweep mosaics still show visual artifact issues.
- For all future GEE exports, use unique Drive folders/prefixes to avoid collisions.
- Reference PDFs saved to `analysis/references/`:
  - `bibek_thesis.txt` — Aryal 2022 thesis
  - `bibek_boundary_aware.txt` — Boundary-aware U-Net paper (arXiv:2301.11454)
  - `lila_paper.txt` — LILA dataset paper (arXiv:2012.05013)
  - `icimod_report.txt` — ICIMOD HKH glacier status report

---

## 2. Label provenance and benchmark truth

### Best benchmark-faithful raw label source now known
Local LILA extraction:
- `/home/devj/local-arch/data/HKH_raw/LILA/glacier_data/vector_data/clean.shp`
- `/home/devj/local-arch/data/HKH_raw/LILA/glacier_data/vector_data/debris.shp`
- `/home/devj/local-arch/data/HKH_raw/LILA/glacier_data/vector_data/hkh.shp`
- `/home/devj/local-arch/data/HKH_raw/LILA/glacier_data/README.md`

LILA README states:
- `hkh.shp` corresponds to `Glacier_2005.shp` from ICIMOD RDS
- clean ice and debris-covered polygons were separated into `clean.shp` and `debris.shp`

### Exact comparison results
Using GLIMS IDs:
- `clean.shp` unique GLIMS IDs: `28,530`
- `debris.shp` unique GLIMS IDs: `1,523`
- `clean ∪ debris`: `28,542`
- current working HKH label file unique GLIMS IDs: `28,542`
- `our minus union`: `0`
- `union minus our`: `0`

Interpretation:
- current working HKH benchmark labels and LILA CI/DC split share same GLIMS ID universe
- current merged HKH CIDC labels are effectively merged/derived representation of same benchmark label universe preserved explicitly in LILA

Important nuance:
- `clean ∩ debris = 1511`
- clean and debris are component polygons, not glacier-level mutually exclusive glacier IDs
- this matches ICIMOD report language that CI and DC were delineated separately and later merged

### Area sanity check
- `clean.shp` total area: `27870.69 km²`
- `debris.shp` total area: `3063.95 km²`
- current merged HKH CIDC area: `30934.63 km²`
- `clean + debris ≈ merged total`

Conclusion:
- Original benchmark GEE scripts use `users/naryal7/Glacier_HKH` asset (same HKH label source, user `naryal7` not `bibekaryal7`)
- future benchmark-faithful rasterization should prefer LILA `clean.shp` + `debris.shp`
- not public `HKH_Glaciers.shp`
- merged `HKH_CIDC_5basins_all.shp` remains useful for compatibility checks

---

## 3. Public ICIMOD product vs benchmark labels

Public ICIMOD product downloaded locally:
- `/home/devj/local-arch/data/HKH_raw/ICIMOD_Status_of_Glaciers_HKH/data/HKH_Glaciers.shp`

Observed properties:
- 38,259 records
- includes `GLIMS_ID`, `Class`, `Area_SqKm`, terrain/basin fields
- `Class` is morphological/TTS-style class code, not direct clean/debris label
- does not expose benchmark CI/DC split like LILA `clean.shp` / `debris.shp`

Supporting report evidence:
- `/home/devj/Downloads/icimod-the_status_of_glaciers_in_the_hindu_kush-himalayan_region[1].pdf`
- report confirms CI and DC were delineated separately, then merged later
- `Class` is morphological, not CI/DC label
- China portion used different methodology and did not distinguish CI/DC same way

Conclusion:
- public `HKH_Glaciers.shp` is provenance evidence, not best benchmark label source

---

## 4. Fishnet and tile inventory

Earth Engine fishnet asset:
- `users/bibekaryal7/HKH/fishnet_clip`

Exported locally:
- `google_earth_scripts/hkh_fishnet.geojson`
- `google_earth_scripts/hkh_fishnet_summary.json`

Key facts:
- feature count: `202`
- each feature includes `_export_index`
- `_export_index` maps to legacy `image{i}.tif` naming

Tile inventory:
- `google_earth_scripts/tile_inventory.csv`
- `google_earth_scripts/tile_inventory_summary.json`

Inventory summary:
- fishnet tiles: `202`
- Landsat tiles found: `202`
- DEM tiles found: `202`
- velocity tiles found: `198`
- missing velocity tiles:
  - `image54.tif`
  - `image101.tif`
  - `image135.tif`
  - `image142.tif`

Legacy raw paths:
- Landsat: `/home/devj/local-arch/data/HKH_raw/Landsat7_2005/`
- DEM: `/home/devj/local-arch/data/HKH_raw/DEM/`
- Velocity: `/home/devj/local-arch/data/HKH_raw/Velocity/`
- Legacy merged labels: `/home/devj/local-arch/data/HKH_raw/labels/HKH_CIDC_5basins_all.shp`
- Geometry-fixed labels: `/home/devj/local-arch/data/HKH_raw/labels_fixed/HKH_CIDC_5basins_all.shp`

---

## 5. Old JS, Python rebuild, and provenance status

Historically relevant JS files:
- `google_earth_scripts/ids.js`
- `google_earth_scripts/Landsat7_2005.js`
- `google_earth_scripts/gapfill.js`
- `google_earth_scripts/get_dem.js`

Python rebuild/export path:
- `google_earth_scripts/export_hkh_rebuild.py`

Conclusions:
- old JS is useful historical evidence, not trustworthy final reproducible basis
- exact C01 replay is not practical now
- provenance in old JS / asset environment remains incomplete
- keep new EE Python exporter with explicit manifests and per-tile metadata
- **Original benchmark GEE scripts (`scripts/ee_code/landsat-7-2005-hkh`) are the actual source queries for the benchmark imagery.** They can serve as the historical anchor for benchmark-faithful rebuild.

Earth Engine Python API status:
- authenticated and working on this machine
- verified with `ee.Initialize(project="hkh-glacier-mapping")`

---

## 6. GEE rebuild sweeps and imagery audit findings

Downloaded rebuild/sweep outputs organized at:
- `/home/devj/local-arch/data/HKH_raw/rebuild/Landsat7_C02_T1`
- `/home/devj/local-arch/data/HKH_raw/rebuild/DEM_NASADEM`
- `/home/devj/local-arch/data/HKH_raw/rebuild/sweep_A/`
- `/home/devj/local-arch/data/HKH_raw/rebuild/sweep_B/`
- `/home/devj/local-arch/data/HKH_raw/rebuild/sweep_C/`

Relevant manifests/policies under:
- `google_earth_scripts/export_manifests/hkh_rebuild_audit_14/`
- `google_earth_scripts/export_manifests/hkh_rebuild_sample/`
- `google_earth_scripts/export_manifests/sweep_A/`
- `google_earth_scripts/export_manifests/sweep_B/`
- `google_earth_scripts/export_manifests/sweep_C/`

Audit scripts/results:
- `scratch_audit.py`
- `scratch_sweep_audit.py`
- `scratch_sweep_audit_full14.py`
- `/tmp/sweep_audit/summary.json`
- `/tmp/sweep_audit_full14/summary.json`
- `/tmp/sweep_audit_full14/aggregate.json`
- `/tmp/sweep_audit_full14/tile*_comparison.png`

Key findings:
- `24 scenes/tile` clearly better than `8 scenes/tile` for coverage rescue
- sweeps `A/B/C` are numerically very similar
- `sweep_C` is slightly best numerically overall
- but user visual inspection found noticeable streaking/artifacts that metrics underweighted
- current broad `qualityMosaic`-style compositing is therefore not final

Interpretation:
- numeric coverage alone is not enough
- visual coherence must be audited for all future imagery experiments
- next GEE work should stay on small audited subsets, not full 202-tile export

### Quantified streaking evidence

Measured per-pixel date jitter via 5×5 coefficient of variation (CV) of NIR band:

| Tile | Single scene (old) | 8-scene qualityMosaic | 24-scene qualityMosaic |
|------|:---:|:---:|:---:|
| 24 | 0.183 | 0.225 (+23%) | 0.260 (+42%) |
| 31 | 0.192 | 0.214 (+12%) | 0.259 (+35%) |
| 96 | 0.244 | 0.251 (+3%) | 0.277 (+13%) |
| 132 | 0.203 | 0.217 (+7%) | 0.274 (+35%) |

Pattern: more qualityMosaic scenes → higher local heterogeneity (streaking). Each pixel picks its own winner scene → adjacent pixels from different dates → visible discontinuities.

### Proposed compositing alternatives

| Mode | Streak risk | SLC gaps | Reasoning |
|------|:-:|:-:|----------|
| `quality_mosaic` (current) | high | filled | Per-pixel winner = streaking |
| `best_scene` | none | yes | Single scene = coherent, matches original benchmark methodology |
| `median` | none | filled | Per-pixel median of all scenes. No winner-take-all. Standard Landsat practice |

Experiment: 3 modes × 5 problematic tiles (24, 31, 96, 131, 132) on GEE, visual + numerical audit to pick improved default.

---

## 7. Original benchmark repo and published baselines

Original benchmark pipeline repo:
- `https://github.com/krisrs1128/glacier_mapping/`

This repo contains the exact GEE scene-selection queries used to build the original benchmark:
- `scripts/ee_code/landsat-7-2005-hkh` — main Landsat 7 2005 query
- `scripts/ee_code/landsat-7-2005-hkh-missing` — gap fill / missing tiles
- `scripts/ee_code/landsat-7-images-used-for-labelling-2000-nepal`
- `scripts/ee_code/landsat-7-images-used-for-labelling-2010-nepal`
- `scripts/ee_code/landsat-7-remaining-images-2000-bhutan`
- `scripts/ee_code/landsat-7-remaining-images-2000-nepal`

It also uses `users/naryal7/Glacier_HKH` as vector label asset (note: `naryal7`, not `bibekaryal7` — user name changed).

### Published LILA paper benchmark IoU

From the LILA dataset paper, the published baselines on the released benchmark:

| Model | Glacier IoU | CI IoU | DCI IoU |
|-------|:-----------:|:------:|:-------:|
| Binary (glacier vs BG) | 0.476 | — | — |
| Multiclass (CI/DCI/BG) | 0.473 | 0.456 | 0.291 |
| Two binary models (CI/DCI separately) | 0.48 | 0.476 | 0.310 |
| U-Net (separate eval per the paper) | — | 0.5829 | 0.3707 |

Key notes from the paper:
- They trained 2-class (glacier vs background) and 3-class (CI vs DCI vs BG) models.
- They also compared the 3-class against two binary models (one per class).
- They filtered to patches where both DCI and CI were present: 648 train + 93 val patches.
- IoU is evaluated over the whole validation set (not per-patch mean) to avoid bias from sparse positive patches.
- Multiclass and binary models deliver comparable overall performance, but multiclass outperforms in debris-heavy regions.

### Our best DCI IoU on comprehensive V3 dataset

From recent training runs on `comprehensive_v3_landsat_dem_flowacc_velmag` (18 runs with test eval):

| Run | DCI IoU | Precision | Recall |
|-----|:------:|:---------:|:------:|
| Best | 0.5493 | 0.7588 | 0.6655 |
| Top-5 avg | 0.5407 | 0.7127 | 0.6960 |

These are not comparable to LILA baseline numbers (different dataset, different splits). They provide context for current model capability with richer inputs (15 bands incl. DEM/indices). LILA baseline runs will produce directly comparable numbers.

### Vector data sources documented in original repo

The original benchmark README documents where labels came from per region/year:
- (2000, Nepal): ICIMOD, filtered to ±2 years from 2000
- (2000, Bhutan): ICIMOD, used as-is
- (2010, Nepal): ICIMOD, filtered to ±2 years from 2010
- (2010, Bhutan): ICIMOD, used as-is

This confirms the stratified approach: Nepal labels were temporally filtered to match the target year, while Bhutan labels were used without filtering.

---

## 8. Published benchmark baselines and dissertation IoU progression

Full reference texts extracted to:
- `analysis/references/bibek_thesis.txt` — Aryal 2022 thesis
- `analysis/references/lila_paper.txt` — LILA dataset paper (arXiv:2012.05013)
- `analysis/references/icimod_report.txt` — ICIMOD HKH glacier status report
- `dissertation/` — LaTeX source for the dissertation (chapters 3-4 contain published results)

### LILA paper baselines (2020)

Table 1 — U-Net vs traditional ML on 55 validation patches:

| Model | CI IoU | DCI IoU |
|-------|:------:|:-------:|
| Random Forest | 0.5807 | 0.2024 |
| Gradient Boosting | 0.5663 | 0.1930 |
| MLP | 0.5452 | 0.1781 |
| U-Net | **0.5829** | **0.3707** |

Table 2 — Binary vs multiclass modelling:

| Model | Glacier IoU | CI IoU | DCI IoU |
|-------|:-----------:|:------:|:-------:|
| Binary (glacier vs BG) | 0.476 | — | — |
| Multiclass (CI/DCI/BG) | 0.473 | 0.456 | 0.291 |
| Two binary models | 0.48 | 0.476 | 0.310 |

Key note: U-Net row in Table 1 reports higher CI/DCI IoU than Table 2 because Table 2 is a separate experiment comparing modelling strategies on a different subset/filtering.

### Bibek thesis baselines (Aryal 2022)

Table 5.3 — Combined loss and self-learning boundary-aware loss (LSLBA) on 8 Landsat 7 features:

| Loss | α | CI IoU | DCI IoU |
|-----|:-:|:------:|:-------:|
| LCombined | 0.0 | 0.25% | 3.00% |
| LCombined | 0.1 | 66.31% | 33.43% |
| LCombined | 0.5 | 68.33% | 32.41% |
| LCombined | 0.9 | 68.33% | 32.41% |
| LCombined | 1.0 | 67.34% | 29.05% |
| **LSLBA** (SOTA) | Dynamic | **68.17%** | **35.94%** |

LSLBA result was the state-of-the-art for DCI segmentation before the dissertation work.

### Dissertation Chapter 3 — Physics-informed data augmentation

Table `tab:combined_results` — Adding terrain-based physics features (flow accumulation, TPI, roughness, curvature) to 8 Landsat + DEM:

| Model | DCI IoU | CI IoU |
|-------|:-------:|:------:|
| Standard U-Net | 28.50% | 65.60% |
| Boundary-Aware SOTA (LSLBA) | 35.94% | **68.17%** |
| Ours (Flow Only) | 38.50% | 63.50% |
| **Ours (Full Physics)** | **45.92%** | **71.22%** |

Full Physics = +9.98pp (27.8% relative) improvement over previous SOTA on DCI.

### Dissertation Chapter 4 — Velocity-informed physics

Table `tab:ch4_combined_results` — Adding ITS_LIVE velocity data and velocity loss:

| Model | DCI IoU | CI IoU |
|-------|:-------:|:------:|
| Standard U-Net | 28.50% | 65.60% |
| Boundary-Aware SOTA (LSLBA) | 35.94% | **68.17%** |
| Ours (Velocity Channels Only) | 32.40% | 70.78% |
| Ours (Velocity Channels + Loss) | 41.91% | 61.83% |
| **Ours (Complete Physics-Informed)** | **46.07%** | 65.85% |

Complete Physics = DCI IoU 46.07%, representing 28.2% relative improvement over previous SOTA. Combines terrain features (ch.3) + velocity channels + velocity loss.

### Performance summary across all methods

| Source | Model | Inputs | DCI IoU | CI IoU |
|--------|-------|--------|:-------:|:------:|
| LILA paper | U-Net | 8 Landsat bands | 37.07% | 58.29% |
| Bibek thesis | LSLBA U-Net | 8 Landsat bands | 35.94% | 68.17% |
| Diss. ch.3 | Full Physics U-Net | 8 Landsat + DEM + 4 terrain | **45.92%** | **71.22%** |
| Diss. ch.4 | Complete Physics U-Net | Above + 4 velocity + vloss | **46.07%** | 65.85% |

### Best post-dissertation-code runs on Bibek data (`output/`)

These are from local `output/*/test_evaluations/test_metrics.json` and use Bibek-data-derived processed datasets with newer/post-dissertation code and architectures.

| Category | Run | DCI IoU | CI IoU |
|----------|-----|:-------:|:------:|
| Best binary DCI | `sota_dci_32_01_L11_seed42_desktop_20260619_060548` | **54.93%** | — |
| Best binary CI | `sota_dci_08_smp_dlv3p_rn34_ci_desktop_20260613_122942` | — | **71.72%** |
| Best multiclass joint | `sota_dci_08_smp_unet_rn34_multiclass_desktop_20260614_002729` | **53.08%** | **72.50%** |

Context against released LILA baselines with current code:

| Dataset/run family | DCI IoU | CI IoU |
|--------------------|:-------:|:------:|
| Released LILA binary baselines | **49.84%** | **72.26%** |
| Released LILA multiclass baseline | 50.06% | 69.77% |

Takeaway:
- Best DCI overall in local `output/` on Bibek data is **54.93%**.
- Best CI overall in local `output/` on Bibek data is **72.50%** from multiclass.
- Best joint multiclass on Bibek data is **CI 72.50% / DCI 53.08%**.
- Released LILA binary baseline is already very strong (**CI 72.26% / DCI 49.84%**) without dissertation extra channels.

---

## 9. LILA released dataset and benchmark baseline conversion

Inspected LILA release contents under:
- `/home/devj/local-arch/data/HKH_raw/LILA/glacier_data/`

Contains:
- `raster_data/` with 35 GeoTIFFs, 15 bands each
- `vector_data/clean.shp`, `debris.shp`, `hkh.shp`
- `masks/`, `slices/`, `splits/`

Important scene-ID comparison:
- LILA unique source IDs: `35`
- old JS `ids.js` IDs: `41`
- intersection: `31`

Conclusion:
- `ids.js` is **not** identical to released LILA imagery provenance
- released LILA artifacts outrank mismatching later JS when reconstructing benchmark packaging

Created conversion script:
- `scripts/convert_lila_to_processed.py`

Converted released LILA split dataset into repo processed format at:
- `/home/devj/local-arch/data/HKH/lila_released_v1`

Important converted band order:
- `[
  "B1","B2","B3","B4","B5","B6_VCID1","B6_VCID2","B7","B8","BQA",
  "NDVI","NDSI","NDWI","elevation","slope_deg"
  ]`

Converted split shapes:
- train `(383, 15, 512, 512)`
- val `(55, 15, 512, 512)`
- test `(110, 15, 512, 512)`

Important mask interpretation finding:
- released LILA split masks behave as one-hot `[clean, debris, other/background]`
- they are **not** usable as `[clean, debris, HKH-ignore]` for those split arrays

Bug fixed during conversion:
- original error: `IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed`
- fixed in `compute_stats()` indexing

---

## 10. Training baseline setup status

Created LILA baseline configs:
- `configs/desktop/debris_ice/lila_dci_baseline_bs8_seed42.yaml`
- `configs/desktop/clean_ice/lila_ci_baseline_bs8_seed42.yaml`
- `configs/desktop/multiclass/lila_multiclass_baseline_bs8_seed42.yaml`

Initial training failure found:
- all runs errored during model construction, not data loading
- error: `TypeError: Unet.__init__() missing 1 required positional argument: 'net_depth'`
- root cause: config/model mismatch from using wrong baseline family

Correction made:
- switched these baseline configs to stronger recent **SMP Unet** style instead of old custom-Unet arg pattern
- baseline now uses recent-output style settings with `framework: "smp"`, `name: "Unet"`, `encoder_name: "resnet18"`, `encoder_weights: null`, `channels_last: true`, and same logvar settings used in stronger current runs

User must execute training. Agent should only prepare/debug configs.

---

## 11. Velocity audit conclusions

Patched files:
- `scripts/create_velocity_from_itslive_mosaic.py`
- `glacier_mapping/data/slice.py`

Established findings:
- ITS_LIVE coordinate transform bug was real and fixed
- redundant second reprojection was correctly skipped when grids already match
- correctness fixes should stay
- velocity still does **not** look like strong CI-vs-DCI discriminator in current setup
- velocity behaves more like glacier-context / glacier-vs-background signal
- pooled/native-support audits did not rescue CI-vs-DCI separation
- do not resume future work with velocity loss first

Saved related analyses:
- `analysis/velocity_audit_summary_for_llm.md`
- `analysis/velocity_transform_sample_results.json`
- `analysis/velocity_audit20_results.json`
- `analysis/velocity_representation_audit.json`
- `analysis/velocity_native_support_audit.json`

---

## 12. Other relevant local glacier data

Older QGIS workspace:
- `/home/devj/local-arch/data/glacier-qgis`

Notable contents:
- `/home/devj/local-arch/data/glacier-qgis/Clean-ice and Debris-covered glaciers of Bhutan for 2010/data/Bhutan_CIDC_2010.shp`
- `/home/devj/local-arch/data/glacier-qgis/The Status of Glaciers in the Hindu Kush Himalayan Region/data/HKH_Glaciers.shp`
- `/home/devj/local-arch/data/glacier-qgis/debug/labels/HKH_CIDC_5basins_all.qml`

Bhutan CIDC evidence remains supporting context, not primary benchmark source.

---

## 13. Operational constraints

Project instructions:
- use `uv` for python execution
- no raw `pip` / `python`
- agents may inspect, analyze, prepare, edit, test, and predict
- agents may **not** run training
- user runs `scripts/train.py` and sequential training scripts

Useful repo paths:
- repo root: `/home/devj/local-arch/code/glacier-mapping`
- analysis dir: `/home/devj/local-arch/code/glacier-mapping/analysis`
- GEE dir: `/home/devj/local-arch/code/glacier-mapping/google_earth_scripts`

---

## 14. Immediate next steps

1. User runs LILA baselines with updated configs:
   - `uv run python scripts/train.py --config configs/desktop/debris_ice/lila_dci_baseline_bs8_seed42.yaml --server desktop --gpu 0`
   - `uv run python scripts/train.py --config configs/desktop/clean_ice/lila_ci_baseline_bs8_seed42.yaml --server desktop --gpu 0`
   - `uv run python scripts/train.py --config configs/desktop/multiclass/lila_multiclass_baseline_bs8_seed42.yaml --server desktop --gpu 0`
2. Define benchmark-faithful track more precisely around released LILA packaging vs Bibek-style repair.
3. Design improved rebuild small-subset GEE experiments with more coherent compositing, likely top-k ordered scene selection rather than broad pixelwise mosaic.
4. Keep using both quantitative audits and visual contact sheets before any full export.

---

## 15. One-paragraph summary

Main provenance question now much clearer. Released LILA data is best public benchmark anchor for both labels and initial imagery baseline, while old JS is only partial historical evidence and does not exactly match released LILA scene provenance. LILA `clean.shp` + `debris.shp` match current HKH CIDC label universe, so they should anchor benchmark-faithful rasterization. Python EE rebuild path remains preferred, but current sweep mosaics still show visually important streaking despite decent audit metrics, so full 202-tile export should wait. Converted released LILA dataset now exists locally in repo-ready processed form, baseline training configs exist, and latest config fix switched them to stronger SMP Unet-style baselines aligned with recent successful runs.
