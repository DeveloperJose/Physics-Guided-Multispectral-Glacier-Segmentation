#!/usr/bin/env python3
"""Gradio demo for glacier segmentation."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from glacier_mapping.data.data import load_band_names  # noqa: E402
from glacier_mapping.lightning.glacier_module import (  # noqa: E402
    GlacierSegmentationModule,
)
from glacier_mapping.model.evaluation import (  # noqa: E402
    create_invalid_mask,
    get_probabilities,
    merge_ci_debris,
)

SEGMENT_COLORS = np.array(
    [
        [0.08, 0.10, 0.14],
        [0.16, 0.67, 0.95],
        [0.96, 0.49, 0.17],
    ],
    dtype=np.float32,
)
VALID_CLASS_NAMES = {1: "Clean Ice", 2: "Debris-Covered Ice"}
VIEW_OPTIONS = [
    "Natural Color",
    "Infrared",
    "Snow and Ice Signal",
    "Surface Velocity",
]


@dataclass(frozen=True)
class AppConfig:
    processed_dir: Path
    ckpt_dir: Path
    host: str
    port: int
    share: bool
    device: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        env_processed_dir = os.environ.get("PROCESSED_DIR")
        processed_dir = (
            Path(env_processed_dir)
            if env_processed_dir
            else resolve_default_processed_dir()
        )
        return cls(
            processed_dir=processed_dir,
            ckpt_dir=Path(os.environ.get("CKPT_DIR", PROJECT_ROOT / "output")),
            host=os.environ.get("GRADIO_HOST", "0.0.0.0"),
            port=int(os.environ.get("GRADIO_PORT", "7860")),
            share=os.environ.get("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"},
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


def resolve_default_processed_dir() -> Path:
    output_dir = PROJECT_ROOT / "output"
    candidate_paths: list[Path] = []

    for run_name in (
        "ablation2_ci_full_physics_velocity_channels_loss_gpu0",
        "ablation2_dci_full_physics_velocity_channels_loss_gpu0",
    ):
        conf_path = output_dir / run_name / "conf.json"
        if not conf_path.exists():
            continue
        with conf_path.open("r", encoding="utf-8") as handle:
            conf = json.load(handle)
        candidate_paths.append(Path(conf["loader_opts"]["processed_dir"]).expanduser())

    candidate_paths.extend(
        [
            Path.home() / "local-arch/data/HKH/gen_robust_comprehensive",
            Path(__file__).resolve().parent / "demo_data",
            PROJECT_ROOT / "data",
        ]
    )

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return Path(__file__).resolve().parent / "demo_data"


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_name_substring: str


@dataclass
class PatchRecord:
    label: str
    image_path: Path
    mask_path: Path | None


RUN_SPECS = {
    "ci": RunSpec("Clean Ice", "ablation2_ci_full_physics_velocity_channels_loss"),
    "dci": RunSpec(
        "Debris-Covered Ice", "ablation2_dci_full_physics_velocity_channels_loss"
    ),
}

RANK_K = 3

CUSTOM_CSS = """
.app-shell {max-width: 1320px; margin: 0 auto;}
.hero {
  background: linear-gradient(135deg, #081824 0%, #11344b 55%, #eef5f8 100%);
  color: white;
  border-radius: 22px;
  padding: 28px 30px;
  margin-bottom: 18px;
}
.hero h1 {margin: 0 0 8px 0; font-size: 2.1rem;}
.hero p {margin: 0; max-width: 58rem; line-height: 1.5;}
.kicker {
  display: inline-block;
  margin-bottom: 10px;
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #b7d8ea;
}
.legend-row {
  display: flex;
  gap: 18px;
  align-items: center;
  flex-wrap: wrap;
  margin: 10px 0 6px 0;
  font-size: 0.95rem;
  color: #344955;
}
.legend-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.legend-swatch {
  width: 14px;
  height: 14px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.35);
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 8px;
}
.summary-card {
  background: #f7fafb;
  border: 1px solid #d6e2e8;
  border-radius: 16px;
  padding: 14px 16px;
}
.summary-card h3 {
  margin: 0 0 8px 0;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #577180;
}
.summary-card .big {
  font-size: 1.5rem;
  font-weight: 700;
  color: #12212c;
}
.summary-card .hero-metric {
  font-size: 1.9rem;
  font-weight: 800;
  color: #0f1f2a;
  margin: 8px 0 10px 0;
}
.summary-card .small {
  margin-top: 4px;
  color: #52636f;
  font-size: 0.92rem;
}
.note-box {
  background: #fffaf0;
  border: 1px solid #f1d9a8;
  border-radius: 14px;
  padding: 12px 14px;
  color: #59461c;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.metric-card {
  background: #fcfdfe;
  border: 1px solid #dfe8ee;
  border-radius: 16px;
  padding: 14px 16px;
}
.metric-card h4 {
  margin: 0 0 8px 0;
  color: #435a69;
  font-size: 0.88rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.metric-card .score {
  font-size: 1.4rem;
  font-weight: 700;
  color: #0f1f2a;
}
.metric-card .sub {
  color: #52636f;
  font-size: 0.85rem;
  margin-top: 4px;
  line-height: 1.4;
}
.caption {
  color: #546672;
  font-size: 0.95rem;
  line-height: 1.45;
}
.metric-section-label {
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #6e838f;
  margin: 18px 0 6px 0;
  border-bottom: 1px solid #d6e2e8;
  padding-bottom: 4px;
}
"""


class DemoBackend:
    def __init__(self, config: AppConfig):
        self.config = config
        self.band_names = load_band_names(config.processed_dir)
        self.band_index = {
            name: idx for idx, name in enumerate(self.band_names.tolist())
        }
        self.models = {
            "ci": self._load_model_for_run(RUN_SPECS["ci"]),
            "dci": self._load_model_for_run(RUN_SPECS["dci"]),
        }
        self.patch_records = self._discover_patch_records()
        self.patch_label_to_index = {
            record.label: idx for idx, record in enumerate(self.patch_records)
        }
        self.has_reference_labels = any(
            record.mask_path is not None for record in self.patch_records
        )

    def _discover_patch_records(self) -> list[PatchRecord]:
        demo_patches = sorted(self.config.processed_dir.glob("tiff_*.npy"))
        if demo_patches:
            return [
                PatchRecord(
                    label=f"Scene {idx + 1}",
                    image_path=path,
                    mask_path=None,
                )
                for idx, path in enumerate(demo_patches)
            ]

        test_dir = self.config.processed_dir / "test"
        test_patches = sorted(test_dir.glob("tiff_*.npy"))
        if not test_patches:
            raise FileNotFoundError(
                f"No test patches found in {self.config.processed_dir} or {test_dir}"
            )

        records: list[PatchRecord] = []
        for path in test_patches:
            mask_name = path.name.replace("tiff_", "mask_", 1)
            mask_path = path.with_name(mask_name)
            records.append(
                PatchRecord(
                    label="",
                    image_path=path,
                    mask_path=mask_path if mask_path.exists() else None,
                )
            )

        labeled = [r for r in records if r.mask_path is not None]
        if len(labeled) >= RANK_K * 3:
            return self._rank_patches(records, labeled)

        for idx, r in enumerate(records):
            r.label = f"Scene {idx + 1}"
        return records

    def _rank_patches(
        self, all_records: list[PatchRecord], labeled: list[PatchRecord]
    ) -> list[PatchRecord]:
        import torch

        with torch.no_grad():
            ious: list[tuple[int, float]] = []
            for idx, record in enumerate(labeled):
                patch = np.load(record.image_path)
                truth = np.load(record.mask_path)

                ci_probs = get_probabilities(self.models["ci"], patch)
                dci_probs = get_probabilities(self.models["dci"], patch)
                merged, _ = merge_ci_debris(ci_probs, dci_probs, 0.5, 0.5)

                invalid = create_invalid_mask(patch, truth)
                valid = ~invalid
                class_ious: list[float] = []
                for cls_id in (1, 2):
                    pred_mask = (merged == cls_id) & valid
                    truth_mask = (truth == cls_id) & valid
                    intersection = np.logical_and(pred_mask, truth_mask).sum()
                    union = np.logical_or(pred_mask, truth_mask).sum()
                    class_ious.append(float(intersection / union) if union else 1.0)
                ious.append((idx, np.mean(class_ious)))

        ious.sort(key=lambda x: x[1], reverse=True)
        k = RANK_K
        tiers = [
            (ious[:k], "Top"),
            (ious[len(ious) // 2 - k // 2 : len(ious) // 2 - k // 2 + k], "Mid"),
            (ious[-k:], "Worst"),
        ]
        ranked: list[PatchRecord] = []
        for group, tier_name in tiers:
            for rank_in_tier, (orig_idx, _) in enumerate(group, 1):
                record = labeled[orig_idx]
                record.label = f"{tier_name} {rank_in_tier}"
                ranked.append(record)

        unlabeled = [r for r in all_records if r.mask_path is None]
        for idx, r in enumerate(unlabeled, 1):
            r.label = f"Unlabeled {idx}"

        return ranked + unlabeled

    def _load_model_for_run(self, run_spec: RunSpec) -> GlacierSegmentationModule:
        ckpt_path = self._find_best_checkpoint(run_spec)
        print(f"Loading {run_spec.label} model from {ckpt_path}...")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        hyper_parameters = checkpoint["hyper_parameters"]
        loader_opts = hyper_parameters.setdefault("loader_opts", {})
        loader_opts["processed_dir"] = str(self.config.processed_dir)

        with NamedTemporaryFile(
            suffix=".ckpt", prefix="gradio_", dir=ckpt_path.parent, delete=False
        ) as handle:
            temp_ckpt = Path(handle.name)

        try:
            torch.save(checkpoint, temp_ckpt)
            module = GlacierSegmentationModule.load_from_checkpoint(temp_ckpt)
        finally:
            temp_ckpt.unlink(missing_ok=True)

        module.eval().to(self.config.device)
        return module

    def _find_best_checkpoint(self, run_spec: RunSpec) -> Path:
        matching_runs = sorted(
            run_dir
            for run_dir in self.config.ckpt_dir.iterdir()
            if run_dir.is_dir() and run_spec.run_name_substring in run_dir.name
        )
        if not matching_runs:
            raise FileNotFoundError(
                f"No run directory matching '{run_spec.run_name_substring}' in "
                f"{self.config.ckpt_dir}"
            )

        best_ckpt: Path | None = None
        best_loss = float("inf")

        for run_dir in matching_runs:
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            for ckpt in ckpt_dir.glob("*epoch=*_val_loss=*.ckpt"):
                loss = self._parse_val_loss(ckpt)
                if loss is not None and loss < best_loss:
                    best_loss = loss
                    best_ckpt = ckpt

        if best_ckpt is not None:
            return best_ckpt

        for run_dir in matching_runs:
            fallback = run_dir / "checkpoints" / "last.ckpt"
            if fallback.exists():
                return fallback

        raise FileNotFoundError(
            f"No checkpoints found for '{run_spec.run_name_substring}' in "
            f"{self.config.ckpt_dir}"
        )

    @staticmethod
    def _parse_val_loss(ckpt_path: Path) -> float | None:
        marker = "val_loss="
        if marker not in ckpt_path.name:
            return None
        try:
            return float(
                ckpt_path.name.split(marker, maxsplit=1)[1].removesuffix(".ckpt")
            )
        except ValueError:
            return None

    def scene_for_dropdown(
        self,
        selected_label: str,
        view_name: str,
        ci_threshold: float,
        dci_threshold: float,
    ):
        idx = self.patch_label_to_index[selected_label]
        return self.render_scene(idx, view_name, ci_threshold, dci_threshold)

    def next_scene(
        self,
        current_idx: int,
        view_name: str,
        ci_threshold: float,
        dci_threshold: float,
    ):
        return self.render_scene(
            min(int(current_idx) + 1, len(self.patch_records) - 1),
            view_name,
            ci_threshold,
            dci_threshold,
        )

    def previous_scene(
        self,
        current_idx: int,
        view_name: str,
        ci_threshold: float,
        dci_threshold: float,
    ):
        return self.render_scene(
            max(int(current_idx) - 1, 0),
            view_name,
            ci_threshold,
            dci_threshold,
        )

    def render_scene(
        self,
        idx: int,
        view_name: str,
        ci_threshold: float,
        dci_threshold: float,
    ):
        scene_idx = max(0, min(int(idx), len(self.patch_records) - 1))
        record = self.patch_records[scene_idx]
        patch = np.load(record.image_path)
        truth = np.load(record.mask_path) if record.mask_path is not None else None

        ci_probs = get_probabilities(self.models["ci"], patch)
        dci_probs = get_probabilities(self.models["dci"], patch)
        merged, _ = merge_ci_debris(ci_probs, dci_probs, ci_threshold, dci_threshold)

        base_view = self.make_view_image(patch, view_name)
        overlay = self.overlay_segmentation(base_view, merged)
        summary_html = self.format_summary_cards(merged, truth, patch)
        scene_heading = self.format_scene_heading(scene_idx, record)
        reference_html = self.format_reference_metrics(patch, merged, truth)
        reference_image = self.reference_or_placeholder(truth)
        ci_confidence = self.make_probability_heatmap(ci_probs[:, :, 1], "ci")
        dci_confidence = self.make_probability_heatmap(dci_probs[:, :, 1], "dci")

        return (
            scene_heading,
            base_view,
            overlay,
            summary_html,
            reference_image,
            reference_html,
            ci_confidence,
            dci_confidence,
            scene_idx,
            gr.update(value=record.label),
        )

    def make_view_image(self, patch: np.ndarray, view_name: str) -> np.ndarray:
        if view_name == "Natural Color":
            return self.make_rgb_from_channels(patch, ("B3", "B2", "B1"))
        if view_name == "Infrared":
            return self.make_rgb_from_channels(patch, ("B4", "B3", "B2"))
        if view_name == "Snow and Ice Signal":
            return self.make_scalar_heatmap(self.get_band(patch, "NDSI"), "ice")
        if view_name == "Surface Velocity":
            return self.make_scalar_heatmap(
                self.get_band(patch, "velocity"), "velocity"
            )
        raise ValueError(f"Unsupported view: {view_name}")

    def get_band(self, patch: np.ndarray, band_name: str) -> np.ndarray:
        return patch[:, :, self.band_index[band_name]]

    def make_rgb_from_channels(
        self, patch: np.ndarray, band_names: tuple[str, str, str]
    ) -> np.ndarray:
        rgb = np.stack([self.get_band(patch, name) for name in band_names], axis=-1)
        return self.normalize_rgb(rgb)

    @staticmethod
    def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.astype(np.float32).copy()
        for channel_idx in range(rgb.shape[2]):
            band = rgb[:, :, channel_idx]
            valid = np.isfinite(band) & (band != 0)
            if valid.any():
                lo, hi = np.percentile(band[valid], [1, 99])
                rgb[:, :, channel_idx] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
            else:
                rgb[:, :, channel_idx] = 0
        return rgb

    @staticmethod
    def normalize_scalar(
        band: np.ndarray, percentile_low: float = 2, percentile_high: float = 98
    ):
        band = np.nan_to_num(band.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid = np.isfinite(band) & (band != 0)
        if valid.any():
            lo, hi = np.percentile(band[valid], [percentile_low, percentile_high])
            return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
        return np.zeros_like(band, dtype=np.float32)

    def make_scalar_heatmap(self, band: np.ndarray, palette: str) -> np.ndarray:
        norm = self.normalize_scalar(band)
        if palette == "ice":
            return np.stack(
                [
                    0.10 + 0.45 * norm,
                    0.18 + 0.70 * norm,
                    0.25 + 0.75 * norm,
                ],
                axis=-1,
            )
        if palette == "velocity":
            return np.stack(
                [
                    0.18 + 0.80 * norm,
                    0.10 + 0.55 * (1.0 - np.abs(norm - 0.5) * 2.0),
                    0.22 + 0.78 * (1.0 - norm),
                ],
                axis=-1,
            )
        raise ValueError(f"Unsupported palette: {palette}")

    def make_probability_heatmap(self, probs: np.ndarray, kind: str) -> np.ndarray:
        norm = np.clip(probs.astype(np.float32), 0, 1)
        if kind == "ci":
            return np.stack(
                [0.08 + 0.10 * norm, 0.15 + 0.70 * norm, 0.20 + 0.75 * norm], axis=-1
            )
        return np.stack(
            [0.25 + 0.70 * norm, 0.12 + 0.42 * norm, 0.05 + 0.18 * norm], axis=-1
        )

    @staticmethod
    def overlay_segmentation(
        base: np.ndarray, segmentation: np.ndarray, alpha: float = 0.52
    ):
        colors = SEGMENT_COLORS[segmentation]
        glacier_mask = segmentation > 0
        output = base.copy()
        output[glacier_mask] = (1.0 - alpha) * base[glacier_mask] + alpha * colors[
            glacier_mask
        ]
        return np.clip(output, 0, 1)

    def reference_or_placeholder(self, truth: np.ndarray | None) -> np.ndarray:
        if truth is None:
            placeholder = np.zeros((512, 512, 3), dtype=np.float32)
            placeholder[:] = np.array([0.16, 0.18, 0.22], dtype=np.float32)
            return placeholder
        return self.colorize_label_mask(truth)

    @staticmethod
    def colorize_label_mask(mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.int16, copy=False)
        display = np.zeros((*mask.shape, 3), dtype=np.float32)
        valid = mask != 255
        display[valid] = SEGMENT_COLORS[mask[valid].astype(np.intp, copy=False)]
        display[~valid] = np.array([0.55, 0.55, 0.55], dtype=np.float32)
        return display

    def format_scene_heading(self, idx: int, record: PatchRecord) -> str:
        subtitle = (
            "Held-out labeled example from the test set."
            if record.mask_path is not None
            else "Curated demo example."
        )
        return f"## {record.label}\n{subtitle}"

    def format_summary_cards(
        self, prediction: np.ndarray, truth: np.ndarray | None, patch: np.ndarray
    ) -> str:
        if truth is None:
            clean_pred = int((prediction == 1).sum())
            debris_pred = int((prediction == 2).sum())
            return """
<div class="summary-grid">
  <div class="summary-card">
    <h3>Clean Ice Prediction</h3>
    <div class="big">{clean_pred:,} pixels</div>
  </div>
  <div class="summary-card">
    <h3>Debris-Ice Prediction</h3>
    <div class="big">{debris_pred:,} pixels</div>
  </div>
</div>
""".format(clean_pred=clean_pred, debris_pred=debris_pred)

        invalid = create_invalid_mask(patch, truth)
        valid = ~invalid

        def binary_metrics(cls_id: int) -> dict:
            pred_cls = (prediction == cls_id) & valid
            truth_cls = (truth == cls_id) & valid
            tp = int(np.logical_and(pred_cls, truth_cls).sum())
            fp = int(np.logical_and(pred_cls, ~truth_cls).sum())
            fn = int(np.logical_and(~pred_cls, truth_cls).sum())
            prec = tp / (tp + fp) if (tp + fp) else None
            rec = tp / (tp + fn) if (tp + fn) else None
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else None
            return {
                "precision": prec,
                "recall": rec,
                "iou": iou,
                "truth": int(truth_cls.sum()),
                "pred": int(pred_cls.sum()),
                "intersection": tp,
                "union": tp + fp + fn,
            }

        def fmt(val: float | None, suffix: str = "%") -> str:
            return f"{val * 100:.1f}{suffix}" if val is not None else "N/A"

        ci = binary_metrics(1)
        deb = binary_metrics(2)

        return """
<div class="metric-section-label">Clean Ice</div>
<div class="metric-grid">
  <div class="metric-card">
    <h4>IoU</h4>
    <div class="score">{ci_iou}</div>
    <div class="sub">Intersection &#247; Union</div>
  </div>
  <div class="metric-card">
    <h4>Precision</h4>
    <div class="score">{ci_prec}</div>
    <div class="sub">Correct predictions &#247; total predictions</div>
  </div>
  <div class="metric-card">
    <h4>Recall</h4>
    <div class="score">{ci_rec}</div>
    <div class="sub">Found pixels &#247; human-labeled pixels</div>
  </div>
</div>
<div style="color:#6e838f;font-size:0.85rem;margin:-6px 0 14px 0">
Labeled: {ci_truth:,} px &middot; Predicted: {ci_pred:,} px &middot; Match: {ci_inter:,} px &middot; Union: {ci_union:,} px
</div>
<div class="metric-section-label">Debris-Covered Ice</div>
<div class="metric-grid">
  <div class="metric-card">
    <h4>IoU</h4>
    <div class="score">{deb_iou}</div>
    <div class="sub">Intersection &#247; Union</div>
  </div>
  <div class="metric-card">
    <h4>Precision</h4>
    <div class="score">{deb_prec}</div>
    <div class="sub">Correct predictions &#247; total predictions</div>
  </div>
  <div class="metric-card">
    <h4>Recall</h4>
    <div class="score">{deb_rec}</div>
    <div class="sub">Found pixels &#247; human-labeled pixels</div>
  </div>
</div>
<div style="color:#6e838f;font-size:0.85rem;margin:-6px 0 0 0">
Labeled: {deb_truth:,} px &middot; Predicted: {deb_pred:,} px &middot; Match: {deb_inter:,} px &middot; Union: {deb_union:,} px
</div>
""".format(
            ci_iou=fmt(ci["iou"]),
            ci_prec=fmt(ci["precision"]),
            ci_rec=fmt(ci["recall"]),
            ci_truth=ci["truth"],
            ci_pred=ci["pred"],
            ci_inter=ci["intersection"],
            ci_union=ci["union"],
            deb_iou=fmt(deb["iou"]),
            deb_prec=fmt(deb["precision"]),
            deb_rec=fmt(deb["recall"]),
            deb_truth=deb["truth"],
            deb_pred=deb["pred"],
            deb_inter=deb["intersection"],
            deb_union=deb["union"],
        )

    def format_reference_metrics(
        self, patch: np.ndarray, prediction: np.ndarray, truth: np.ndarray | None
    ) -> str:
        if truth is None:
            return (
                "<div class='note-box'>"
                "Human-labeled reference masks are not available for this example."
                "</div>"
            )
        return """
<div class="caption">
Gray pixels in the human label are ignored during metric computation (they correspond
to areas where expert labelers could not confidently determine glacier boundaries).
</div>
"""


def build_demo(backend: DemoBackend) -> gr.Blocks:
    with gr.Blocks(title="Glacier Mapping Demo", css=CUSTOM_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
<div class="hero">
  <div class="kicker">Ph.D. Dissertation &middot; UTEP Computer Science &middot; December 2025</div>
  <h1>Glacier Segmentation with Physics-Guided Deep Learning</h1>
  <p>
    <strong>Physics-Guided Strategies for Enhancing Neural Networks Trained With Limited Data</strong><br>
    Jose Guadalupe Perez Zamora &middot;
    <a href="https://github.com/DeveloperJose/Python-Glacier-Mapping-by-Segmentation" target="_blank" style="color:#b7d8ea">GitHub</a>
  </p>
  <p style="margin-top:12px">
    This demo runs two Boundary-Aware U-Net models (Clean Ice + Debris-Covered Ice) on Landsat-7
    imagery from the Hindu Kush-Himalaya. The models use physics-informed data augmentation
    (terrain features from DEM) and a physics-informed velocity loss — achieving <strong>46.07%</strong>
    Debris-Covered Ice IoU (28.2% relative improvement over prior art).
    Pick an example, switch satellite views, and compare predictions against expert human labels.
  </p>
</div>
"""
            )

            with gr.Row():
                patch_choice = gr.Dropdown(
                    choices=[record.label for record in backend.patch_records],
                    value=backend.patch_records[0].label,
                    label="Example",
                )
                view_choice = gr.Radio(
                    choices=VIEW_OPTIONS,
                    value="Natural Color",
                    label="Satellite view",
                )

            with gr.Row():
                prev_button = gr.Button("Previous Example", variant="secondary")
                next_button = gr.Button("Next Example", variant="primary")

            scene_title = gr.Markdown()
            with gr.Row():
                base_image = gr.Image(label="Satellite Image", height=380)
                overlay_image = gr.Image(label="Model Prediction", height=380)
                reference_image = gr.Image(label="Human Label", height=380)

            gr.HTML(
                """
<div class="legend-row">
  <div class="legend-chip"><span class="legend-swatch" style="background:#29abf2"></span>Clean ice</div>
  <div class="legend-chip"><span class="legend-swatch" style="background:#f57d2c"></span>Debris-covered ice</div>
  <div class="legend-chip"><span class="legend-swatch" style="background:#141a24"></span>Non-glacier</div>
  <div class="legend-chip"><span class="legend-swatch" style="background:#8c8c8c"></span>Ignored label pixels</div>
</div>
"""
            )
            summary_html = gr.HTML()
            reference_html = gr.HTML()

            with gr.Accordion("What is each model contributing?", open=False):
                with gr.Row():
                    ci_confidence = gr.Image(
                        label="Clean Ice Confidence",
                        height=320,
                    )
                    dci_confidence = gr.Image(
                        label="Debris-Covered Ice Confidence",
                        height=320,
                    )

            with gr.Accordion("About this demo", open=False):
                gr.HTML(
                    """
<div style="margin-bottom:14px">
<h3 style="margin:0 0 6px 0">Ph.D. Dissertation</h3>
<p style="margin:0;color:#435a69;font-size:0.95rem;line-height:1.5">
<strong>Physics-Guided Strategies for Enhancing Neural Networks Trained With Limited Data</strong><br>
Jose Guadalupe Perez Zamora &mdash; Ph.D. in Computer Science, UTEP, December 2025<br>
<a href="https://github.com/DeveloperJose/Python-Glacier-Mapping-by-Segmentation" target="_blank">GitHub</a>
</p>
</div>
<div style="margin-bottom:14px">
<h4 style="margin:0 0 4px 0">Dataset</h4>
<p style="margin:0;color:#435a69;font-size:0.95rem;line-height:1.5">
Hindu Kush-Himalaya (HKH) glacier dataset &mdash; 792 Landsat-7 ETM+ patches (512&times;512, 30m resolution),
expert-delineated boundaries from ICIMOD. Three classes: background, clean ice, and debris-covered ice.
Channels include 7 spectral bands, DEM derivatives, velocity from ITS&nbsp;LIVE, and physics-derived terrain
features (flow accumulation, TPI, roughness, plan curvature).
</p>
</div>
<div style="margin-bottom:14px">
<h4 style="margin:0 0 4px 0">Model</h4>
<p style="margin:0;color:#435a69;font-size:0.95rem;line-height:1.5">
Boundary-Aware U-Net (Aryal et al. 2023) with uncertainty-weighted multi-task learning (Dice + boundary loss).
Two binary models (clean ice, debris-covered ice) trained separately with physics-informed data augmentation
and a physics-informed velocity loss. Debris-covered ice IoU: <strong>46.07%</strong> (28.2% relative improvement over prior art).
</p>
</div>
<div>
<h4 style="margin:0 0 4px 0">How to read the page</h4>
<p style="margin:0;color:#435a69;font-size:0.95rem;line-height:1.5">
The colored overlay on the satellite image shows the model prediction: blue is clean ice, orange is
debris-covered ice. When a human-labeled reference is available, the metrics panel shows IoU, precision,
and recall for each class. Examples are ranked by overall IoU (Top &rarr; Mid &rarr; Bottom) so you can see
where the model performs well and where it struggles. Use the threshold sliders to adjust model sensitivity.
</p>
</div>
"""
                )
                ci_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Clean Ice Sensitivity",
                )
                dci_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Debris-Ice Sensitivity",
                )

            session_index = gr.State(0)

            outputs = [
                scene_title,
                base_image,
                overlay_image,
                summary_html,
                reference_image,
                reference_html,
                ci_confidence,
                dci_confidence,
                session_index,
                patch_choice,
            ]

            common_inputs = [view_choice, ci_threshold, dci_threshold]

            demo.load(
                fn=lambda view_name, ci_thr, dci_thr: backend.render_scene(
                    0, view_name, ci_thr, dci_thr
                ),
                inputs=common_inputs,
                outputs=outputs,
            )

            patch_choice.change(
                fn=backend.scene_for_dropdown,
                inputs=[patch_choice, *common_inputs],
                outputs=outputs,
            )
            view_choice.change(
                fn=backend.render_scene,
                inputs=[session_index, view_choice, ci_threshold, dci_threshold],
                outputs=outputs,
            )
            ci_threshold.change(
                fn=backend.render_scene,
                inputs=[session_index, view_choice, ci_threshold, dci_threshold],
                outputs=outputs,
            )
            dci_threshold.change(
                fn=backend.render_scene,
                inputs=[session_index, view_choice, ci_threshold, dci_threshold],
                outputs=outputs,
            )

            prev_button.click(
                fn=backend.previous_scene,
                inputs=[session_index, *common_inputs],
                outputs=outputs,
            )
            next_button.click(
                fn=backend.next_scene,
                inputs=[session_index, *common_inputs],
                outputs=outputs,
            )

    return demo


def launch_demo(
    demo: gr.Blocks, host: str, preferred_port: int, share: bool, max_attempts: int = 10
) -> None:
    last_error: OSError | None = None
    for port in range(preferred_port, preferred_port + max_attempts):
        try:
            if port != preferred_port:
                print(f"Port {preferred_port} busy, trying {port}...")
            demo.launch(server_name=host, server_port=port, share=share)
            return
        except OSError as error:
            if "Cannot find empty port" not in str(error):
                raise
            last_error = error

    if last_error is not None:
        raise last_error


def main() -> None:
    config = AppConfig.from_env()
    backend = DemoBackend(config)

    print(f"Device: {config.device}")
    print(f"Examples available: {len(backend.patch_records)}")
    print(
        f"Reference labels available: {sum(record.mask_path is not None for record in backend.patch_records)}"
    )

    demo = build_demo(backend)
    demo.queue()
    launch_demo(demo, config.host, config.port, config.share)


if __name__ == "__main__":
    main()
