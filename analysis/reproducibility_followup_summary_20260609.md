# Reproducibility follow-up results - 2026-06-09

Source: MLflow experiment `reproducibility`, follow-up DCI runs. Callback/test metrics are triage only; final paper-style checks use `scripts/predict.py`.

## Paper-path checks
| model | predict_iou | predict_precision | predict_recall | callback_iou | delta_iou |
| --- | --- | --- | --- | --- | --- |
| baseline_landsat_dem_seed3407 | 0.3941 | 0.5697 | 0.5612 | 0.3769 | 0.0173 |
| full_threshold_10p0_seed42 | 0.3912 | 0.6885 | 0.4753 | 0.3892 | 0.0020 |
| full_threshold_5p0_seed42 | 0.3766 | 0.7721 | 0.4237 | 0.3703 | 0.0063 |
| full_threshold_2p0_seed2026 | 0.3753 | 0.6805 | 0.4555 | 0.3688 | 0.0065 |
| full_threshold_2p0_seed42 | 0.3724 | 0.7175 | 0.4364 | 0.3702 | 0.0022 |

## Best callback runs
| group | threshold | seed | test_iou | test_precision | test_recall | best_val_loss | duration_min | run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_threshold | 10.0000 | 42 | 0.3892 | 0.6890 | 0.4721 | -2.0345 | 31.8610 | reproducibility_dci_followup_full_threshold_10p0_seed42_desktop_20260609_021038 |
| baseline_landsat_dem |  | 3407 | 0.3769 | 0.5686 | 0.5278 | 0.2720 | 14.2797 | reproducibility_dci_followup_baseline_landsat_dem_seed3407_desktop_20260608_233957 |
| full_threshold | 5.0000 | 42 | 0.3703 | 0.7738 | 0.4153 | -1.5753 | 36.4194 | reproducibility_dci_followup_full_threshold_5p0_seed42_desktop_20260609_143647 |
| full_threshold | 2.0000 | 42 | 0.3702 | 0.7177 | 0.4333 | -1.2205 | 36.2795 | reproducibility_dci_followup_full_threshold_2p0_seed42_desktop_20260609_082513 |
| full_threshold | 2.0000 | 2026 | 0.3688 | 0.6835 | 0.4447 | -1.3365 | 31.7998 | reproducibility_dci_followup_full_threshold_2p0_seed2026_desktop_20260609_072342 |
| full_threshold | 1.5000 | 42 | 0.3595 | 0.7350 | 0.4130 | -1.1226 | 30.7162 | reproducibility_dci_followup_full_threshold_1p5_seed42_desktop_20260609_062246 |
| full_threshold | 1.0000 | 42 | 0.3515 | 0.7006 | 0.4137 | -1.1187 | 31.5909 | reproducibility_dci_followup_full_threshold_1p0_seed42_desktop_20260609_041737 |
| full_threshold | 10.0000 | 2026 | 0.3496 | 0.7374 | 0.3993 | -2.0776 | 32.0284 | reproducibility_dci_followup_full_threshold_10p0_seed2026_desktop_20260609_010354 |
| full_threshold | 1.5000 | 2026 | 0.3475 | 0.6120 | 0.4457 | -1.1717 | 26.4751 | reproducibility_dci_followup_full_threshold_1p5_seed2026_desktop_20260609_052454 |
| full_threshold | 2.5000 | 42 | 0.3418 | 0.7415 | 0.3880 | -1.2422 | 27.5685 | reproducibility_dci_followup_full_threshold_2p5_seed42_desktop_20260609_103508 |
| full_threshold | 3.1600 | 2026 | 0.3289 | 0.7743 | 0.3638 | -1.3372 | 30.5841 | reproducibility_dci_followup_full_threshold_3p16_seed2026_desktop_20260609_113144 |
| full_threshold | 2.0000 | 3407 | 0.3281 | 0.5059 | 0.4828 | -1.1697 | 28.5067 | reproducibility_dci_followup_full_threshold_2p0_seed3407_desktop_20260609_075606 |

## Threshold callback means
| threshold | runs | mean_iou | std_iou | min_iou | max_iou | mean_precision | mean_recall | mean_duration_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10.0000 | 4.0 | 0.2860 | 0.1173 | 0.1231 | 0.3892 | 0.7758 | 0.3284 | 32.4117 |
| 1.0000 | 4.0 | 0.2705 | 0.0742 | 0.1776 | 0.3515 | 0.7820 | 0.3011 | 31.0691 |
| 2.0000 | 4.0 | 0.2686 | 0.1752 | 0.0075 | 0.3702 | 0.7232 | 0.3421 | 31.4022 |
| 5.0000 | 4.0 | 0.2589 | 0.1157 | 0.1073 | 0.3703 | 0.6658 | 0.3154 | 31.9707 |
| 2.5000 | 4.0 | 0.2551 | 0.1295 | 0.0623 | 0.3418 | 0.7721 | 0.2915 | 29.7004 |
| 1.5000 | 4.0 | 0.2310 | 0.1434 | 0.0802 | 0.3595 | 0.7819 | 0.2699 | 30.4654 |
| 3.1600 | 4.0 | 0.2256 | 0.1391 | 0.0218 | 0.3289 | 0.8300 | 0.2454 | 29.4460 |

## Controls
| group | seed | test_iou | test_precision | test_recall | best_val_loss | duration_min | run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_landsat_dem | 3407 | 0.3769 | 0.5686 | 0.5278 | 0.2720 | 14.2797 | reproducibility_dci_followup_baseline_landsat_dem_seed3407_desktop_20260608_233957 |
| full_no_velocity_loss | 3407 | 0.2973 | 0.6545 | 0.3526 | 0.0999 | 35.5807 | reproducibility_dci_followup_full_no_velocity_loss_seed3407_desktop_20260608_235451 |
| physics_velocity_no_spectral_threshold_2p0 | 3407 | 0.3152 | 0.7221 | 0.3588 | -1.1562 | 23.2877 | reproducibility_dci_followup_physics_velocity_no_spectral_threshold_2p0_seed3407_desktop_20260609_151349 |