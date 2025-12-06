# Gen5 Experiment Findings

This generation focused on stabilizing the training pipeline and achieved a breakthrough in multi-class debris detection without relying on velocity channels.

- **Best Model:** The multi-class baseline ('multi_base_gen5') proved to be the best overall model, with a Clean Ice IoU of ~0.69 and a Debris IoU of ~0.41.
- **Key Insight:** A robust multi-class training strategy can compensate for the lack of complex input features like velocity, achieving comparable performance.
- **Metric System:** Gen5 introduced a reliable and correctly labeled metric system, making its results the ground truth for future comparisons.
