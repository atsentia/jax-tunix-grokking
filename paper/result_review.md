# Distilled Grokking Result Review

## Overview
This note documents a technical review of the "Distilling Grokking" experiment materials in `paper/paper.md` and supporting assets. The focus is on verifying internal consistency, experimental support for the claims, and reproducibility considerations.

## Strengths
- The paper clearly states the teacher/student setup and training hyperparameters, making it straightforward to understand the experimental design (Sections 3.1–3.2).
- Mechanistic analyses (CKA, effective rank, attention entropy, weight norms) provide valuable qualitative insight into how the distilled students relate to the teacher.
- Open-source code and scripts are included, enabling interested readers to inspect the implementation details.

## Issues Found
1. **Single-run evidence only.** Section 4.2 explicitly notes that the reported student results come from “single representative runs,” with statistical validation deferred to future work (§4.4). This weakens claims of reliability until variance across seeds is measured.
2. **Contradictory statements about logit distillation.**
   - Section 4.2.1 reports that logit distillation succeeds (99.98% validation accuracy).
   - The ASCII training-curve description immediately afterward still says the “Logit curve remains flat near 0%, indicating no learning” and “No grokking transition in students,” which directly contradicts the summary above.
   - The README also has inconsistent outcomes, listing “Complete failure (~1% val acc)” for the logit run in the configuration summary while simultaneously flagging the run as a success in the key results table.
   These discrepancies call the reported success of the logit baseline into question.
3. **Placeholder quantitative analysis.** Section 4.4 (10× runs per strategy) is entirely a placeholder, so no statistical tests or confidence intervals currently back the conclusions.
4. **Plotting directory mismatch.** `paper/scripts/generate_paper_plots.py` expects run folders such as `runs/distill_0.5x_logit`, but the provided distillation scripts write to suffixed names like `runs/distill_0.5x_logit_wd1.0`. Without aligning these paths, the plotting script throws a `FileNotFoundError` even if you generated the runs locally.
5. **Missing raw training logs.** The repository references specific run directories (e.g., `runs/distill_0.5x_logit`) for plotting, but those histories are not committed. Without them, readers cannot verify the curves or reproduce the exact figures without rerunning experiments.

## Recommendations
- Resolve the contradictory descriptions of logit distillation and ensure the text, tables, and plots all reflect the same outcome.
- Complete the promised multi-seed evaluation, or at minimum temper the language around generalization guarantees until the data exists.
- Commit the exact training histories used for plots, or provide scripts/checkpoints that deterministically regenerate them.
- Align the plotting script’s expected run directory names with the outputs produced by the distillation scripts, and consider adding lightweight unit tests for the analysis utilities.

Addressing these items will make the reported distillation results far more convincing and easier to validate externally.
