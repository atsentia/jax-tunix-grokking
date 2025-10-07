# Research Paper: Distilling Grokking

## Quick Links

📄 **Complete Paper:** [`paper/paper.md`](paper/paper.md)

📊 **Figures & Results:** [`paper/figures/`](paper/figures/)

🔧 **Analysis Scripts:** [`paper/scripts/`](paper/scripts/)

📖 **Paper Guide:** [`paper/README.md`](paper/README.md)

---

## Paper Summary

**Title:** Distilling Grokking: Knowledge Transfer from Grokked Teacher Models

**Author:** Amund Tveit (Atsentia)

**Repository:** [github.com/atsentia/jax-tunix-grokking](https://github.com/atsentia/jax-tunix-grokking)

### Key Findings

1. **All three distillation strategies succeed** with weight decay = 1.0
   - Logit: 99.98% val acc (converges epoch 123)
   - Attention: 99.96% val acc (converges epoch 121)
   - Feature: 100.00% val acc (converges epoch 91) ⚡ **1.4× faster**

2. **Weight decay is critical** for transferring grokking from teacher to student

3. **Mechanistic insights:**
   - High CKA alignment in early layers (0.89-0.92)
   - Students learn more compact representations (19.7-30.7 vs 38.7-41.2 dims)
   - Feature distillation produces lowest-rank solutions

4. **Delayed generalization persists** even with intermediate supervision

### Paper Structure

```
paper/
├── paper.md                    # Complete manuscript (51KB)
├── README.md                   # Detailed paper guide
├── figures/                    # All visualizations
│   ├── paper_teacher_grokking.{png,pdf}
│   ├── paper_distillation_comparison.{png,pdf}
│   ├── paper_mechanistic_analysis.{png,pdf}
│   ├── paper_10x_runs_placeholder.{png,pdf}
│   └── paper_results_table.tex
├── scripts/                    # Reproducibility
│   ├── generate_paper_plots.py
│   ├── mechanistic_analysis.py
│   ├── plot_mechanistic_analysis.py
│   └── create_placeholder_figure.py
└── data/
    └── mechanistic_analysis_results.json
```

### Sections

1. **Abstract** - High-level summary with key findings
2. **Introduction** - Motivation and contributions
3. **Background** - Grokking phenomenon and knowledge distillation
4. **Methods** - Three distillation strategies with detailed explanations
5. **Experiments** - Results from snapshot runs + mechanistic analysis
6. **Discussion** - Why weight decay matters, why feature distillation is faster
7. **Related Work** - Prior work on grokking and distillation
8. **Future Work** - 10× validation study, 4-bit training, recursive distillation
9. **Conclusion** - Summary and implications
10. **Appendices** - Hyperparameters and reproducibility details

### Status

- ✅ Complete paper draft (40+ pages)
- ✅ All figures generated
- ✅ Mechanistic interpretability analysis complete
- 🔄 Large-scale validation (10× runs) - placeholder added (§4.4)
- 🔄 W&B integration - planned

---

**Quick Start:** Read [`paper/paper.md`](paper/paper.md) for the full manuscript.
