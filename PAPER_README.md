# Distilled Grokking Paper - Summary

## Paper Location
**Main paper:** `paper_distilled_grokking.md`

## Generated Figures

1. **paper_teacher_grokking.png/pdf** - Teacher model grokking dynamics showing the phase transition around epoch 70
2. **paper_distillation_comparison.png/pdf** - Side-by-side comparison of all three distillation strategies (validation accuracy and training loss)
3. **paper_mechanistic_analysis.png/pdf** - Mechanistic interpretability analysis (CKA, effective rank, attention entropy)
4. **paper_10x_runs_placeholder.png/pdf** - Placeholder for large-scale validation study (10 runs per strategy)
5. **paper_results_table.tex** - LaTeX table with final results

## Key Results (Snapshot Runs)

### Teacher Model (128-dim, 550k params)
- **Final Training Accuracy:** 99.85%
- **Final Validation Accuracy:** 100.00%
- **Grokking Transition:** ~Epoch 70
- **Total Training:** 150 epochs

### Student Models (64-dim, 140k params = 25% of teacher)

| Strategy | Epochs to 99% | Final Val Acc | Success? |
|----------|---------------|---------------|----------|
| **Logit Distillation (wd=1.0)** | 123 | 99.98% | ✅ Success |
| **Attention Transfer** | 121 | 99.96% | ✅ Success |
| **Feature Projection** | 91 | 100.00% | ✅ Success (1.4× faster) |

**Key Findings:**
1. **Weight decay (1.0) is critical** - All strategies succeed with proper weight decay
2. **Feature distillation is fastest** - Converges 1.4× faster than logit/attention
3. **All students exhibit delayed generalization** - Stay near-random until epoch ~75

### Mechanistic Interpretability Analysis

**CKA Scores (Teacher-Student Similarity):**
- Layer 0: High alignment (0.89-0.92) across all strategies
- Layer 1: Lower alignment (0.59-0.68), with attention transfer showing highest (0.677)

**Effective Rank (Representation Dimensionality):**
- Teacher: 38.7 / 41.2 (Layer 0/1)
- Students: 19.7-30.7 dimensions (more compact than teacher)
- Feature distillation learns most compact representations

**Attention Entropy:**
- Layer 0: Diffuse patterns (~0.77 bits) across all models
- Layer 1: More focused, with feature distillation showing highest entropy (0.611)

**Weight Statistics:**
- All students have similar L1/L2 ratios (~245), suggesting comparable sparsity
- Teacher has higher ratio (511) due to larger capacity

### Large-Scale Validation (§4.4)

**Status:** Placeholder section added for 10× independent runs per strategy

**When completed, will provide:**
- Mean ± std validation accuracy
- Convergence epoch statistics
- Statistical significance tests (t-tests, Cohen's d)
- Training curves with 95% confidence intervals
- Wall-clock time and memory usage comparisons

## Reproducing Results

```bash
# Generate plots
python generate_paper_plots.py

# This will create:
# - paper_teacher_grokking.png/pdf
# - paper_distillation_comparison.png/pdf
# - paper_results_table.tex
```

## Experiment Details

**Task:** Modular division (p=97)
**Teacher:** 128-dim, 2 layers, ~550k parameters
**Students:** 64-dim, 2 layers, ~140k parameters (0.5× width, 0.25× total params)
**Student Training:** 50 epochs (vs. teacher's 150 epochs)

### Distillation Configurations

1. **Logit-Based:**
   - Temperature: 2.0
   - Alpha (hard/soft balance): 0.5
   - Result: Complete failure (~1% val acc)

2. **Attention Transfer:**
   - MSE loss on attention maps
   - Layer-wise matching (2 layers)
   - Result: 99.96% val acc

3. **Feature Projection:**
   - MSE loss on hidden states via learned projections
   - Maps 128-dim teacher features → 64-dim student features
   - Result: 100.00% val acc

## Future Work

1. **Statistical Validation:** Run 10 independent trials per strategy with error bars
2. **Weights & Biases Integration:** Log experiments to W&B for public sharing
3. **Recursive Distillation:** Test 0.25× students (32-dim) distilled from 0.5× students
4. **4-bit Quantization:** Explore INT4 training for 100× compression
5. **Mechanistic Analysis:** Use interpretability tools to compare teacher/student circuits

## Citations

**Grokking:**
```bibtex
@article{power2022grokking,
  title={Grokking: Generalization beyond overfitting on small algorithmic datasets},
  author={Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal={arXiv preprint arXiv:2201.02177},
  year={2022}
}
```

**This Work:**
```bibtex
@article{tveit2025distilledgrokking,
  title={Distilling Grokking: Knowledge Transfer from Grokked Teacher Models},
  author={Tveit, Amund},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  url={https://github.com/atsentia/jax-tunix-grokking}
}
```

## Contact

For questions or collaboration:
- **Author:** Amund Tveit
- **Repository:** [github.com/atsentia/jax-tunix-grokking](https://github.com/atsentia/jax-tunix-grokking)
- **Email:** [Your Email]

## Files Generated

```
.
├── paper_distilled_grokking.md           # Main paper (markdown format)
├── PAPER_README.md                        # This file
├── generate_paper_plots.py                # Script to generate figures
├── paper_teacher_grokking.png             # Teacher grokking dynamics plot
├── paper_teacher_grokking.pdf             # PDF version
├── paper_distillation_comparison.png      # Main results comparison
├── paper_distillation_comparison.pdf      # PDF version
└── paper_results_table.tex                # LaTeX results table
```

## Next Steps

1. **Add your affiliation** in `paper_distilled_grokking.md` (line 4)
2. **Add your email** in `paper_distilled_grokking.md` (line 5)
3. **Run 10 independent trials** for each distillation strategy to compute statistics
4. **Set up Weights & Biases** and link experiments in the paper
5. **Convert to LaTeX** if submitting to a conference/journal (e.g., ICLR, NeurIPS)
6. **Upload to arXiv** (get arXiv ID and update citations)
7. **Consider running on GPU** (A100) for faster iteration on larger experiments

## Notes on Paper Structure

The paper follows a standard ML research format inspired by:
- arXiv:2504.16041 (referenced inspiration paper)
- AOSE survey paper (aose.pdf) for structure and writing style

**Sections:**
1. **Abstract** - High-level summary (150-200 words)
2. **Introduction** - Motivation and contributions
3. **Background** - Grokking, distillation, and related work
4. **Methods** - Experimental setup and three distillation strategies
5. **Experiments** - Results and analysis
6. **Discussion** - Why logit fails, why attention/feature succeed
7. **Related Work** - Prior work on grokking and distillation
8. **Future Work** - Extensions and open questions
9. **Conclusion** - Summary and implications
10. **Appendices** - Hyperparameters and reproducibility details

**Key Features:**
- Mermaid diagrams for distillation architectures
- Crisp explanations of each strategy
- References to code on GitHub
- Placeholder sections for W&B links and 10× experiment runs
- Focus on "distilling grokking" as the novel angle
