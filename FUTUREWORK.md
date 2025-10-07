# Future Work and Experiments

## Softmax Variants for Grokking

**Reference**: [Grokking with Muon paper](https://arxiv.org/pdf/2504.16041)

The paper tested three softmax variants:
1. **Regular softmax** (baseline)
2. **Stablemax** (slightly better performance)
3. **Third variant** (to be identified from paper)

### Hypothesis
Stablemax may provide more stable gradients during the grokking transition phase, potentially:
- Accelerating the grokking transition
- Improving final generalization
- Better numerical stability with high-precision distillation

### Proposed Experiments

1. **Baseline comparison**:
   - Train with regular softmax (current implementation)
   - Train with stablemax
   - Compare grokking speed and final accuracy

2. **Distillation impact**:
   - Does stablemax in teacher improve student distillation?
   - Does stablemax in student improve learning from teacher?
   - Combined: both teacher and student use stablemax

3. **Implementation**:
   ```python
   # In models.py, replace softmax with stablemax option
   def stablemax(logits, axis=-1):
       """Numerically stable softmax"""
       shifted = logits - jax.lax.stop_gradient(jnp.max(logits, axis=axis, keepdims=True))
       return jax.nn.softmax(shifted, axis=axis)
   ```

4. **Metrics to track**:
   - Epoch at which grokking starts
   - Transition speed (epochs from 50% to 95% val accuracy)
   - Final validation accuracy
   - Training stability (gradient norms, loss variance)

### Priority
**Medium** - Not critical for initial distillation experiments, but valuable for optimization.

---

## Other Future Directions

### 1. Learning Rate Schedules
- Current: Constant LR after warmup
- Test: Cosine decay, linear decay, cyclic schedules
- Hypothesis: Dynamic LR may accelerate grokking

### 2. Weight Decay Variations
- Current: Fixed wd=1.0 (critical for grokking)
- Test: Scheduled weight decay (start low, increase)
- Hypothesis: Gradual regularization may provide smoother transition

### 3. Recursive Distillation
- Already planned in PRD (docs/prd_grokking_distillation.md)
- Teacher → Student → Grandstudent chain
- Track: Does grokking behavior transfer through multiple generations?

### 4. Early-Stopped Teachers
- Train teachers for various epochs before grokking
- Compare: Pre-grokking vs mid-grokking vs post-grokking teachers
- Question: Can students learn to grok from non-grokked teachers?

### 5. Mixed Strategy Distillation
- Current: Single strategy per run (logit OR attention OR feature)
- Test: Combined strategies with learned weighting
- Hypothesis: Multi-objective learning may improve distillation quality

### 6. Cross-Optimizer Distillation
- AdamW teacher → Muon student
- Muon teacher → AdamW student
- Question: Do optimizer-specific patterns transfer?

### 7. Attention Mechanism Variations
- Current: Standard multi-head self-attention
- Test: Relative position encodings, ALiBi, other variants
- Impact on grokking and distillation?

### 8. Larger Problem Spaces
- Current: p=97 (prime modulus)
- Test: p=113, 127, 151 (larger primes)
- Test: Different operations (multiplication, addition)
- Hypothesis: Larger spaces may show clearer grokking dynamics

---

## Questions to Answer

1. **Does stablemax affect the memorization → generalization transition?**
2. **Can we predict which students will grok successfully from early distillation metrics?**
3. **What is the minimum teacher performance needed for successful distillation?**
4. **Do different distillation strategies preserve different aspects of grokking?**

---

## Notes
- This file tracks ideas from papers, discussions, and experimental observations
- Priority levels: High (next experiments), Medium (soon), Low (exploratory)
- Link relevant paper sections and experimental results as we go
