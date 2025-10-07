#!/bin/bash
# Monitor distillation progress

OUTDIR="runs/distill_0.5x_logit_wd1.0"
MAX_EPOCHS=150

while true; do
    if [ -f "$OUTDIR/distillation_history.json" ]; then
        python3 -c "
import json
try:
    data = json.load(open('$OUTDIR/distillation_history.json'))
    epoch = data['epoch'][-1] if data['epoch'] else 0
    train_acc = data['train_acc'][-1] if data['train_acc'] else 0.0
    val_acc = data['val_acc'][-1] if data['val_acc'] else 0.0
    print(f'[Distillation] Epoch {epoch:3d}/$MAX_EPOCHS | Train: {train_acc:5.1%} | Val: {val_acc:5.1%}', end='\r')
except:
    print('[Distillation] Initializing...', end='\r')
" 2>/dev/null
    else
        echo "[Distillation] Waiting for training to start..."
    fi

    sleep 3

    # Check if complete
    if [ -f "$OUTDIR/distillation_history.json" ]; then
        EPOCH=$(python3 -c "import json; data = json.load(open('$OUTDIR/distillation_history.json')); print(data['epoch'][-1] if data['epoch'] else 0)" 2>/dev/null)
        if [ "$EPOCH" = "$MAX_EPOCHS" ]; then
            echo ""
            echo "[Distillation] Complete!"
            break
        fi
    fi
done
