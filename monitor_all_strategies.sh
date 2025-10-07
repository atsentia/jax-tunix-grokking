#!/bin/bash
# Monitor all distillation strategies in parallel

MAX_EPOCHS=150

echo "Monitoring distillation experiments..."
echo "="*70

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════════"
    echo "       Distillation Strategy Comparison (Epoch progress)"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""

    # Logit strategy (already complete)
    if [ -f "runs/distill_0.5x_logit_wd1.0/distillation_history.json" ]; then
        python3 -c "
import json
try:
    data = json.load(open('runs/distill_0.5x_logit_wd1.0/distillation_history.json'))
    epoch = data['epoch'][-1]
    train_acc = data['train_acc'][-1]
    val_acc = data['val_acc'][-1]
    status = '✅ COMPLETE' if epoch >= $MAX_EPOCHS else 'Running'
    print(f'[LOGIT]      Epoch {epoch:3d}/$MAX_EPOCHS | Train: {train_acc:5.1%} | Val: {val_acc:5.1%} | {status}')
except:
    print('[LOGIT]      Waiting...')
" 2>/dev/null
    else
        echo "[LOGIT]      No data yet"
    fi

    # Attention strategy
    if [ -f "runs/distill_0.5x_attention/distillation_history.json" ]; then
        python3 -c "
import json
try:
    data = json.load(open('runs/distill_0.5x_attention/distillation_history.json'))
    epoch = data['epoch'][-1]
    train_acc = data['train_acc'][-1]
    val_acc = data['val_acc'][-1]
    status = '✅ COMPLETE' if epoch >= $MAX_EPOCHS else 'Running'
    print(f'[ATTENTION]  Epoch {epoch:3d}/$MAX_EPOCHS | Train: {train_acc:5.1%} | Val: {val_acc:5.1%} | {status}')
except:
    print('[ATTENTION]  Initializing...')
" 2>/dev/null
    else
        echo "[ATTENTION]  No data yet"
    fi

    # Feature strategy
    if [ -f "runs/distill_0.5x_feature/distillation_history.json" ]; then
        python3 -c "
import json
try:
    data = json.load(open('runs/distill_0.5x_feature/distillation_history.json'))
    epoch = data['epoch'][-1]
    train_acc = data['train_acc'][-1]
    val_acc = data['val_acc'][-1]
    status = '✅ COMPLETE' if epoch >= $MAX_EPOCHS else 'Running'
    print(f'[FEATURE]    Epoch {epoch:3d}/$MAX_EPOCHS | Train: {train_acc:5.1%} | Val: {val_acc:5.1%} | {status}')
except:
    print('[FEATURE]    Initializing...')
" 2>/dev/null
    else
        echo "[FEATURE]    No data yet"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════════"

    # Check if both are complete
    ATT_DONE=0
    FEAT_DONE=0

    if [ -f "runs/distill_0.5x_attention/distillation_history.json" ]; then
        ATT_EPOCH=$(python3 -c "import json; data = json.load(open('runs/distill_0.5x_attention/distillation_history.json')); print(data['epoch'][-1] if data['epoch'] else 0)" 2>/dev/null)
        [ "$ATT_EPOCH" = "$MAX_EPOCHS" ] && ATT_DONE=1
    fi

    if [ -f "runs/distill_0.5x_feature/distillation_history.json" ]; then
        FEAT_EPOCH=$(python3 -c "import json; data = json.load(open('runs/distill_0.5x_feature/distillation_history.json')); print(data['epoch'][-1] if data['epoch'] else 0)" 2>/dev/null)
        [ "$FEAT_EPOCH" = "$MAX_EPOCHS" ] && FEAT_DONE=1
    fi

    if [ $ATT_DONE -eq 1 ] && [ $FEAT_DONE -eq 1 ]; then
        echo "✅ All experiments complete!"
        break
    fi

    sleep 3
done
