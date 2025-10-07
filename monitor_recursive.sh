#!/bin/bash
# Monitor all recursive distillation experiments in parallel

MAX_EPOCHS=300

echo "Monitoring recursive distillation experiments..."
echo "="*70

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════════"
    echo "     Recursive Distillation (0.25x from 0.5x teachers)"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""

    # Logit → Logit recursive
    if [ -f "runs/recursive_0.25x_from_logit/distillation_history.json" ]; then
        python3 -c "
import json
try:
    data = json.load(open('runs/recursive_0.25x_from_logit/distillation_history.json'))
    epoch = data['epoch'][-1]
    train_acc = data['train_acc'][-1]
    val_acc = data['val_acc'][-1]
    status = '✅ COMPLETE' if epoch >= $MAX_EPOCHS else 'Running'
    print(f'[LOGIT→LOGIT]      Epoch {epoch:3d}/$MAX_EPOCHS | Train: {train_acc:5.1%} | Val: {val_acc:5.1%} | {status}')
except:
    print('[LOGIT→LOGIT]      Initializing...')
" 2>/dev/null
    else
        echo "[LOGIT→LOGIT]      No data yet"
    fi

    # Attention → Attention recursive
    if [ -f "runs/recursive_0.25x_from_attention/distillation_history.json" ]; then
        python3 -c "
import json
try:
    data = json.load(open('runs/recursive_0.25x_from_attention/distillation_history.json'))
    epoch = data['epoch'][-1]
    train_acc = data['train_acc'][-1]
    val_acc = data['val_acc'][-1]
    status = '✅ COMPLETE' if epoch >= $MAX_EPOCHS else 'Running'
    print(f'[ATTN→ATTN]        Epoch {epoch:3d}/$MAX_EPOCHS | Train: {train_acc:5.1%} | Val: {val_acc:5.1%} | {status}')
except:
    print('[ATTN→ATTN]        Initializing...')
" 2>/dev/null
    else
        echo "[ATTN→ATTN]        No data yet"
    fi

    # Feature → Feature recursive
    if [ -f "runs/recursive_0.25x_from_feature/distillation_history.json" ]; then
        python3 -c "
import json
try:
    data = json.load(open('runs/recursive_0.25x_from_feature/distillation_history.json'))
    epoch = data['epoch'][-1]
    train_acc = data['train_acc'][-1]
    val_acc = data['val_acc'][-1]
    status = '✅ COMPLETE' if epoch >= $MAX_EPOCHS else 'Running'
    print(f'[FEAT→FEAT]        Epoch {epoch:3d}/$MAX_EPOCHS | Train: {train_acc:5.1%} | Val: {val_acc:5.1%} | {status}')
except:
    print('[FEAT→FEAT]        Initializing...')
" 2>/dev/null
    else
        echo "[FEAT→FEAT]        No data yet"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "Model sizes: 128d (550k) → 64d (138k) → 32d (~35k params)"
    echo "════════════════════════════════════════════════════════════════════"

    # Check if all are complete
    LOGIT_DONE=0
    ATTN_DONE=0
    FEAT_DONE=0

    if [ -f "runs/recursive_0.25x_from_logit/distillation_history.json" ]; then
        LOGIT_EPOCH=$(python3 -c "import json; data = json.load(open('runs/recursive_0.25x_from_logit/distillation_history.json')); print(data['epoch'][-1] if data['epoch'] else 0)" 2>/dev/null)
        [ "$LOGIT_EPOCH" = "$MAX_EPOCHS" ] && LOGIT_DONE=1
    fi

    if [ -f "runs/recursive_0.25x_from_attention/distillation_history.json" ]; then
        ATTN_EPOCH=$(python3 -c "import json; data = json.load(open('runs/recursive_0.25x_from_attention/distillation_history.json')); print(data['epoch'][-1] if data['epoch'] else 0)" 2>/dev/null)
        [ "$ATTN_EPOCH" = "$MAX_EPOCHS" ] && ATTN_DONE=1
    fi

    if [ -f "runs/recursive_0.25x_from_feature/distillation_history.json" ]; then
        FEAT_EPOCH=$(python3 -c "import json; data = json.load(open('runs/recursive_0.25x_from_feature/distillation_history.json')); print(data['epoch'][-1] if data['epoch'] else 0)" 2>/dev/null)
        [ "$FEAT_EPOCH" = "$MAX_EPOCHS" ] && FEAT_DONE=1
    fi

    if [ $LOGIT_DONE -eq 1 ] && [ $ATTN_DONE -eq 1 ] && [ $FEAT_DONE -eq 1 ]; then
        echo "✅ All recursive experiments complete!"
        break
    fi

    sleep 3
done
