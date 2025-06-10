#!/usr/bin/env bash

# Get path to this script’s parent directory  
BASE_DIR=$(dirname "$(realpath "$0")")  

SESSION="PrivTrace"  

SCRIPT="${BASE_DIR}/evaluate.py"
PYTHON="$HOME/miniconda3/envs/ptg/bin/python3"

N_FOLDS=5  

EPS_WITHOUT_ARGS=(1.0 2.0 5.0)  
EPS_WITH_ARGS=(1.0 2.0 5.0 10.0)  

ARGS="--custom-c"  

cd "$BASE_DIR" || exit 1  

# Create tmux session if it does not exist  
if ! tmux has-session -t "$SESSION" 2>/dev/null; then  
    tmux new-session -d -s "$SESSION"  
fi  

# Tabs without extra args  
for eps in "${EPS_WITHOUT_ARGS[@]}"; do  
    win="noargs_${eps}"  
    tmux new-window -t "$SESSION" -n "$win"  
    tmux select-window -t "$SESSION:$win"  

    # pane for fold 0  
    tmux send-keys "conda activate ptg" C-m  
    tmux send-keys "${PYTHON} $SCRIPT -m --dataset PORTO   --fold 0 --epsilon $eps; ${PYTHON} $SCRIPT -m --dataset GEOLIFE --fold 0 --epsilon $eps" C-m

    # panes for folds 1..4  
    for fold in $(seq 1 $((N_FOLDS - 1))); do  
        tmux split-window -t "$SESSION:$win"  
        tmux select-window -t "$SESSION:$win"  
        tmux select-layout -t "$SESSION:$win" tiled  

        tmux send-keys "conda activate ptg" C-m  
        tmux send-keys "${PYTHON} $SCRIPT -m --dataset PORTO   --fold $fold --epsilon $eps; ${PYTHON} $SCRIPT -m --dataset GEOLIFE --fold $fold --epsilon $eps" C-m
    done  
done  

# Tabs with extra args  
for eps in "${EPS_WITH_ARGS[@]}"; do  
    win="args_${eps}"  
    tmux new-window -t "$SESSION" -n "$win"  
    tmux select-window -t "$SESSION:$win"  

    # pane for fold 0  
    tmux send-keys "conda activate ptg" C-m  
    tmux send-keys "${PYTHON} $SCRIPT -m --dataset PORTO   --fold 0 --epsilon $eps $ARGS; ${PYTHON} $SCRIPT -m --dataset GEOLIFE --fold 0 --epsilon $eps $ARGS" C-m

    # panes for folds 1..4  
    for fold in $(seq 1 $((N_FOLDS - 1))); do  
        tmux split-window -t "$SESSION:$win"  
        tmux select-window -t "$SESSION:$win"  
        tmux select-layout -t "$SESSION:$win" tiled  

        tmux send-keys "conda activate ptg" C-m  
        tmux send-keys "${PYTHON} $SCRIPT -m --dataset PORTO   --fold $fold --epsilon $eps $ARGS; ${PYTHON} $SCRIPT -m --dataset GEOLIFE --fold $fold --epsilon $eps $ARGS" C-m
    done  
done
