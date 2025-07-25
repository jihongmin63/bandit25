#!/bin/bash
mkdir -p logs

for i in {1..8}; do
    python "recovery$i.py" > "logs/log$i.txt" 2>&1 &
    sleep 0.2
done

wait
