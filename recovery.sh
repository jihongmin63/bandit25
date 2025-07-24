#!/bin/bash


for i in {1..8}
do
    python "src/recovery$i.py" &
    sleep 5
done

wait