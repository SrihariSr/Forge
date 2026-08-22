#!/bin/bash
# Sweep block sizes and thread counts, and report the best combination.
# BLOCK is a compile-time constant, so each size needs its own build.

echo "tuning on a 768x768 multiply, median of 9 runs"
echo

for BLOCK in 32 48 64 96 128; do
    gcc -O2 -DTUNE_BLOCK=$BLOCK tune.c -o tune_$BLOCK -lpthread
done

for BLOCK in 32 48 64 96 128; do
    for THREADS in 1 4 8 10 12 14; do
        ./tune_$BLOCK $THREADS
    done
    echo
done

rm -f tune_32 tune_48 tune_64 tune_96 tune_128