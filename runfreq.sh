#!/bin/bash

if [ $# -ne 1 ];
then
    echo "Usage ./runfreq <frequency in MHz>"
    exit
fi

mhz=$1

# Max and min values for skylake nodes on Kay
# Note - running at MAX_FREQ will not lock the frequency - Turbo will be enabled.
MAX_FREQ=2101
MIN_FREQ=1000
if [ $mhz -gt $MAX_FREQ ];
then
    mhz=$MAX_FREQ
fi
if [ $mhz -lt $MIN_FREQ ];
then
    mhz=$MIN_FREQ
fi
khz=$(($1*1000))

if [ $mhz -eq $MAX_FREQ ];
then
    echo "Running at $mhz MHz with TURBO ON"
else
    echo "Running with frequency locked at $mhz MHz"
fi

aprun -q --cpu-binding=0 --p-state=$khz numactl --membind=0 --cpunodebind=0 ./a.out $mhz | grep "cycle"
