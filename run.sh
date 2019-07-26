#!/bin/bash
echo "---"
hostname
echo "---"
cat /proc/cpuinfo | grep "processor.* 0" -A 7 | tail -n 1 | cut -f3 -d' '
echo "---"
./a.out
echo "---"
cat /proc/cpuinfo | grep "processor.* 0" -A 7 | tail -n 1 | cut -f3 -d' '
