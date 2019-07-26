#!/bin/bash

cat /proc/cpuinfo | grep "processor.* 0" -A 7 | tail -n 1 | cut -f3 -d' '
sleep 2
cat /proc/cpuinfo | grep "processor.* 0" -A 7 | tail -n 1 | cut -f3 -d' '
