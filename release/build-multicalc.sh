#!/bin/bash
# -----------------------------------------------------------------------------Ï
# Filename:    build-multicalc.sh
# Revision:    None
# Date:        2018/09/21 - 10:40
# Author:      Haixiang HOU
# Email:       hexid26@outlook.com
# Website:     [NULL]
# Notes:       [NULL]
# -----------------------------------------------------------------------------
# Copyright:   2018 (c) Haixiang
# License:     GPL
# -----------------------------------------------------------------------------
# Version [1.0]
# build ../GPU/multicalc.cu

nvcc -std=c++11 -O3 ../GPU/multicalc.cu -o multicalc_cpp
