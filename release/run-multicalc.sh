#!/usr/bin/zsh
# -----------------------------------------------------------------------------Ï
# Filename:    run-multicalc.sh
# Revision:    None
# Date:        2018/09/21 - 10:41
# Author:      Haixiang HOU
# Email:       hexid26@outlook.com
# Website:     [NULL]
# Notes:       [NULL]
# -----------------------------------------------------------------------------
# Copyright:   2018 (c) Haixiang
# License:     GPL
# -----------------------------------------------------------------------------
# Version [1.0]
# run multicalc_cpp

if [ $# != 2 ]; then
    echo "只接受 2 个参数 [N] [filepath] "
    echo "N 为 2 的次方，表示同时处理的发射源数量"
    exit
fi

./multicalc_cpp $1 b_64.filter $2 mul$1.txt
./2image_cpp mul$1.txt
mv 160.png mul$1.png
imgcat mul$1.png
