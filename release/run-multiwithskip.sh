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

if [ $# != 3 ]; then
    echo "接受 3 个参数 [N] [filepath] [skip_num]"
    echo "N 为 2 的次方，表示同时处理的发射源数量"
    exit
fi

./multicalc$3_cpp $1 b_64.filter $2 mul$1-skip$3.txt
./2image_cpp mul$1-skip$3.txt
cp mul$1-skip$3.txt /home/samba/anonymous_shares/Skip_Compare/mul$1-skip$3.txt
cp 160.png /home/samba/anonymous_shares/Skip_Compare/mul$1-skip$3.png
mv 160.png mul$1-skip$3.png
