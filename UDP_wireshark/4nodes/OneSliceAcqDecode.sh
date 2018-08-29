#!/bin/bash
pause() {
#   echo -n "Press SPACE to continue or Ctrl+C to exit ... "
  while true; do
    read -n1 -r
    [[ $REPLY == ' ' ]] && break
  done
  echo "Continuing ..."
}
# service gdm3 stop
# service lightdm stop

if [ $# != 2 ]; then
  echo "请输入两个参数 name 和 slice_ID"
  exit
fi

killall MATLAB
echo -en '\x52\x53' > /dev/ttyUSB0
# sleep 1
var_datetime=$(date +"%Y%m%d")
var_datadir=$HOME/PatientArchive/
var_datadir=$var_datadir$var_datetime\_$1/$2

if [ -x "$var_datadir" ]; then
  echo $var_datadir
  echo "文件夹已存在，重新输入 name 和 slice_ID。"
  exit
fi

echo $var_datadir
./UDPCatch.sh $1 $2 &
ssh uct2 "~/Program/Capture/UDPCatch.sh $1 $2" &
ssh uct3 "~/Program/Capture/UDPCatch.sh $1 $2" &
ssh uct4 "~/Program/Capture/UDPCatch.sh $1 $2" &

sleep 4
echo "开始发送数据"

# 开始发送数据
echo -en '\x46\x41' > /dev/ttyUSB0

sleep 8

# 开始解码
echo "解码还没添加，需要适配"
# $HOME/Program/Pcap2Bin/DataProcess $HOME/Program/Pcap2Bin/config.ini $var_datetime $1
# ./UDPCatchDecode.sh $subject_var
# | grep workDir | awk -F'[:]' '{print $2}'`
echo "data acq!"
echo $(date +"%H:%M:%S.%N")
