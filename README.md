# 生科院的 UDP 收包项目

## ●　UDP_wireshark

利用 `wireshark` 和 `tshark` 工具进行抓包。临时方案。  
两台服务器，`CentOS 7.5` + `Ubuntu 16.04 LTS server`。  
`Ubuntu 16.04 LTS server` 使用 USB 串口控制扫描装置开始扫描。
`Ubuntu 16.04 LTS server` 开启 Samba 服务让外部设备访问扫描结果。

## Todolist

- [ ] 将 `wireshark` 抓取的 pcap 文件处理成单个可用的 slice 文件。
