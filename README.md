# 生科院的 UDP 收包项目 + GPU 

`VSCode` 项目，调试编译均用 `VSCode` 完成。

## ●　src

该目录包含所有源文件。

- `header` 目录下包含所有的头文件（类）
- `UDP_Rcv` 目录下包含收包程序主文件
- `GPU` 目录下包含 GPU 重建运算的主程序
- `txt2png` 目录下包含重建数据建立图像的主程序

`src` 目录下调用 `make` 命令会自动编译所有的程序并存放到项目文件夹下的 `release` 目录中。

## ●　debug & release

`debug` 和 `release` 目录主要用来调试和生成可用的执行程序。

## ●　Tools

`Tools` 目录存放辅助工具和其它调试用的脚本。

## ●　UDP_wireshark

`UDP_wireshark` 目录下为用 `wireshark` 抓包配套的脚本，需要和 `Pcap2Bin` 程序搭配使用。

---

## Todolist

●　**UDP_Receiver**

- [x] 多线程监听不同的 IP 地址
- [ ] 异常状态的判断
- [ ] 整合 *UDP_wireshark* 中的功能
- [ ] daemon 进程后台调度

●　**Daemon 进程**

- [ ] 和其它节点通信
- [ ] 和工控机通信
- [ ] 调用 *UDP_Receiver*
- [ ] 调用 *CUDA* 程序
- [ ] 节点状态检测管理

●　**UDP_wireshark**

- [x] 将 `wireshark` 抓取的 pcap 文件读取到内存
- [x] 提取原始 UDP 数据包中的有效数据
- [x] 把有效数据转从 14bit 为最小单位换成为 16bit，数据类型为 signed int
- [x] 将所有数据按照 GPU 中读取矩阵的顺序保存成文件
