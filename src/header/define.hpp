#pragma once
#ifndef __CONST_VALUE__
#define __CONST_VALUE__
constexpr auto PI = 3.14159265358979323846;
constexpr auto PIC_RESOLUTION = 4096;    // 出图的分辨率：N x N, must be multiple of 64

// 接收阵元到发射阵元的最大距离（阵元个数），所以接收孔径为2*M+1
constexpr auto RCV_OFFSET = 256;

constexpr auto ELE_NO = 2048;         // 发射振元的数量, 
constexpr auto OD = 64;               // 滤波参数
constexpr auto NSAMPLE = 3750;        // 每一次发射的采样数量
constexpr auto SOUND_SPEED = 1520;    // 声速
constexpr auto FS = 25e6;             // 频率
constexpr auto IMAGE_WIDTH = 0.20;    // 成像方形边长
constexpr auto DATA_DIAMETER = 0.22;
constexpr auto POINT_LENGTH = DATA_DIAMETER / SOUND_SPEED * FS + 0.5;
constexpr auto COORD_STEP = IMAGE_WIDTH / (PIC_RESOLUTION - 1);
constexpr auto MIDDOT = -160;
constexpr auto TGC = 0.0014;
constexpr auto RADIUS = 0.112;    // 探头半径
constexpr auto PACKET_LENGTH = 1452;
constexpr auto PACKET_SUM_PER_INTERFACE = 614400;
constexpr auto VALID_BYTES_LENGTH_PER_PACKAGE = 1400;
constexpr auto FLAG_BITS_PER_PACKAGE = 5;
#endif
