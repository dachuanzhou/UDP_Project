#ifndef __CONST_VALUE__
#define __CONST_VALUE__
#define PI 3.14159265358979323846
#define PIC_RESOLUTION 2048 // 出图的分辨率：N x N
#define RCV_OFFSET 256 // 接收阵元到发射阵元的最大距离（阵元个数），所以接收孔径为2*M+1
#define ELE_NO 2048      // 发射振元的数量
#define OD 64            // 滤波参数
#define NSAMPLE 3750     // 每一次发射的采样数量
#define SOUND_SPEED 1520 // 声速
#define FS 25e6          // 频率
#define IMAGE_WIDTH 0.20 // 成像方形边长
#define DATA_DIAMETER 0.22
#define POINT_LENGTH DATA_DIAMETER / SOUND_SPEED *FS + 0.5
#define COORD_STEP IMAGE_WIDTH / (N - 1)
#define MIDDOT -160
#define TGC 0.0014
#define RADIUS 0.112 // 探头半径
#define PACKET_LENGTH 1452
#define PACKET_SUM_PER_INTERFACE 614400
#define VALID_BYTES_LENGTH_PER_PACKAGE 1400
#define FLAG_BITS_PER_PACKAGE 5
#endif
