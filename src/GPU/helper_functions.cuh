#pragma once
#include <time.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_functions.cuh"
#include "../header/define.hpp"

inline __host__ __device__ float distance(float x1, float y1, float x2, float y2) {
  auto dx = x1 - x2;
  auto dy = y1 - y2;
  return sqrtf(dx * dx + dy * dy);
}

bool __device__ __host__ is_close(int delta, int range) {
  int abs_delta = abs(delta);
  return (abs_delta < range || range > 2048 - range);
  // return (delta + range + 2047) % 2048 < 2 * range - 1;
}

// 滤波函数
__global__ void filter_func(float *filtered_data, short *data_in_process) {
  int column_id = blockDim.x * blockIdx.x + threadIdx.x;
  for (int sample_cnt = 0; sample_cnt < NSAMPLE; sample_cnt++) {
    for (int j = 0; sample_cnt >= j && j < OD; j++) {
      filtered_data[(column_id / 2048) * ELE_NO * NSAMPLE +
                    sample_cnt * ELE_NO + column_id % 2048] +=
        (dev_filter_data[j] *
         data_in_process[(sample_cnt - j) * ELE_NO +
                         (column_id / 2048) * ELE_NO * NSAMPLE +
                         column_id % 2048]);
    }
  }
}

__global__ void add(float *sumdata, int *sumpoint, float *imagedata,
                    int *point_count) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < PIC_RESOLUTION * PIC_RESOLUTION) {
    sumdata[tid] += imagedata[tid];
    sumpoint[tid] += point_count[tid];
    tid += blockDim.x * gridDim.x;
  }
}

void get_ele_position(float *ele_coord_x, float *ele_coord_y) {
  float rfocus = (float)112 / 1000;
  float ele_angle = (2 * PI * 43.4695 / (256 - 1)) / 360;    //阵元间隔角度
  float first_one = 2 * PI * (45 - 43.4695) / 360;    //第一个阵元角度

  for (int i = 0; i < 256; i++) {
    ele_coord_x[i] = rfocus * cos(first_one + i * ele_angle);
    ele_coord_y[i] = -rfocus * sin(first_one + i * ele_angle);
  }
  for (int i = 256; i < 512; i++) {
    ele_coord_x[i] = rfocus * cos(first_one + (i - 256) * ele_angle + PI / 4);
    ele_coord_y[i] = -rfocus * sin(first_one + (i - 256) * ele_angle + PI / 4);
  }
  for (int i = 512; i < 768; i++) {
    ele_coord_x[i] = rfocus * cos(first_one + (i - 512) * ele_angle + PI / 2);
    ele_coord_y[i] = -rfocus * sin(first_one + (i - 512) * ele_angle + PI / 2);
  }
  for (int i = 768; i < 1024; i++) {
    ele_coord_x[i] =
      rfocus * cos(first_one + (i - 768) * ele_angle + 3 * PI / 4);
    ele_coord_y[i] =
      -rfocus * sin(first_one + (i - 768) * ele_angle + 3 * PI / 4);
  }
  for (int i = 1024; i < 1280; i++) {
    ele_coord_x[i] = rfocus * cos(first_one + (i - 1024) * ele_angle + PI);
    ele_coord_y[i] = -rfocus * sin(first_one + (i - 1024) * ele_angle + PI);
  }
  for (int i = 1280; i < 1536; i++) {
    ele_coord_x[i] =
      rfocus * cos(first_one + (i - 1280) * ele_angle + 5 * PI / 4);
    ele_coord_y[i] =
      -rfocus * sin(first_one + (i - 1280) * ele_angle + 5 * PI / 4);
  }
  for (int i = 1536; i < 1792; i++) {
    ele_coord_x[i] =
      rfocus * cos(first_one + (i - 1536) * ele_angle + 3 * PI / 2);
    ele_coord_y[i] =
      -rfocus * sin(first_one + (i - 1536) * ele_angle + 3 * PI / 2);
  }
  for (int i = 1792; i < 2048; i++) {
    ele_coord_x[i] =
      rfocus * cos(first_one + (i - 1792) * ele_angle + 7 * PI / 4);
    ele_coord_y[i] =
      -rfocus * sin(first_one + (i - 1792) * ele_angle + 7 * PI / 4);
  }
}