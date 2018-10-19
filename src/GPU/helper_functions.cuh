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
