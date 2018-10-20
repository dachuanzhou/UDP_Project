#include <algorithm>
#include <cassert>
#include <vector>
#include "../header/define.hpp"
#include "cuda_runtime.h"
#include "debug/helper_functions.h"
#include "device_launch_parameters.h"

__device__ float dev_ele_coord_x[ELE_NO];    // 写到纹理内存里面
__device__ float dev_ele_coord_y[ELE_NO];    // 写到纹理内存里面
__device__ float dev_filter_data[OD];        // filter parameter

template <typename F>
int bin_search(F f, int beg, int end) {
  if (beg > end) {
    std::swap(beg, end);
  }
  bool res_beg = f(beg);
  bool res_end = f(end);
  assert(res_beg != res_end);
  while (beg < end) {
    int mid = (beg + end) / 2;
    if (res_beg == f(mid)) {
      res_beg = mid + 1;
    } else {
      res_end = mid;
    }
  }
  return beg;
}

// grid param: <pixel_group_idx, pixel_group_idy, sender_offset>
// block param: <pixel_offset_idx, pixel_offset_idy>
// for loop size is roughly equal for each block, never panic
__global__ fast_calc_kernel(           //
    const float* trans_data,           //
    const int sender_id_group_base,    //
    float* image_data,                 //
    int* point_count                   //
) {
  const int sender_id = gridDim.z + sender_id_group_base;
  const float sender_coord_x = dev_ele_coord_x[sender_id];
  const float sender_coord_y = dev_ele_coord_y[sender_id];

  const int pixel_offset_x = threadIdx.x;
  const int pixel_offset_y = threadIdx.y;
  const int pixel_idx = gridDim.x * blockIdx.x + threadIdx.x;
  const int pixel_idy = gridDim.y * blockIdx.y + threadIdx.x;
  const float pixel_coord_x = -image_width / 2 + coord_step * image_x_id;
  const float pixel_coord_y = -image_width / 2 + coord_step * image_y_id;

  const 

}

void fast_calc(                        //
    const float* trans_data,           // adjusted for senders
    const int sender_id_group_base,    //
    const int sender_id_group_size,
    float* image_data,    //
    int* point_count      //
) {
  const float total_max_length =
      (double)(POINT_LENGTH - MIDDOT - (OD - 1 - 1) / 2) / FS * SOUND_SPEED;

  const float total_min_length =
      (double)(100 - MIDDOT - (OD - 1 - 1) / 2) / FS * SOUND_SPEED;
  // float sender_x = dev_ele_coord_x[sender_id];
  // float sender_y = dev_ele_coord_y[sender_id];
  dim3 grid_param(64, 64, sender_id_group_size);
  dim3 block_param(32, 32, 1);
  fast_calc_kernel<<<grid_param, block_param>>>(    //
      trans_data,                                   //
      sender_id_group_base,                         //
      image_data,                                   //
      point_count,                                  //
  );
}

int main() {
  return 0;
}