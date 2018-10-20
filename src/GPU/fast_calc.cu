#include <algorithm>
#include <vector>
#include <cassert>
#include "cuda_runtime.h"
#include "debug/helper_functions.h"
#include "device_launch_parameters.h"
#include "../header/define.hpp"

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


void fast_calc(const float* trans_data,           // adjusted for senders
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
      

}


// fast_calc(const float* trans_data){}

int main() {
  return 0;
}