#include <algorithm>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float dev_ele_coord_x[ELE_NO];    // 写到纹理内存里面
__device__ float dev_ele_coord_y[ELE_NO];    // 写到纹理内存里面
__device__ float dev_filter_data[OD];        // filter parameter

template <typename F>
void bin_search(F f, int beg, int end) {
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


void fast_calc(const float* trans_data,
               const int sender_id,    //
               float* image_data,      //
               int* point_count        //
) {
  //
  constexpr float total_length = 

  float sender_x = dev_ele_coord_x[sender_id];
  float sender_y = dev_ele_coord_y[sender_id];
  for (int image_x = 0; image_x < PIC_RESOLUTION; ++image_x) {
    for (int image_y = 0; image_y < PIC_RESOLUTION; ++image_y) {
      
    }
  }
}