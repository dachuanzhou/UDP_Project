#include <algorithm>
#include <vector>

#include "cuda_runtime.h"
#include "debug/helper_functions.h"
#include "device_launch_parameters.h"

extern __device__ float dev_ele_coord_x[ELE_NO];    // 写到纹理内存里面
extern __device__ float dev_ele_coord_y[ELE_NO];    // 写到纹理内存里面
extern __device__ float dev_filter_data[OD];        // filter parameter

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

  const float total_max_length =
    (double)(POINT_LENGTH - MIDDOT - (OD - 1 - 1) / 2) / fs * sound_speed;

  const float total_min_length =
    (double)(100 - MIDDOT - (OD - 1 - 1) / 2) / fs * sound_speed;

  float sender_x = dev_ele_coord_x[sender_id];
  float sender_y = dev_ele_coord_y[sender_id];
  for (int image_x = 0; image_x < PIC_RESOLUTION; ++image_x) {
    auto int left_barrier = 0;
    auto int right_barrier = 2048;
    // circular limit 
    const float coord_x = -IMAGE_WIDTH / 2 + COORD_STEP * image_x_id;
    auto int cir_y = sqrtf(RADIUS * RADIUS - coord_x * coord_x);
    left_barrier = round((-cir_y + IMAGE_WIDTH / 2) / COORD_STEP);
    right_barrier = round((+cir_y + IMAGE_WIDTH / 2) / COORD_STEP);
    

    for (int image_y = 0; image_y < PIC_RESOLUTION; ++image_y) {
      const float coord_y = -IMAGE_WIDTH / 2 + COORD_STEP * image_y_id;
      float dis_snd = distance(coord_x, coord_y, sender_x, sender_y);
    }
  }
}