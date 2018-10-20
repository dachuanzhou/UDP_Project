#pragma once
#include <thrust/device_vector.h>
#include <algorithm>
#include <cassert>
#include <vector>
#include "../header/define.hpp"
#include "cuda_runtime.h"
#include "debug/helper_functions.h"
#include "device_launch_parameters.h"
#include "helper_functions.cuh"

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

// tanf(pi/9)
const float tanpi_9 = 0.36397023426620234;
__device__ bool check_validity(    //
    float sender_coord_x,          //
    float sender_coord_y,          //
    float pixel_coord_x,           //
    float pixel_coord_y            //
) {
  // TODO: try to skip most fucking calculation here
  //  TODO: use angle wisely
  float r_base = -sender_coord_x;
  float im_base = -sender_coord_y;
  float r_target = (pixel_coord_x - sender_coord_x);
  float im_target = (pixel_coord_y - sender_coord_y);
  // complex calc: ~base * target
  float r_compute = r_base * r_target + im_base * im_target;
  float im_compute = r_base * im_target - im_base * r_target;
  bool angle = (r_compute >= 0 && (abs(im_compute / r_compute) < tanpi_9));
  bool range = r_compute < RADIUS * RADIUS;
  return angle && range;
}

// grid param: <pixel_group_idy, pixel_group_idx, sender_offset>
// block param: <pixel_offset_idy, pixel_offset_idx>
// for loop size is roughly equal for each block, never panic
__global__ void fast_calc_kernel(      //
    const float* trans_data,           //
    const int sender_id_group_base,    //
    float* global_image_data,          //
    int* global_count_data             //
) {
  // senders
  const int sender_offset = gridDim.z;
  const int sender_id = gridDim.z + sender_id_group_base;
  const float sender_coord_x = dev_ele_coord_x[sender_id];
  const float sender_coord_y = dev_ele_coord_y[sender_id];

  // pixels
  const int pixel_offset_x = threadIdx.y;
  const int pixel_offset_y = threadIdx.x;
  const int pixel_idx = gridDim.y * blockIdx.y + threadIdx.x;
  const int pixel_idy = gridDim.x * blockIdx.x + threadIdx.x;
  const float pixel_coord_x = -IMAGE_WIDTH / 2 + COORD_STEP * pixel_idx;
  const float pixel_coord_y = -IMAGE_WIDTH / 2 + COORD_STEP * pixel_idy;

  const float dis_snd = distance(pixel_coord_x, pixel_coord_y,    //
                                 sender_coord_x, sender_coord_y);
  bool valid_flag = check_validity(sender_coord_x, sender_coord_y,
                                   pixel_coord_x, pixel_coord_y);

  if (valid_flag) {
    // float fuck = 1.0;
    const int recv_region = 244 * sqrtf(10 * dis_snd);
    const int beg_recv_id = sender_id - recv_region + 1;
    const int end_recv_id = sender_id + recv_region;
    float sum_image_data = 0.0;
    int sum_count_data = 0;
    for (int recv_id_iter = beg_recv_id; recv_id_iter < end_recv_id;
         ++recv_id_iter) {
      // TODO: use some trick to fast kill unnecessary points
      // fuck *= 1.2 + pixel_coord_x - pixel_coord_y;
      const int recv_id = (recv_id_iter + 2048) % 2048;
      const float recv_coord_x = dev_ele_coord_x[recv_id_iter];
      const float recv_coord_y = dev_ele_coord_y[recv_id_iter];
      const float dis_recv =
          distance(pixel_coord_x, pixel_coord_y, recv_coord_x, recv_coord_y);
      const int waves = round((dis_snd + dis_recv) / SOUND_SPEED * FS);
      const int magic = waves + MIDDOT + (OD - 1 - 1) / 2;
      if (magic > 100 && magic <= POINT_LENGTH) {
        const float image = trans_data[recv_id + magic * ELE_NO +
                                       sender_offset * ELE_NO * NSAMPLE];
        sum_image_data += image;
        sum_count_data++;
      }
      // int waves = (dis_snd + dis_recv) / SOUND_SPEED * FS + 0.5;
    }
    int overall_offset = pixel_idx * PIC_RESOLUTION + pixel_idy;
    atomicAdd(global_image_data + overall_offset, sum_image_data);
    atomicAdd(global_image_data + overall_offset, sum_count_data);
    // count_data[] += 1;
  }
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
      point_count                                   //
  );
}

// int main() {
//   // auto dummy = ::cudaMalloc(2048 * 2048 * sizeof(float));
//   // auto dummy_int = (int*)::cudaMalloc(2048 * 2048 * sizeof(float));
//   // auto dummy_float = (float*)::cudaMalloc(2048 * 2048 * sizeof(int));
//   thrust::device_vector<float> in_float(2048 * 2048);
//   thrust::device_vector<int> out_int(2048 * 2048);
//   thrust::device_vector<float> out_float(2048 * 2048);
//   get_ele_position(dev_ele_coord_x, dev_ele_coord_y);

//   fast_calc(thrust::raw_pointer_cast(in_float.data()),     //
//             0, 32,                                         //
//             thrust::raw_pointer_cast(out_float.data()),    //
//             thrust::raw_pointer_cast(out_int.data())       //
//   );
//   return 0;
// }