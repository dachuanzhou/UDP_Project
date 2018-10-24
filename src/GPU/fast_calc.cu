#include <time.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <chrono>
using std::cin;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::ios_base;
using std::ofstream;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../header/define.hpp"
constexpr int DEBUG_SAMPLE_RATE_REV = 32;
int parallel_emit_sum = 1;    // 并行处理多个发射节点，优化使用

__constant__ float dev_filter_data_reverse[OD];
__device__ float dev_ele_coord_x[ELE_NO];    // 写到纹理内存里面
__device__ float dev_ele_coord_y[ELE_NO];    // 写到纹理内存里面
__device__ float dev_filter_data[OD];        // filter parameter

#include "fast_calc_kernel.cuh"
#include "helper_functions.cuh"

float image_data[PIC_RESOLUTION * PIC_RESOLUTION] = {0};
int image_point_count[PIC_RESOLUTION * PIC_RESOLUTION] = {0};


static double total_time_consumption = 0;
cudaError_t precalcWithCuda(short *dev_data_samples_in_process, int ele_emit_id,
                            float *dev_sumdata, int *dev_sumpoint,
                            float *dev_filtered_data, float *dev_imagedata,
                            int *dev_pointcount, int parallel_emit_sum) {
  cudaError_t cudaStatus;

  // kernel 1,kernel2 decode
  // kernel3 filter
  auto begin = std::chrono::high_resolution_clock::now();
  // cudaMemset(dev_filtered_data, 0,
  //            NSAMPLE * ELE_NO * sizeof(float) * parallel_emit_sum);
  // filter_func<<<4 * parallel_emit_sum, 512>>>(dev_filtered_data,
  //                                             dev_data_samples_in_process);
  fast_filter(dev_filtered_data, dev_data_samples_in_process, parallel_emit_sum);

  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  total_time_consumption += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

  // dim3 gridimage(PIC_RESOLUTION, PIC_RESOLUTION);
  // // dim3 threads(RCV_OFFSET);
  // calc_func<<<gridimage, 2 * RCV_OFFSET>>>(
  //   ele_emit_id, dev_imagedata, dev_pointcount, dev_filtered_data,
  //   parallel_emit_sum);    //启动一个二维的PIC_RESOLUTION*PIC_RESOLUTION个block，每个block里面RCV_OFFSET个thread





  fast_calc(dev_filtered_data, ele_emit_id, parallel_emit_sum, dev_sumdata, dev_sumpoint);

  // // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  // if (cudaStatus != cudaSuccess) {
  //   cout << "calcKernel launch failed: " << cudaGetErrorString(cudaStatus);
  //   // goto Error;
  //   return cudaStatus;
  // }
  // // cudaDeviceSynchronize();

  // //把所有的结果加到一起
  // add<<<32, 32>>>(dev_sumdata, dev_sumpoint, dev_imagedata, dev_pointcount);
  // cudaStatus = cudaGetLastError();
  // if (cudaStatus != cudaSuccess) {
  //   cout << "addKernel launch failed: " << cudaGetErrorString(cudaStatus);
  //   // goto Error;
  //   return cudaStatus;
  // }

  return cudaStatus;
}



void write_txtfile(std::string output_path) {
  ofstream outfile(output_path);
  if (!outfile.is_open()) {
    cout << " the file open fail" << endl;
    exit(1);
  }

  for (int k = 0; k < PIC_RESOLUTION; k++) {
    for (int j = 0; j < PIC_RESOLUTION; j++) {
      if (image_point_count[k * PIC_RESOLUTION + j] == 0)
        outfile << image_data[k * PIC_RESOLUTION + j] << " ";
      else
        outfile << image_data[k * PIC_RESOLUTION + j] /
                     image_point_count[k * PIC_RESOLUTION + j]
                << " ";
    }
    outfile << "\r\n";
  }

  outfile.close();
}

int main(int argc, char const *argv[]) {
  time_t start, over;
  start = time(NULL);

  std::string filter_path = "";
  std::string bin_path = "";
  std::string output_path = "";
  switch (argc) {
    case 4:
      parallel_emit_sum = atoi(argv[1]);
      filter_path = argv[2];
      bin_path = argv[3];
      output_path = "origin.txt";
      break;
    case 5:
      parallel_emit_sum = atoi(argv[1]);
      filter_path = argv[2];
      bin_path = argv[3];
      output_path = argv[4];
      break;
    default:
      std::cout << "Please input 3 or 4 paras" << std::endl;
      std::cout << "[parallel emit sum] [filter path] [bin path]" << std::endl;
      std::cout << "[parallel emit sum] [filter path] [bin path] [output path]"
                << std::endl;
      exit(-1);
      break;
  }
  // parallel_emit_sum = 16;

  cudaError_t cudaStatus;

  time_t start_read, over_read;
  start_read = time(NULL);
  // Read filter data and put in GPU
  ifstream file_read;
  file_read.open(filter_path.c_str(), ios_base::in | ios::binary);
  if (!file_read.is_open()) {
    cout << " the file filter open fail" << endl;
    return -1;
  }
  float filter_data[OD];
  // for (int ii = 0; ii < OD; ii++) {
  file_read.read((char*)filter_data, OD * sizeof(float));
  // }
  file_read.close();
  float filter_data_reverse[OD];
  // cout << ">>";
  for(int i = 0; i < OD; ++i){
    filter_data_reverse[i] = filter_data[OD - 1 - i];
    // cout << filter_data[OD - 1 - i] << " ";
  }
  // cout << endl;
  cudaStatus =
    cudaMemcpyToSymbol(dev_filter_data, filter_data, sizeof(float) * OD);

  cudaStatus =
    cudaMemcpyToSymbol(dev_filter_data_reverse, filter_data_reverse, sizeof(float) * OD);

  if (cudaStatus != cudaSuccess) {
    cout << "center Fail to cudaMemcpyToSymbol on GPU" << endl;
    return -1;
  }

  file_read.open(bin_path.c_str(), ios_base::in | ios::binary | ios::ate);
  if (!file_read.is_open()) {
    cout << " the bin file open fail" << endl;
    return -1;
  }
  long long int filesize = file_read.tellg() / DEBUG_SAMPLE_RATE_REV;
  file_read.seekg(0, file_read.beg);
  // 为 bin_buffer 申请空间，并把 filepath 的数据载入内存
  char *bin_buffer = (char *)std::malloc(filesize);
  if (bin_buffer == NULL) {
    std::cout << "ERROR :: Malloc data for buffer failed." << std::endl;
    return -1;
  }
  file_read.read(bin_buffer, filesize);
  // if (file_read.peek() == EOF) {
  //   file_read.close();
  // } else {
  //   std::cout << "ERROR :: Read bin file error." << std::endl;
  //   exit(-1);
  // }
  over_read = time(NULL);
  cout << "Reading time is : " << difftime(over_read, start_read) << "s!"
       << endl;

  // image grid
  float ele_coord_x[ELE_NO] = {0};
  float ele_coord_y[ELE_NO] = {0};
  get_ele_position(&ele_coord_x[0], &ele_coord_y[0]);

  if (cudaMemcpyToSymbol(dev_ele_coord_x, ele_coord_x,
                         sizeof(float) * ELE_NO) != cudaSuccess) {
    cout << "ERROR :: Failed for cudaMemcpyToSymbol dev_ele_coord_x." << endl;
    return -1;
  }

  if (cudaMemcpyToSymbol(dev_ele_coord_y, ele_coord_y,
                         sizeof(float) * ELE_NO) != cudaSuccess) {
    cout << "ERROR :: Failed for cudaMemcpyToSymbol dev_ele_coord_y." << endl;
    return -1;
  }

  float *dev_sumdata;
  int *dev_sumpoint;
  if (cudaMalloc((void **)(&dev_sumdata), PIC_RESOLUTION * PIC_RESOLUTION *
                                            sizeof(float)) != cudaSuccess) {
    cout << "ERROR :: Failed for cudaMalloc dev_sumdata." << endl;
    return -1;
  }
  if (cudaMalloc((void **)(&dev_sumpoint), PIC_RESOLUTION * PIC_RESOLUTION *
                                             sizeof(int)) != cudaSuccess) {
    cout << "ERROR :: Failed for cudaMalloc dev_sumpoint." << endl;
    return -1;
  }
  // init dev_sumdata and dev_sumpoint
  if (cudaMemcpy(dev_sumdata, image_data,
                 PIC_RESOLUTION * PIC_RESOLUTION * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cout << "ERROR :: Failed for cudaMemcpy dev_sumdata." << endl;
    return -1;
  }
  if (cudaMemcpy(dev_sumpoint, image_point_count,
                 PIC_RESOLUTION * PIC_RESOLUTION * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cout << "ERROR :: Failed for cudaMemcpy dev_sumpoint." << endl;
    return -1;
  }

  long long length_of_data_in_process =
    NSAMPLE * ELE_NO * sizeof(short) * parallel_emit_sum;
  short *dev_data_samples_in_process;
  float *dev_filtered_data;

  cudaStatus = cudaMalloc((void **)(&dev_data_samples_in_process),
                          length_of_data_in_process);
  /* if (cudaStatus != cudaSuccess)
    {
        cout << "data_samples_in_process Fail to cudaMalloc on GPU" << endl;
        return -1;
    } */

  cudaStatus =
    cudaMalloc((void **)(&dev_filtered_data), length_of_data_in_process * 2);
  /* if (cudaStatus != cudaSuccess) // 转 float 乘以 2
    {
        cout << "ERROR :: Failed for cudaMalloc dev_filtered_data." << endl;
        return -1;
    } */

  float *dev_imagedata;

  int *dev_pointcount;

  cudaStatus = cudaMalloc((void **)(&dev_imagedata),
                          PIC_RESOLUTION * PIC_RESOLUTION * sizeof(float));
  /* if (cudaStatus != cudaSuccess)
    {
        cout << "imagedata Fail to cudaMalloc on GPU" << endl;
        //goto Error;
        return cudaStatus;
    } */
  cudaStatus = cudaMalloc((void **)(&dev_pointcount),
                          PIC_RESOLUTION * PIC_RESOLUTION * sizeof(int));
  /* if (cudaStatus != cudaSuccess)
    {
        cout << "pointcount Fail to cudaMalloc on GPU" << endl;
        //goto Error;
        return cudaStatus;
    } */

  long long bin_buffer_index = 0;
  for (int ele_emit_id = 0; ele_emit_id < ELE_NO / DEBUG_SAMPLE_RATE_REV;
       ele_emit_id += parallel_emit_sum) {
    fprintf(stderr, "wavesber of element : %d\n", ele_emit_id);

    // memcpy(&data_samples_in_process[0], &bin_buffer[bin_buffer_index], length_of_data_in_process);
    // bin_buffer_index = bin_buffer_index + length_of_data_in_process;

    // cudaStatus = cudaMemcpy(dev_data_samples_in_process,
    // data_samples_in_process, length_of_data_in_process,
    // cudaMemcpyHostToDevice);
    cudaStatus =
      cudaMemcpy(dev_data_samples_in_process, &bin_buffer[bin_buffer_index],
                 length_of_data_in_process, cudaMemcpyHostToDevice);
    bin_buffer_index = bin_buffer_index + length_of_data_in_process;
    if (cudaStatus != cudaSuccess) {
      cout << "data_samples_in_process Fail to cudaMemcpy on GPU" << endl;
      // goto Error;
      return cudaStatus;
    }
    cudaStatus = precalcWithCuda(
      dev_data_samples_in_process, ele_emit_id, dev_sumdata, dev_sumpoint,
      dev_filtered_data, dev_imagedata, dev_pointcount, parallel_emit_sum);
    //}
    // over=time(NULL);
    // cout<<"Running time is : "<<difftime(over,start)<<"s!"<<endl;
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "calcWithCuda failed!");
      return 1;
    }
    // cudaError_t cudaStatus = calcWithCuda(
    // i,dev_sumdata,dev_sumpoint,dev_filtered_data);
    // ::cudaDeviceSynchronize();
  }
  cudaStatus = cudaMemcpy(image_data, dev_sumdata,
                          PIC_RESOLUTION * PIC_RESOLUTION * sizeof(float),
                          cudaMemcpyDeviceToHost);
  /* if (cudaStatus != cudaSuccess)
    {
        cout << "allimagedata Fail to cudaMemcpy to CPU" << endl;
        return 1;
        //goto Error;
    } */

  cudaStatus = cudaMemcpy(image_point_count, dev_sumpoint,
                          PIC_RESOLUTION * PIC_RESOLUTION * sizeof(int),
                          cudaMemcpyDeviceToHost);
  /* if (cudaStatus != cudaSuccess)
    {
        cout << "allpointcount Fail to cudaMemcpy to CPU" << endl;
        return 1;
        //goto Error;
    } */

  write_txtfile(output_path);
  over = time(NULL);
  cout << "Running time is : " << (int)difftime(over, start) / 60 << "min "
       << (int)difftime(over, start) % 60 << "s." << endl;
  cout << "Crucial time is: " << total_time_consumption / 1000 << endl;
  cudaFree(dev_sumdata);
  cudaFree(dev_sumpoint);
  cudaFree(dev_data_samples_in_process);
  cudaFree(dev_filtered_data);
  cudaFree(dev_imagedata);
  cudaFree(dev_pointcount);
  // cudaStatus = cudaDeviceReset();
  // if (cudaStatus != cudaSuccess)
  // {
  //     fprintf(stderr, "cudaDeviceReset failed!");
  //     return 1;
  // }
  return 0;
}
