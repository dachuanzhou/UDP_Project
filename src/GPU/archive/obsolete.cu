// 原始代码，被 filter_func 取代，但保留以备用
__global__ void kernel3(float *filtered_data, short *data_in_process) {
  int column_id = blockDim.x * blockIdx.x + threadIdx.x;
  short data[NSAMPLE];
  float filter_temp_data[NSAMPLE];

  if (column_id < gridDim.x * blockDim.x)    // 没有意义，但是不能删除
  {
    memset(filter_temp_data, 0, NSAMPLE * sizeof(float));
    for (int sample_cnt = 0; sample_cnt < NSAMPLE; sample_cnt++) {
      data[sample_cnt] = data_in_process[sample_cnt * ELE_NO + column_id];
      for (int j = 0; sample_cnt >= j && j < OD; j++)

      {
        filter_temp_data[sample_cnt] +=
          (dev_filter_data[j] * data[sample_cnt - j]);
      }
    }

    for (int i = 0; i < NSAMPLE; i++) {
      filtered_data[i * ELE_NO + column_id] = filter_temp_data[i];
    }
  }
}
