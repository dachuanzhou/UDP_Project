// -----------------------------------------------------------------------------
// Filename:    rebuild.cu
// Revision:    None
// Date:        2018/10/16 - 03:19
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// 重建工作的代码重构

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"

#include "../header/define.hpp"

// 全局变量

int parallel_emit_sum = 1;

bool load_file_2_memory(std::string filepath, char *&data_raw)
// 把 filepath 指向的文件读取到 data_raw
{
    // TODO 以二进制方式读取 std::ifstream::binary
    std::ifstream openfile(filepath.c_str(), std::ifstream::binary | std::ifstream::ate);
    if (!openfile) {
        printf("ERROR :: load \"%s\" failed.\n", filepath.c_str());
        return false;
    }
    size_t filesize = openfile.tellg();
    openfile.seekg(0, openfile.beg);

    // 为 data_raw 申请空间，并把 filepath 的数据载入内存
    data_raw = (char *)std::malloc(filesize * sizeof(char));
    if (data_raw == NULL) {
        printf("ERROR :: Malloc memory for \"%s\" failed.\n", filepath.c_str());
        return false;
    }
    openfile.read(data_raw, filesize);
    if (openfile.peek() == EOF) {
        openfile.close();
        printf("INFO :: load \"%s\" to memory done.\n", filepath.c_str());
        return true;

    } else {
        printf("ERROR :: Read \"%s\" failed.\n", filepath.c_str());
        return false;
    }
}

bool read_files(std::string &filter_path, std::string &bin_path, char *&filter_buffer, char *&bin_buffer) {
    if (!load_file_2_memory(filter_path, filter_buffer)) {
        printf("Error :: Load %s filter failed.\n", filter_path.c_str());
        return false;
    }
    if (!load_file_2_memory(bin_path, bin_buffer)) {
        printf("Error :: Load %s filter failed.\n", bin_path.c_str());
        return false;
    }
    return true;
}

int main(int argc, char const *argv[]) {
    auto start_time_main = std::chrono::high_resolution_clock::now();
    // 运行时的参数处理
    // ! 第一个参数，并发计算的发射源数量
    // ! 第二个参数，滤波器文件路径
    // ! 第三个参数，源数据文件(bin)路径
    // ! 第四个参数，输出重建文件的(bin or txt)路径
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
        std::cout << "[parallel emit sum] [filter path] [bin path] [output path]" << std::endl;
        exit(-1);
        break;
    }

    char *bin_buffer;
    char *filter_buffer;
    if(!read_files(filter_path, bin_path, filter_buffer, bin_buffer)){
        free(filter_buffer);
        free(bin_buffer);
        exit(-1);
    }
    auto end_time_main = std::chrono::high_resolution_clock::now();
    std::cout << "INFO :: Total running "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_main - start_time_main).count() << "ms"
              << std::endl;
}