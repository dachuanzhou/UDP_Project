// -----------------------------------------------------------------------------
// Filename:    DataProcessTest.cpp
// Revision:    None
// Date:        2018/08/07 - 23:08
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// 测试 DataProcess 类

// #include <time.h>
#include <chrono>
#include "DataProcess.hpp"

int main(int argc, char const *argv[])
{
    auto start_time_main = std::chrono::high_resolution_clock::now();
    std::string cfg_file, id, name;
    int slice_index;

    switch (argc)
    {
    case 1:
        cfg_file = "config.ini";
        id = "1";
        name = "TEST";
        slice_index = 1;
        break;
    case 2:
        cfg_file = argv[1];
        id = "1";
        name = "TEST";
        slice_index = 1;
        break;
    case 4:
        cfg_file = argv[1];
        id = argv[2];
        name = argv[3];
        slice_index = 1;
        break;
    case 5:
        cfg_file = argv[1];
        id = argv[2];
        name = argv[3];
        slice_index = atoi(argv[4]);
        break;
    default:
        std::cout << "Args are not correct, such as:" << std::endl
                  << "[Config file]" << std::endl
                  << "[config file] [ID] [Name]" << std::endl;
        return -1;
        break;
    }

    Config config(cfg_file);

    Patient *pptr = new Patient(config.storage_path, id, name);
    DataProcess *ptr = new DataProcess(config, *pptr);

    if (ptr->load_slice(slice_index) == 1)
    {
        std::cout << "Slice " << slice_index << " : "
                  << "pcap from " << config.raw_data_ids[0] << " to " << config.raw_data_ids[config.raw_data_ids.size() - 1] << " load successful." << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_main).count() << "ms for load pcap files." << std::endl;

    auto start_time_check = std::chrono::high_resolution_clock::now();
    if (ptr->check_index_data())
    {
        std::cout << "ERROR :: slice " << slice_index << " order error." << std::endl;
        return 0;
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_check).count() << "ms for check package order." << std::endl;

    auto start_time_decode = std::chrono::high_resolution_clock::now();
    if (ptr->decode_slice() != 1)
    {
        std::cout << "ERROR :: slice " << slice_index << " decode error." << std::endl;
        return 0;
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_decode).count() << "ms for decode data." << std::endl;

    free(pptr);
    free(ptr);

    end_time = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_main).count() << "ms" << std::endl;

    exit(0);
}
