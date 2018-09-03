// -----------------------------------------------------------------------------
// Filename:    VerifyValue.cpp
// Revision:    None
// Date:        2018/09/01 - 22:12
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// 在生成的大 Bin 文件中读取数值并判断范围有没有溢出

#include <iostream>
#include <fstream>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <boost/thread/mutex.hpp>

#define THREAD_SUM 40

boost::atomic_llong error_sum;
boost::atomic_llong checked_sum;
char *file_buffer1;
char *file_buffer2;
boost::mutex io_mutex;

void verify_2_value(long long start_offset, long long verify_length)
{
    int16_t *file1, *file2;
    file1 = (int16_t *)&file_buffer1[start_offset];
    file2 = (int16_t *)&file_buffer2[start_offset];
    for (long long index = 0; index < verify_length; index++)
    {
        if (file1[index] != file2[index])
        {
            printf("%04lld-%04lld-%08lld ::", index / (2048 * 3750), index % (2048 * 3750) / 2048, (index % (2048 * 3750)) % 2048);
            printf("** %4d %4d %4x\n", file1[index], file2[index], file1[index] - file2[index]);
        }
    }
}

void verify_value(long long start_offset, long long verify_length)
{
    int16_t *verify_data = (int16_t *)&file_buffer1[start_offset];
    for (long long index = 0; index < verify_length; index++)
    {
        if (verify_data[index] > 16383 || verify_data[index] < -16384)
        {
            error_sum++;
        }
    }
    boost::mutex::scoped_lock lock(io_mutex);
    checked_sum = checked_sum + verify_length;
    std::cout << "\rVerified Samples = " << checked_sum;
    return;
}

int main(int argc, char const *argv[])
{
    error_sum = 0;
    checked_sum = 0;
    std::string verify_file;
    std::string compare_file;
    switch (argc)
    {
    case 1:
        std::cout << "Please input the file path." << std::endl;
        exit(0);
        break;
    case 2:
        verify_file = argv[1];
        break;
    case 3:
        verify_file = argv[1];
        compare_file = argv[2];
        break;
    default:
        verify_file = argv[0];
        std::cout << "Args too much, only 1 or 2 paras for file path." << std::endl;
        return -1;
        break;
    }

    // 读取文件 verifyfile
    std::ifstream verifyfile(verify_file.c_str(), std::ifstream::binary | std::ifstream::ate);
    if (!verifyfile)
    {
        std::cout << "ERROR :: open pcap file error!" << std::endl;
        std::cout << "ERROR :: " << verify_file << std::endl;
        return 0;
    }
    long long int filesize = verifyfile.tellg();
    verifyfile.seekg(0, verifyfile.beg);

    // 为 file_buffer1 申请空间，并把 verifyfile 的数据载入内存
    file_buffer1 = (char *)std::malloc(filesize);
    if (file_buffer1 == NULL)
    {
        std::cout << "ERROR :: Malloc data for buffer failed." << std::endl;
        return 0;
    }
    verifyfile.read(file_buffer1, filesize);
    if (verifyfile.peek() == EOF)
    {
        verifyfile.close();
    }
    else
    {
        std::cout << "ERROR :: Read file error." << std::endl;
    }
    std::cout << "INFO :: File loaded!" << std::endl;

    if (argc == 2)
    {
        long long sample_sum = filesize / 2;
        long long samples_per_thread = sample_sum / 1000;
        std::cout << "INFO :: Samples per thread = " << samples_per_thread << std::endl;

        boost::asio::thread_pool threadpool(THREAD_SUM);
        for (int i = 0; i < 1000; i++)
        {
            // verify_value(i * samples_per_thread, samples_per_thread);
            boost::asio::post(threadpool, boost::bind(&verify_value, i * samples_per_thread, samples_per_thread));
        }

        threadpool.join();

        std::cout << "\nError Sum = " << error_sum << std::endl;
    }
    else if (argc == 3)
    {
        // 读取文件 verifyfile2
        std::ifstream verifyfile2(compare_file.c_str(), std::ifstream::binary);
        if (!verifyfile)
        {
            std::cout << "ERROR :: open pcap file error!" << std::endl;
            std::cout << "ERROR :: " << compare_file << std::endl;
            return 0;
        }
        // 为 file_buffer2 申请空间，并把 verifyfile2 的数据载入内存
        file_buffer2 = (char *)std::malloc(7680000 * 2 * 2);
        if (file_buffer2 == NULL)
        {
            std::cout << "ERROR :: Malloc data for buffer failed." << std::endl;
            return 0;
        }
        verifyfile2.read(file_buffer2, 7680000 * 2 * 2);
        verifyfile2.close();
        std::cout << "INFO :: File loaded!" << std::endl;
        verify_2_value(0, 7680000 * 2);
    }

    return 0;
}
