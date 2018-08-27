// -----------------------------------------------------------------------------
// Filename:    DataProcess.hpp
// Revision:    None
// Date:        2018/08/07 - 22:54
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// 根据病人的信息处理数据

#include <thread>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <vector>
#include <string.h>
#include <fstream>
#include "Patient.hpp"
#include "Config.hpp"

#define PACKET_LENGTH 1452
#define PACKET_SUM_PER_INTERFACE 614400
#define VALID_BYTES_LENGTH_PER_PACKAGE 1400
#define FLAG_BITS_PER_PACKAGE 5

class DataProcess
{
  private:
    /* data */
    long long raw_data_length;
    long long index_data_length;
    long long decode_data_length;
    int start_file_index;
    int end_file_index;
    bool flag_raw_data_ready;
    bool flag_decode_data_ready;
    int cur_index_id;

    /* function */
    inline unsigned int read_as_int(char *ptr);
    int read_pcap_2_memory(std::string filepath, char *packets_raw, char *index_raw);
    void check_udp_packets_order(std::string filepath);
    inline void convert_14bits_to_16bits(long long raw_data_index, long long decode_data_index);
    void process_tables(int start_table_index, int tables_per_thread, int board_sum);

  public:
    /* data */
    Config config;
    Patient patient;
    char *raw_data;
    char *index_data;
    int16_t *decode_data;
    /* function */
    DataProcess(Config in_config, Patient in_patient);
    ~DataProcess();
    int load_slice(int index);
    int check_index_data();
    int decode_slice();
    int map_raw_2_decode();
    int save_decode_data();
};

DataProcess::DataProcess(Config in_config, Patient in_patient) : config(in_config), patient(in_patient)
{
    start_file_index = config.raw_data_ids[0];
    end_file_index = config.raw_data_ids[config.raw_data_ids.size() - 1];

    flag_raw_data_ready = false;
    flag_decode_data_ready = false;
    patient.print_info();
}

DataProcess::~DataProcess()
{
    free(raw_data);
    free(decode_data);
    free(index_data);
}

int DataProcess::load_slice(int index)
// 获得读取文件的范围 start_file_index 和 end_file_index
// 利用多线程把文件加载进入内存 *raw_data 和 *index_data
// 返回值 1 加载成功 其它：加载失败。
{
    if (index > patient.slice_sum)
    {
        return -1;
    }

    cur_index_id = index;
    std::string filepath;
    raw_data_length = (end_file_index - start_file_index + 1) * (long long)PACKET_SUM_PER_INTERFACE * VALID_BYTES_LENGTH_PER_PACKAGE;
    index_data_length = (end_file_index - start_file_index + 1) * (long long)PACKET_SUM_PER_INTERFACE * FLAG_BITS_PER_PACKAGE;

    raw_data = (char *)std::calloc(raw_data_length, 1);
    index_data = (char *)std::calloc(index_data_length, 1);

    if (raw_data == NULL || index_data == NULL)
    {
        std::cout << "ERROR :: Malloc data for raw_data or index_data failed." << std::endl;
        exit(-1);
    }

    boost::asio::thread_pool threadpool(config.program_pcap_read_threads_sum);

    for (int i = start_file_index; i <= end_file_index; i++)
    {
        filepath = patient.path_of_pcapfile_from_slice(index, i);
        // std::cout << filepath << std::endl;
        boost::asio::post(threadpool, boost::bind(&DataProcess::read_pcap_2_memory, this, filepath, &raw_data[(i - start_file_index) * (long long)PACKET_SUM_PER_INTERFACE * VALID_BYTES_LENGTH_PER_PACKAGE], &index_data[(i - start_file_index) * (long long)PACKET_SUM_PER_INTERFACE * FLAG_BITS_PER_PACKAGE]));
    }

    threadpool.join();

    // printf("raw_data_length = %lld, index_data_length = %lld, decode_data_length = %lld\n", raw_data_length, index_data_length, decode_data_length);
    flag_raw_data_ready = true;
    return 1;
}

int DataProcess::check_index_data()
// 利用 read_pcap_2_memory 读取的 index_data 内容识别是否有 UDP 包乱序
// 检测完成，没有问题返回 1
{
    long long cnt = 0;
    long long check_no = 0;
    long long temp = 0;
    while (check_no < (long long)PACKET_SUM_PER_INTERFACE * (end_file_index - start_file_index + 1))
    {
        cnt = 0;
        temp = (int8_t)index_data[5 * check_no + 1];

        if (temp != check_no % 75)
        {
            return -1;
        }

        check_no++;
    }
    return 1;
}

inline unsigned int DataProcess::read_as_int(char *ptr)
// read 4 char as int
{
    unsigned int *int_ptr = (unsigned int *)ptr;
    return *int_ptr;
}

int DataProcess::read_pcap_2_memory(std::string filepath, char *packets_raw, char *index_raw)
// 把 filepath 指向的文件读取到 packets_raw
// packets_raw 保存的内容为 1400*614400 Bytes
// index_raw 保存的内容为 10*614400 Bytes
{
    std::ifstream openfile(filepath.c_str(), std::ifstream::binary | std::ifstream::ate);
    if (!openfile)
    {
        std::cout << "ERROR :: open pcap file error!" << std::endl;
        std::cout << "ERROR :: " << filepath << std::endl;
        return 0;
    }
    long long int filesize = openfile.tellg();
    openfile.seekg(0, openfile.beg);

    // 为 file_buffer 申请空间，并把 filepath 的数据载入内存
    char *file_buffer = (char *)std::malloc(filesize);
    if (file_buffer == NULL)
    {
        std::cout << "ERROR :: Malloc data for buffer failed." << std::endl;
        return 0;
    }
    openfile.read(file_buffer, filesize);
    if (openfile.peek() == EOF)
    {
        openfile.close();
    }
    else
    {
        std::cout << "ERROR :: Read file error." << std::endl;
    }

    // 在 file_buffer 中把实际数据和标记为分别复制到 packets_raw 和 index_raw
    unsigned int block_head_index = 0;
    unsigned int block_length = read_as_int(&file_buffer[block_head_index + 4]);
    int packet_cnt = 0;
    int packet_error_cnt = 0;

    while (block_head_index < filesize)
    {
        block_head_index += block_length;
        block_length = read_as_int(&file_buffer[block_head_index + 4]);

        if (read_as_int(&file_buffer[block_head_index]) != 6)
        {
            continue;
        }
        else if (read_as_int(&file_buffer[block_head_index]) == 6)
        {
            ++packet_cnt;
            if (read_as_int(&file_buffer[block_head_index + 20]) != PACKET_LENGTH)
            {
                ++packet_error_cnt;
            }
            // 把数据读取到 * packets_raw 和 * index_raw
            // From block_head_index + 74 To block_head_index + 1473
            // From block_head_index + 70 To block_head_index + 73
            // From block_head_index + 1474 To block_head_index + 1479
            memcpy(&packets_raw[(packet_cnt - 1) * VALID_BYTES_LENGTH_PER_PACKAGE], &file_buffer[block_head_index + 74], VALID_BYTES_LENGTH_PER_PACKAGE);
            memcpy(&index_raw[(packet_cnt - 1) * 5], &file_buffer[block_head_index + 71], 2);
            // std::cout << std::hex << std::showbase << std::uppercase << read_as_int(&file_buffer[block_head_index + 70]) << std::endl;
            memcpy(&index_raw[(packet_cnt - 1) * 5 + 2], &file_buffer[block_head_index + 1475], 3);
        }
    }

    // 销毁 file_buffer
    std::free(file_buffer);
    if (packet_error_cnt == 0 && packet_cnt == PACKET_SUM_PER_INTERFACE)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

inline void DataProcess::convert_14bits_to_16bits(long long raw_data_index, long long decode_data_index)
// 从 pointer_14 读取 14 个字节（8个 samples，转换为 8个 int16_t （16bits）
// 循环四次，转换32个 samples，正好对应32个通道的 samples
// 调用一次转换 56 bytes
{
    int16_t *ptr;
    char *pointer_14 = &raw_data[raw_data_index];
    int16_t *pointer_16 = &decode_data[decode_data_index];

    // TODO :: 需要根据条带化顺序修改 pointer_16 的索引位置
    for (long long i = 0; i < 4; i++)
    {
        ptr = (int16_t *)&pointer_14[14 * i];
        pointer_16[8 * i + 0] = (ptr[0] & 0xfffc) / 4;
        pointer_16[8 * i + 1] = ((((ptr[0] & 0x3) << 12) | (ptr[1] >> 4)) << 4) / 4;
        pointer_16[8 * i + 2] = ((((ptr[1] & 0xf) << 10) | (ptr[2] >> 6)) << 4) / 4;
        pointer_16[8 * i + 3] = ((((ptr[2] & 0x3f) << 8) | (ptr[3] >> 8)) << 4) / 4;
        pointer_16[8 * i + 4] = ((((ptr[3] & 0xff) << 6) | (ptr[4] >> 10)) << 4) / 4;
        pointer_16[8 * i + 5] = ((((ptr[4] & 0x3ff) << 4) | (ptr[5] >> 12)) << 4) / 4;
        pointer_16[8 * i + 6] = ((((ptr[5] & 0xfff) << 2) | (ptr[6] >> 14)) << 4) / 4;
        pointer_16[8 * i + 7] = (ptr[6] << 2) / 4;
    }

    return;
}

void DataProcess::process_tables(int start_table_index, int tables_per_thread, int board_sum)
{
    for (int cur_table_index = start_table_index; cur_table_index < tables_per_thread + start_table_index; cur_table_index++)
    {

        for (int double_column_index = 0; double_column_index < 1875; double_column_index++)
        {

            for (int board_id = 0; board_id < board_sum; board_id++)
            {
                convert_14bits_to_16bits((long long)860160000 * (0 + board_id * 4) + cur_table_index * (long long)420000 + double_column_index * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_id * 256);
                convert_14bits_to_16bits((long long)860160000 * (2 + board_id * 4) + cur_table_index * (long long)420000 + double_column_index * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_id * 256 + 32);
                convert_14bits_to_16bits((long long)860160000 * (0 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 1875) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_id * 256 + 64);
                convert_14bits_to_16bits((long long)860160000 * (2 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 1875) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_id * 256 + 96);
                convert_14bits_to_16bits((long long)860160000 * (0 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 3750) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_id * 256 + 128);
                convert_14bits_to_16bits((long long)860160000 * (2 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 3750) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_id * 256 + 160);
                convert_14bits_to_16bits((long long)860160000 * (0 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 5625) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_id * 256 + 192);
                convert_14bits_to_16bits((long long)860160000 * (2 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 5625) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_id * 256 + 224);

                convert_14bits_to_16bits((long long)860160000 * (1 + board_id * 4) + cur_table_index * (long long)420000 + double_column_index * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_sum * 256 + board_id * 256);
                convert_14bits_to_16bits((long long)860160000 * (3 + board_id * 4) + cur_table_index * (long long)420000 + double_column_index * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_sum * 256 + board_id * 256 + 32);
                convert_14bits_to_16bits((long long)860160000 * (1 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 1875) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_sum * 256 + board_id * 256 + 64);
                convert_14bits_to_16bits((long long)860160000 * (3 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 1875) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_sum * 256 + board_id * 256 + 96);
                convert_14bits_to_16bits((long long)860160000 * (1 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 3750) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_sum * 256 + board_id * 256 + 128);
                convert_14bits_to_16bits((long long)860160000 * (3 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 3750) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_sum * 256 + board_id * 256 + 160);
                convert_14bits_to_16bits((long long)860160000 * (1 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 5625) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_sum * 256 + board_id * 256 + 192);
                convert_14bits_to_16bits((long long)860160000 * (3 + board_id * 4) + cur_table_index * (long long)420000 + (double_column_index + 5625) * 56, cur_table_index * board_sum * (long long)960000 + double_column_index * board_sum * (long long)512 + board_sum * 256 + board_id * 256 + 224);
            }
        }
    }

    return;
}

int DataProcess::map_raw_2_decode()
{
    int board_sum;
    board_sum = (end_file_index - start_file_index + 1) / 4;

    decode_data_length = (end_file_index - start_file_index + 1) * (long long)PACKET_SUM_PER_INTERFACE * 1600 / 2;
    // decode_data_length = 32;
    decode_data = (int16_t *)std::calloc(decode_data_length, 2);

    if ((end_file_index - start_file_index + 1) % 4 != 0)
    {
        std::cout << "ERROR :: The sum of ports must be times of 4.";
    }

    int tables_per_thread = 2048 / config.program_decode_pcap_threads_sum;
    int rest_tables = 2048 % config.program_decode_pcap_threads_sum;
    boost::asio::thread_pool threadpool(config.program_decode_pcap_threads_sum);

    for (int thread_index = 0; thread_index < config.program_decode_pcap_threads_sum; thread_index++)
    {
        // process_tables(thread_index * tables_per_thread, tables_per_thread, board_sum);
        boost::asio::post(threadpool, boost::bind(&DataProcess::process_tables, this, thread_index * tables_per_thread, tables_per_thread, board_sum));

        // std::cout << "Info :: table from " << (thread_index * tables_per_thread) << " to " << (thread_index * tables_per_thread + tables_per_thread - 1) << std::endl;
    }

    if (rest_tables != 0)
    {
        // process_tables(tables_per_thread * config.program_decode_pcap_threads_sum, rest_tables, board_sum);
        boost::asio::post(threadpool, boost::bind(&DataProcess::process_tables, this, tables_per_thread * config.program_decode_pcap_threads_sum, rest_tables, board_sum));
    }

    // 用线程池的时候要 join()
    threadpool.join();

    free(raw_data);
    free(index_data);
    flag_raw_data_ready = false;
    flag_decode_data_ready = true;

    return 1;
}

int DataProcess::save_decode_data()
{

    if (flag_decode_data_ready == false)
    {
        return 0;
    }

    std::string save_file = config.storage_path + patient.id + "_" + patient.name + "/" + std::to_string(cur_index_id) + "/decode_data.bin";
    std::ofstream f_stream(save_file, std::fstream::out);
    if (f_stream)
    {
        f_stream.write((char *)&decode_data[0], (long long)decode_data_length * 2);

        if (f_stream.good())
        {
            return 1;
        }

        return 0;
    }
    return 0;
}
