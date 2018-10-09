// -----------------------------------------------------------------------------
// Filename:    UDP_Rcv.hpp
// Revision:    None
// Date:        2018/09/30 - 02:00
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// UDP 收包类

#include <boost/thread/thread_pool.hpp>
#include <vector>

#include "Config.hpp"
#include "Error_Code.hpp"
#include "UDP_Socket.hpp"

class UDP_Rcv {
  private:
    Error_Code error_code;                    // 错误码
    Config *config;                           // 配置信息读取类
    char **data_buffer;                       // 保存的采集信息缓存空间
    char **check_buffer;                      // 保存的验证信息缓存空间
    std::vector<std::string> ip_address_list; // 收包的 IP 地址
    void init_buffer(int ip_count);
    void listening(int ip_index); // 开始监听
    boost::asio::io_context io_context;
    int *msg_cnt;                  // 收包数量计数器，每个线程一个
    unsigned long long *bytes_cnt; // ! 收包字节计数器，可能删除掉

    /* data */
  public:
    UDP_Rcv(Config *config);
    ~UDP_Rcv();
};

void UDP_Rcv::init_buffer(int ip_count) {
    for (uint8_t index = 0; index < ip_count; index++) {
        ip_address_list.push_back("172.16." + std::to_string(config->raw_data_ids[index]) + ".1");
    }
    data_buffer = (char **)malloc(sizeof(char *) * ip_count +
                                  (long long)ip_count * 1400 * PACKET_SUM_PER_INTERFACE * sizeof(char));
    int tmp_index = ip_count;
    if (data_buffer != NULL) {
        char *head = (char *)(data_buffer + ip_count * sizeof(char *));
        while (tmp_index--) {
            data_buffer[tmp_index] = (char *)(head + (int)tmp_index * 1400 * PACKET_SUM_PER_INTERFACE * sizeof(char));
        }
    } else {
        std::cout << error_code.Description(error_code.mem_malloc_error);
    }
    check_buffer =
        (char **)malloc(sizeof(char *) * ip_count + (long long)ip_count * 5 * PACKET_SUM_PER_INTERFACE * sizeof(char));
    tmp_index = ip_count;
    if (check_buffer != NULL) {
        char *head = (char *)(check_buffer + ip_count * sizeof(char *));
        while (tmp_index--) {
            check_buffer[tmp_index] = (char *)(head + (int)tmp_index * 5 * PACKET_SUM_PER_INTERFACE * sizeof(char));
        }
    } else {
        std::cout << error_code.Description(error_code.mem_malloc_error);
    }
}

void UDP_Rcv::listening(int ip_index) {
    UDP_Socket *socket;
    printf("%s\n", ip_address_list[ip_index].c_str());
    socket = new UDP_Socket(io_context, msg_cnt[ip_index], bytes_cnt[ip_index], ip_address_list[ip_index], 58000,
                            &data_buffer[ip_index][0], &check_buffer[ip_index][0]);
    io_context.run();
}

UDP_Rcv::UDP_Rcv(Config *cfg) {
    config = cfg;
    uint8_t ip_count = config->raw_data_ids.size();
    init_buffer(ip_count);
    msg_cnt = (int *)calloc(sizeof(int), ip_count);
    memset(&msg_cnt[0], 0, sizeof(int) * ip_count);
    bytes_cnt = (unsigned long long *)calloc(sizeof(unsigned long long), ip_count);
    memset(&bytes_cnt[0], 0, sizeof(unsigned long long) * ip_count);
    // memset(&data_buffer[0][0], 0, (unsigned long long)ip_address_list.size() *
    // PACKET_LENGTH * PACKET_SUM_PER_INTERFACE); sleep(10);
    int *test = new int[ip_count]();
    int *cnt = new int[ip_count]();

    boost::asio::thread_pool udp_rcv_thread_pool(ip_count);
    // ! index < 1 这里要改成 ip_count 才能监听所有端口，1 为测试用，监听一个端口
    for (int index = 0; index < ip_count; index++) {
        boost::asio::post(udp_rcv_thread_pool, boost::bind(&UDP_Rcv::listening, this, index));
    }
    usleep(40000);
    std::cout << "INFO :: Ready done." << std::endl;
    // TODO 开始监听，告诉 Daemon 程序，可以开始发包
    while (true) {
        sleep(1);

        for (uint8_t index = 0; index < ip_count; index++) {
            if (test[index] != msg_cnt[index]) {
                printf("Interface %d :: %d\n", index, msg_cnt[index]);
                test[index] = msg_cnt[index];
                cnt[index] = 0;
            } else {
                cnt[index]++;
                if (cnt[index] > 3) {
                    msg_cnt[index] = 0;
                    if (test[index] != 0) {
                        printf("Interface %d :: msg_cnt = %d\n", index, msg_cnt[index]);
                    }
                    test[index] = 0;
                    cnt[index] = 0;
                    bytes_cnt[index] = 0;
                }
                if (msg_cnt[index] == 614400) {
                    printf("Interface %d :: Rcv done. %llu bytes\n", index, bytes_cnt[index]);
                    // printf("Interface 1 CNT :: %d\n", msg_cnt[1]);
                    // printf("Interface 2 CNT :: %d\n", msg_cnt[2]);
                    // printf("Interface 3 CNT :: %d\n", msg_cnt[3]);
                    // printf("Interface 4 CNT :: %d\n", msg_cnt[4]);
                    // printf("Interface 5 CNT :: %d\n", msg_cnt[5]);
                    // printf("Interface 6 CNT :: %d\n", msg_cnt[6]);
                    // printf("Interface 7 CNT :: %d\n", msg_cnt[7]);
                    break;
                }
            }
        }
    }
}

UDP_Rcv::~UDP_Rcv() {
    free(data_buffer);
    free(check_buffer);
}
