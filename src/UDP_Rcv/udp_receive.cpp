// -----------------------------------------------------------------------------
// Filename:    udp_rcv_test.cpp
// Revision:    None
// Date:        2018/09/30 - 02:21
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// 测试 UDP_Rcv.hpp

#include <string>

#include "../header/UDP_Rcv.hpp"

int main(int argc, char const *argv[]) {
    Error_Code error_code;
    std::string config_file;
    switch (argc) {
    case 2:
        config_file = argv[1];
        break;
    default:
        std::cout << "参数接收：配置文件" << std::endl;
        std::cout << error_code.Description(error_code.main_argc_err);
        return error_code.main_argc_err;
    }
    Config *config;
    UDP_Rcv *udp_rcv;
    config = new Config(config_file);
    udp_rcv = new UDP_Rcv(config);

    // study

    return 0;
}