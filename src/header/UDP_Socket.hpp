// -----------------------------------------------------------------------------
// Filename:    UDP_Socket.hpp
// Revision:    None
// Date:        2018/10/11 - 04:24
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// UDP socket 类封装，监听指定的 IP 和 port，并将数据拷贝到指定的内存空间里。

#include <iostream>
#include <string>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include "define.hpp"

using boost::asio::ip::udp;

class UDP_Socket {
  private:
    udp::socket socket_;            // 私有的 socket 变量
    udp::endpoint remote_endpoint_; // 发送源的 IP
    int *packets_cnt;                   // 
    unsigned long long *bytes_cnt;
    std::string ip_addr;

    char *data_content;
    char *check_content;
    char *msg_buffer;
    void start_receive();
    void handle_receive(const boost::system::error_code &error, std::size_t /*bytes_transferred*/);
    void handle_send(boost::shared_ptr<std::string> /*message*/, const boost::system::error_code & /*error*/,
                     std::size_t /*bytes_transferred*/);

  public:
    UDP_Socket(boost::asio::io_service &io_context, int &packets_cnt_ctl, unsigned long long &bytes_cnt_ctl,
               std::string ip_address, int port_num, char *data_buffer, char *check_butter);
    ~UDP_Socket();
};

UDP_Socket::UDP_Socket(boost::asio::io_service &io_context, int &packets_cnt_ctl, unsigned long long &bytes_cnt_ctl,
                       std::string ip_address, int port_num, char *data_buffer, char *check_butter)
    : socket_(io_context, udp::endpoint(boost::asio::ip::address_v4::from_string(ip_address), port_num)) {
    ip_addr = ip_address;
    packets_cnt = &packets_cnt_ctl;
    bytes_cnt = &bytes_cnt_ctl;
    data_content = data_buffer;
    check_content = check_butter;
    msg_buffer = (char *)malloc(1512 * sizeof(char));
    start_receive();
}

UDP_Socket::~UDP_Socket() { free(msg_buffer); }

void UDP_Socket::start_receive() {
    socket_.async_receive_from(boost::asio::buffer(msg_buffer, (size_t)1512),
                               remote_endpoint_,
                               boost::bind(&UDP_Socket::handle_receive, this, boost::asio::placeholders::error,
                                           boost::asio::placeholders::bytes_transferred));
}

void UDP_Socket::handle_receive(const boost::system::error_code &error, std::size_t bytes_transferred) {
    if (!error) {
        (*bytes_cnt) += bytes_transferred;
        // ! 保存所有数据
        // memcpy(data_content + *packets_cnt * 1410l, msg_buffer, VALID_BYTES_LENGTH_PER_PACKAGE);
        // ! 按照协议保存数据
        memcpy(data_content + *packets_cnt * 1400l, msg_buffer, VALID_BYTES_LENGTH_PER_PACKAGE);
        memcpy(check_content + *packets_cnt * 5, msg_buffer + 1, 2);
        memcpy(check_content + *packets_cnt * 5, msg_buffer + 1405, 3);

        // printf("rcv: %luBytes.\n", bytes_transferred);
        (*packets_cnt)++;
        start_receive();
    } else {
        printf("error\n");
    }
}

void UDP_Socket::handle_send(boost::shared_ptr<std::string> /*message*/, const boost::system::error_code & /*error*/,
                             std::size_t /*bytes_transferred*/) {}