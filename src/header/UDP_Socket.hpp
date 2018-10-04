
#include <iostream>
#include <string>

#include <boost/asio.hpp>
#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include "define.hpp"

using boost::asio::ip::udp;

class UDP_Socket {
  private:
    udp::socket socket_;
    udp::endpoint remote_endpoint_;
    int *msg_cnt;
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
    UDP_Socket(boost::asio::io_service &io_context, int &msg_cnt_ctl, unsigned long long &bytes_cnt_ctl,
               std::string ip_address, int port_num, char *data_buffer, char *check_butter);
    ~UDP_Socket();
};

UDP_Socket::UDP_Socket(boost::asio::io_service &io_context, int &msg_cnt_ctl, unsigned long long &bytes_cnt_ctl,
                       std::string ip_address, int port_num, char *data_buffer, char *check_butter)
    : socket_(io_context, udp::endpoint(boost::asio::ip::address_v4::from_string(ip_address), port_num)) {
    ip_addr = ip_address;
    msg_cnt = &msg_cnt_ctl;
    bytes_cnt = &bytes_cnt_ctl;
    data_content = data_buffer;
    check_content = check_butter;
    msg_buffer = (char *)malloc(1512 * sizeof(char));
    start_receive();
}

UDP_Socket::~UDP_Socket() { free(msg_buffer); }

void UDP_Socket::start_receive() {
    socket_.async_receive_from(boost::asio::buffer(msg_buffer, PACKET_LENGTH * PACKET_SUM_PER_INTERFACE),
                               remote_endpoint_,
                               boost::bind(&UDP_Socket::handle_receive, this, boost::asio::placeholders::error,
                                           boost::asio::placeholders::bytes_transferred));
}

void UDP_Socket::handle_receive(const boost::system::error_code &error, std::size_t bytes_transferred) {
    if (!error) {
        (*bytes_cnt) += bytes_transferred;
        // ! 按照协议保存数据
        // memcpy(data_content + *msg_cnt * 1410l, msg_buffer, VALID_BYTES_LENGTH_PER_PACKAGE);
        memcpy(data_content + *msg_cnt * 1400l, msg_buffer, VALID_BYTES_LENGTH_PER_PACKAGE);
        memcpy(check_content + *msg_cnt * 5, msg_buffer + 1, 2);
        memcpy(check_content + *msg_cnt * 5, msg_buffer + 1405, 3);

        // printf("rcv: %luBytes.\n", bytes_transferred);
        (*msg_cnt)++;
        start_receive();
    } else {
        printf("error\n");
    }
}

void UDP_Socket::handle_send(boost::shared_ptr<std::string> /*message*/, const boost::system::error_code & /*error*/,
                             std::size_t /*bytes_transferred*/) {}