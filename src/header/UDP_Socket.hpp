
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

    char *recv_buffer;
    char *msg_buffer;
    void start_receive();
    void handle_receive(const boost::system::error_code &error, std::size_t /*bytes_transferred*/);
    void handle_send(boost::shared_ptr<std::string> /*message*/, const boost::system::error_code & /*error*/,
                     std::size_t /*bytes_transferred*/);

  public:
    UDP_Socket(boost::asio::io_service &io_context, int &msg_cnt_ctl, unsigned long long &bytes_cnt_ctl,
               std::string ip_address, int port_num, char *buffer);
};

UDP_Socket::UDP_Socket(boost::asio::io_service &io_context, int &msg_cnt_ctl, unsigned long long &bytes_cnt_ctl,
                       std::string ip_address, int port_num, char *buffer)
    : socket_(io_context, udp::endpoint(boost::asio::ip::address_v4::from_string(ip_address), port_num)) {
    ip_addr = ip_address;
    msg_cnt = &msg_cnt_ctl;
    bytes_cnt = &bytes_cnt_ctl;
    recv_buffer = buffer;
    msg_buffer = (char *)malloc(1024 * 1024 * 1024);
    start_receive();
}

void UDP_Socket::start_receive() {
    socket_.async_receive_from(boost::asio::buffer(msg_buffer, PACKET_LENGTH * PACKET_SUM_PER_INTERFACE),
                               remote_endpoint_,
                               boost::bind(&UDP_Socket::handle_receive, this, boost::asio::placeholders::error,
                                           boost::asio::placeholders::bytes_transferred));
}

void UDP_Socket::handle_receive(const boost::system::error_code &error, std::size_t bytes_transferred) {
    if (!error) {
        (*msg_cnt)++;
        (*bytes_cnt) += bytes_transferred;
        memcpy(recv_buffer, msg_buffer, bytes_transferred);
        start_receive();
    } else {
        printf("error\n");
    }
}

void UDP_Socket::handle_send(boost::shared_ptr<std::string> /*message*/, const boost::system::error_code & /*error*/,
                             std::size_t /*bytes_transferred*/) {}