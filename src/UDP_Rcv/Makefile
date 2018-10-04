CXX	= clang++
SXXstd = -std=c++11
PERARGS = -Wall -O3
CFLAGS = -lboost_system -lpthread

.PHONY:all clean

all:
	@echo "----- make all in UDP_Rcv -----"
	@echo "     ----> make udp_receive.cpp"
	${CXX} ${SXXstd} ${PERARGS} udp_receive.cpp -o ../../release/udp_receive_cpp ${CFLAGS}
	${CXX} ${SXXstd} ${PERARGS} Pcap2Bin.cpp -o ../../release/Pcap2Bin_cpp ${CFLAGS}

clean:

	@echo "----- make clean in UDP_Rcv -----"
