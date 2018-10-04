CXX	= clang++
SXXstd = -std=c++11
PERARGS = -Wall -O3
CFLAGS = -lboost_system -lpthread
OPENCVFLAGS = `pkg-config --cflags --libs opencv`

.PHONY:all clean

all:
	@echo "----- make all in UDP_Rcv -----"
	@echo "     ----> make udp_receive.cpp"
	clang++ ${OPENCVFLAGS} ${SXXstd} ${PERARGS} -lpthread lashenimage.cpp -o ../../release/2image_cpp

clean:

	@echo "----- make clean in UDP_Rcv -----"
