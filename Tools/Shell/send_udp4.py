#!/usr/bin/env python
# coding:utf-8
"""send_udp.py"""

# import argparse

import socket
import time
from time import clock

UDP_IP = "172.16.1.4"
UDP_PORT = 58000
MESSAGE = "0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"

print("UDP target IP:", UDP_IP)
print("UDP target port:", UDP_PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

start = time.monotonic()
for index in range(1, 614401, 1):
    test = 0
    for test in range(0, 38, 1):
        test += 1
    sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT))

finish = time.monotonic()
print("Duration ", (finish - start))
