#!/usr/bin/env python
# coding:utf-8
"""draw txt to png by matplotlib"""

# import argparse
import logging
import numpy as np
from matplotlib import pyplot as plt

import argparse


def set_argparse():
    """Set the args&argv for command line mode"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "index", nargs="?", type=int, default=0, help="start index")
    parser.add_argument(
        "step", nargs="?", type=int, default=1, help="step offset")
    return parser.parse_args()


def get_logger(logname: str):
    """Config the logger in the module
    Arguments:
        logname {str} -- logger name
    Returns:
        logging.Logger -- the logger object
    """
    logger = logging.getLogger(logname)
    formater = logging.Formatter(
        fmt='%(asctime)s - %(filename)s : %(levelname)-5s :: %(message)s',
        # filename='./log.log',
        # filemode='a',
        datefmt='%m/%d/%Y %H:%M:%S')
    stream_hdlr = logging.StreamHandler()
    stream_hdlr.setFormatter(formater)
    logger.addHandler(stream_hdlr)
    logger.setLevel(logging.DEBUG)
    return logger


__logger__ = get_logger('draw.py')


def draw_txt_to_png(file_index):
    file_name = "Tri_%04d.txt" % (file_index)
    # x_values = []
    # y_values = []
    # f = open(file_name, 'r')
    # lines = f.readlines()
    # for line in lines:
    #     values = line.split(',')
    #     x_values.append(int(values[0]))
    #     y_values.append(int(values[1]))
    __logger__.info(file_name)
    # f.close()
    data = np.loadtxt(file_name, delimiter=',')
    plt.figure(figsize=(8, 8))
    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.grid(True)
    plt.axis([0, 2100, 0, 2100])
    plt.xlabel('X_ID')
    plt.ylabel('Y_ID')
    plt.title(file_index)
    # plt.show()
    plt.savefig("Png_%04d" % (file_index))
    plt.close()


def main():
    """Main function"""
    __logger__.info('Process start!')
    for file_index in np.arange(__START_INDEX__, 2048, __STEP_OFFSET__):
        draw_txt_to_png(file_index)
    __logger__.info('Process end!')


if __name__ == '__main__':
    # Uncomment the next line to read args from cmd-line
    ARGS = set_argparse()
    __START_INDEX__ = ARGS.index
    __STEP_OFFSET__ = ARGS.step
    main()
