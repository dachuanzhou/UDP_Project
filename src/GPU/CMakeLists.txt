cmake_minimum_required(VERSION 3.0)
project(lab5)
find_package(CUDA)
cuda_add_executable(multicalc multicalc.cu)