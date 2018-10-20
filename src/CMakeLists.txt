cmake_minimum_required(VERSION 3.5)
add_subdirectory(GPU)
add_subdirectory(txt2png)
enable_testing()
set(data_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../data/)
add_test(
  NAME Gen_sample_txt
  COMMAND multicalc 
    16
    ${data_dir}/b_64.filter 
    ${data_dir}/decode_data.bin 
    ${CMAKE_CURRENT_BINARY_DIR}/sample.txt
)

add_test(
  NAME Gen_sample_graph
  COMMAND 2image
    ${CMAKE_CURRENT_BINARY_DIR}/sample.txt
)

set_tests_properties(Gen_sample_graph PROPERTIES FIXTURES_REQUIRED Gen_sample_txt)
