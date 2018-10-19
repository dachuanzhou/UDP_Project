find_package(OpenCV REQUIRED)
add_executable(2image lashenimage.cpp)
target_link_libraries(2image ${OpenCV_LIBS})