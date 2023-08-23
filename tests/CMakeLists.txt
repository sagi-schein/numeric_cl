cmake_minimum_required(VERSION 3.22)
add_executable (test_mat_cl main.cpp)
link_directories(${CMAKE_SOURCE_DIR}/matrix/matrix)
target_link_libraries (test_mat_cl matrix_cl OpenCL)