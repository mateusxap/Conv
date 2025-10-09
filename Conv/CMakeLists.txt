cmake_minimum_required(VERSION 3.10)
project(Conv)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Conv
    ${CMAKE_CURRENT_SOURCE_DIR}/Conv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_include_directories(Conv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# If you want to use torch, add find_package(Torch REQUIRED) and link libraries
# find_package(Torch REQUIRED)
# target_link_libraries(Conv "${TORCH_LIBRARIES}")
# set_property(TARGET Conv PROPERTY CXX_STANDARD 14)

# For Python integration, you may need pybind11 or similar
# find_package(pybind11 REQUIRED)
# target_link_libraries(Conv PRIVATE pybind11::module)

# Optionally, set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -march=native")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Conv PRIVATE OpenMP::OpenMP_CXX)
endif()
