cmake_minimum_required(VERSION 3.16)
project(flash_kda_npu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)
find_package(ASC REQUIRED)

set(CMAKE_ASC_ARCHITECTURES "dav-c90" CACHE STRING "Ascend NPU architecture")

add_library(flash_kda_npu_kernels SHARED
    kda_host.cpp
    kda_k1.cpp
    kda_k2.cpp
)


target_compile_definitions(flash_kda_npu_kernels PRIVATE KDA_CHUNK=64)
target_include_directories(flash_kda_npu_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(flash_kda_npu_kernels PRIVATE ${TORCH_LIBRARIES})
set_target_properties(flash_kda_npu_kernels PROPERTIES
    OUTPUT_NAME flash_kda_npu_kernels
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

install(TARGETS flash_kda_npu_kernels
    LIBRARY DESTINATION flash_kda_npu/cpp_extension_asc/lib
    RUNTIME DESTINATION flash_kda_npu/cpp_extension_asc/lib
)
