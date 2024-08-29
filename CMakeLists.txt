project(gpu-framework-benchmark)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

cmake_minimum_required(VERSION 3.16.0)

add_subdirectory(MNN)

add_executable(benchmark main.cpp)
message("HELLO")
message("${MNN_DEPS}")
target_link_libraries(benchmark MNN MNN_Express MNN_Vulkan MNN_CL ${OpenCL_LIBRARIES})

include_directories(${CMAKE_CURRENT_LIST_DIR}/MNN/include/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/source/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/express/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/tools/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/codegen/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/schema/current/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/3rd_party/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/3rd_party/flatbuffers/include
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/3rd_party/half
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/3rd_party/imageHelper
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/3rd_party/OpenCLHeaders/
                  )
