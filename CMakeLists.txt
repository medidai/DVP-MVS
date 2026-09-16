cmake_minimum_required (VERSION 3.18)
project (APD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# Default to the L4 (sm_89); override with -DCMAKE_CUDA_ARCHITECTURES=... for
# other GPUs. Without matching SASS the kernels silently produce no depth.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
${OpenCV_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS})
include_directories(.)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --maxrregcount=128")

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native")
endif()

add_executable(
    APD
    main.h
    APD.h
    APD.cpp
    APD.cu
    main.cpp
)

target_link_libraries(APD
${Boost_LIBRARIES}
"${OpenCV_LIBS}"
CUDA::cudart
CUDA::curand
OpenMP::OpenMP_CXX
)
