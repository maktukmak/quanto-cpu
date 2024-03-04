cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(kernel-dev)
set(CMAKE_CXX_STANDARD 17)

# COMPILE
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
add_compile_options(-mfma -mavx512f -mavx512bf16 -mavx512vl)

# LIBRARIES
set(LIBTORCH /media/MERCURY/mehmet/anaconda3/envs/hf_quanto_cpu/lib/python3.10/site-packages/torch/share/cmake/Torch)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${LIBTORCH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


find_package(BLAS REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BLAS_CXX_FLAGS}")

find_package(OpenMP REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(PythonLibs 3 REQUIRED)

find_package(MKL REQUIRED) # PATHS $ENV{MKLROOT})

find_package(IntelSYCL REQUIRED)

cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PARENT_DIR)

set(KERNELS_INCLUDE_DIR ${PARENT_DIR}/quanto/library/cpu)
add_subdirectory(${KERNELS_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/KERNELS)

set(SAMPLES_INCLUDE_DIR ${PARENT_DIR}/oneAPI-samples/Libraries/oneMKL/matrix_mul_mkl)
add_subdirectory(${SAMPLES_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/SAMPLES)


# TARGETS
add_executable(test_blass_int8 test_blass_int8.cpp)
target_compile_options(test_blass_int8 INTERFACE ${CMAKE_CXX_FLAGS})
target_link_libraries(test_blass_int8 PUBLIC MKL::MKL)

add_executable(test_torch_int8_mm test_torch_int8_mm.cpp)
target_compile_options(test_torch_int8_mm INTERFACE ${CMAKE_CXX_FLAGS})
target_include_directories(test_torch_int8_mm PUBLIC ${PYTHON_INCLUDE_DIRS} ${KERNELS_INCLUDE_DIR})
target_link_libraries(test_torch_int8_mm PUBLIC ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES} MKL::MKL)


add_executable(matrix_mul_mkl ${SAMPLES_INCLUDE_DIR}/matrix_mul_mkl.cpp)
target_compile_options(matrix_mul_mkl INTERFACE ${CMAKE_CXX_FLAGS})
target_include_directories(matrix_mul_mkl PUBLIC ${SAMPLES_INCLUDE_DIR})
target_link_libraries(matrix_mul_mkl PUBLIC MKL::MKL)