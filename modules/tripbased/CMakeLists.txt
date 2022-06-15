cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-tripbased-files src/*.cc)

add_library(motis-tripbased STATIC ${motis-tripbased-files})
target_include_directories(motis-tripbased PUBLIC include)
target_link_libraries(motis-tripbased
  boost-system
  motis-routing
  motis-module
  motis-core
  geo
  date
)
target_compile_features(motis-tripbased PUBLIC cxx_std_17)
target_compile_options(motis-tripbased PRIVATE ${MOTIS_CXX_FLAGS})

if (MOTIS_CUDA)
    include(FindCUDAToolkit)

    file(GLOB_RECURSE gpu-tripbased-files src/gpu/*.cu)
    add_library(gpu-tripbased SHARED ${gpu-tripbased-files})

    set_target_properties(gpu-tripbased PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 17
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/"
            )
    target_compile_definitions(gpu-tripbased PRIVATE MOTIS_CUDA)
    target_link_libraries(gpu-tripbased utl motis-core)
    target_include_directories(gpu-tripbased PRIVATE include)
    set_property(TARGET gpu-tripbased PROPERTY CUDA_ARCHITECTURES 75 61)
    target_compile_options(gpu-tripbased PRIVATE "--expt-relaxed-constexpr")
    target_link_libraries(motis-tripbased gpu-tripbased CUDA::cudart_static)
endif ()