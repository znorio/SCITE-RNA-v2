cmake_minimum_required(VERSION 3.27)
project(SCITE-RNA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(src_cpp)

add_executable(SCITE-RNA
        src_cpp/cell_tree.cpp
        src_cpp/cell_tree.h
        src_cpp/mutation_tree.cpp
        src_cpp/mutation_tree.h
        src_cpp/swap_optimizer.cpp
        src_cpp/swap_optimizer.h
        src_cpp/mutation_filter.cpp
        src_cpp/mutation_filter.h
        src_cpp/config.cpp
        src_cpp/config.h
#        generate_results_cpp/comparison_num_clones.cpp
#        generate_results_cpp/run_sciterna.cpp
#        generate_results_cpp/MM.cpp
        generate_results_cpp/comparison_tree_spaces_switching.cpp

)
