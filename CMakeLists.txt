cmake_minimum_required(VERSION 3.27)
project(SCITE-RNA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(src)

add_executable(SCITE-RNA
        src/cell_tree.cpp
        src/cell_tree.h
        src/mutation_tree.cpp
        src/mutation_tree.h
        src/swap_optimizer.cpp
        src/swap_optimizer.h
        src/mutation_filter.cpp
        src/mutation_filter.h
#        tests/comparison_num_clones.cpp
#        tests/comparison_tree_spaces_switching.cpp
        tests/MM.cpp
)
