cmake_minimum_required(VERSION 3.10)
project(Garnet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set debug mode
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)

# set optimization level
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(/Library/gurobi1002/macos_universal2/include)
include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3)

add_executable(Garnet main.cpp garnet_sa.cpp garnetS.cpp garnet.h)
link_directories(/Library/gurobi1002/macos_universal2/lib)
target_link_libraries(Garnet 
                    /Library/gurobi1002/macos_universal2/lib/libgurobi_c++.a
                    /Library/gurobi1002/macos_universal2/lib/libgurobi100.dylib)