cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(
    roml
    VERSION 0.0.1
    DESCRIPTION "A naive, non-optimized machine learning library for education."
    HOMEPAGE_URL "https://github.com/RohitAwate/roml/"
)

# set(LIB_SRC)

add_executable(main main.cpp)
# target_link_libraries(main romlib)
target_include_directories(main PUBLIC include)

# add_library(romlib SHARED ${LIB_SRC})
# target_include_directories(romlib PUBLIC include)