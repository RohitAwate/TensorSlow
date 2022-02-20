cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(
    roml
    VERSION 0.0.1
    DESCRIPTION "A naive, non-optimized machine learning library for education."
    HOMEPAGE_URL "https://github.com/RohitAwate/roml/"
)

set(SRC
    src/roml/math/vector.cpp
    src/roml/math/matrix.cpp
)

add_library(romlib SHARED ${SRC})
target_include_directories(romlib PUBLIC include)