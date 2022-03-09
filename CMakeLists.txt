cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(
    tensorslow
    VERSION 0.0.1
    DESCRIPTION "A naive, non-optimized machine learning library for education."
    HOMEPAGE_URL "https://github.com/RohitAwate/tensorslow/"
)

# set(LIB_SRC)

add_executable(main main.cpp)
add_executable(gd gd.cpp)

# target_link_libraries(main libts)
target_include_directories(main PUBLIC include)
target_include_directories(gd PUBLIC include)

# add_library(libts SHARED ${LIB_SRC})
# target_include_directories(libts PUBLIC include)