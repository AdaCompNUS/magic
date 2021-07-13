cmake_minimum_required(VERSION 2.8.3)
project(carla)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-fPIC -O3 -msse2 -mfpmath=sse -Wno-maybe-uninitialized -Wdouble-promotion")

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

file(GLOB_RECURSE source_list "src/carla/*.cpp")
add_library("${PROJECT_NAME}" STATIC ${source_list})

target_include_directories("${PROJECT_NAME}" SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS ${PROJECT_NAME}
    EXPORT carlaTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY "include/carla/"
    DESTINATION "${INCLUDE_INSTALL_PATH}/carla"
)

install(EXPORT carlaTargets
    FILE carlaTargets.cmake
    NAMESPACE carla::
    DESTINATION lib/cmake
)
