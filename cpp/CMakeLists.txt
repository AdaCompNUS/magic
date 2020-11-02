cmake_minimum_required(VERSION 3.14)
project(MacroNav)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "Require at least gcc-7.0")
endif()
SET(CMAKE_CXX_FLAGS "-fPIC -O3 -Wfatal-errors -Werror -Wall -Wextra -Wdeprecated -Wno-shadow -Wuninitialized -Wunreachable-code -Wold-style-cast -Wconversion -Wsign-conversion -Wno-unused-parameter -Wno-type-limits -Wunused-function")

if(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
endif()
SET(CMAKE_EXPORT_COMPILE_COMMANDS True)
#if(NOT CMAKE_PREFIX_PATH)
#  SET(CMAKE_PREFIX_PATH $ENV{HOME}/.local)

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem system program_options)

# Add macaron dependency.
file(GLOB MACARON_SOURCES dependencies/macaron/src/macaron/*.cpp)
add_library(macaron SHARED ${MACARON_SOURCES})
target_include_directories(macaron SYSTEM PUBLIC dependencies/macaron/include)

# Add despot dependency.
add_subdirectory(dependencies/despot)

# Add RVO2 dependency.
file(GLOB RVO2_SOURCES dependencies/RVO2/src/rvo2/*.cpp)
add_library(RVO2 SHARED ${RVO2_SOURCES})
target_include_directories(RVO2 SYSTEM PUBLIC dependencies/RVO2/include)

# Create macro-nav library.
file(GLOB SOURCES
  src/core/*.cpp
  src/core/random/*.cpp
  src/core/simulations/*.cpp)
add_library(macro_nav STATIC ${SOURCES})
target_include_directories(macro_nav PUBLIC include)
target_link_libraries(macro_nav ${OpenCV_LIBS} ${Boost_LIBRARIES} macaron despot RVO2)

message("Adding gym environments:")
foreach(TASK
    CornerNav
    LightDark
    IntentionTag
    PuckPush
    )
  message("  GymEnv${TASK}")
  add_executable("GymEnv${TASK}" src/experiments/GymEnv.cpp)
  target_compile_definitions("GymEnv${TASK}" PRIVATE -D"SIM_${TASK}")
  target_link_libraries("GymEnv${TASK}" macro_nav)
endforeach()

message("Adding POMCPOW environments")
foreach(TASK
    CornerNav
    LightDark
    IntentionTag
    PuckPush
    VdpTag
    )
  message("  PomcpowEnv${TASK}")
  add_executable("PomcpowEnv${TASK}" src/experiments/PomcpowEnv.cpp)
  target_compile_definitions("PomcpowEnv${TASK}" PRIVATE -D"SIM_${TASK}")
  target_link_libraries("PomcpowEnv${TASK}" macro_nav)
endforeach()

message("Adding DESPOT (Handcrafted) environments:")
foreach(TASK
    CornerNav
    LightDark
    IntentionTag
    PuckPush
    )
  message("  DespotHandcraftedEnv${TASK}")
  add_executable("DespotHandcraftedEnv${TASK}" src/experiments/DespotHandcraftedEnv.cpp)
  target_compile_definitions("DespotHandcraftedEnv${TASK}" PRIVATE -D"SIM_${TASK}")
  target_link_libraries("DespotHandcraftedEnv${TASK}" macro_nav)
endforeach()

message("Adding DESPOT (MAGIC) environments:")
foreach(TASK
    CornerNav
    LightDark
    IntentionTag
    PuckPush
    )
  message("  DespotMagicEnv${TASK}")
  add_executable("DespotMagicEnv${TASK}" src/experiments/DespotMagicEnv.cpp)
  target_compile_definitions("DespotMagicEnv${TASK}" PRIVATE -D"SIM_${TASK}")
  target_link_libraries("DespotMagicEnv${TASK}" macro_nav)
endforeach()

message("Adding environment tests:")
foreach(TASK
    CornerNav
    LightDark
    IntentionTag
    PuckPush
    VdpTag
    )
  message("  TaskTest${TASK}")
  add_executable("TaskTest${TASK}" src/experiments/TaskTest.cpp)
  target_compile_definitions("TaskTest${TASK}" PRIVATE -D"SIM_${TASK}")
  target_link_libraries("TaskTest${TASK}" macro_nav)
endforeach()

message("Adding real world targets tests:")

message("  RealWorldPuckPushDespot")
add_executable("RealWorldPuckPushDespot" src/experiments/RealWorldPuckPushDespot.cpp)
target_compile_definitions("RealWorldPuckPushDespot" PRIVATE -D"SIM_PuckPush")
target_link_libraries("RealWorldPuckPushDespot" macro_nav)

message("  RealWorldPuckPushPomcpow")
add_executable("RealWorldPuckPushPomcpow" src/experiments/RealWorldPuckPushPomcpow.cpp)
target_compile_definitions("RealWorldPuckPushPomcpow" PRIVATE -D"SIM_PuckPush")
target_link_libraries("RealWorldPuckPushPomcpow" macro_nav)
