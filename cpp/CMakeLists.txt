cmake_minimum_required(VERSION 3.14)
project(MacroNav)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "Require at least gcc-7.0")
endif()
SET(CMAKE_CXX_FLAGS "-fPIC -O3 -msse2 -mfpmath=sse -Wfatal-errors -Werror -Wall -Wextra -Wdeprecated -Wno-shadow -Wuninitialized -Wunreachable-code -Wold-style-cast -Wconversion -Wsign-conversion -Wno-unused-parameter -Wno-type-limits -Wunused-function -Wno-stringop-overflow -Wno-maybe-uninitialized -Wdouble-promotion")

if(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
endif()
SET(CMAKE_EXPORT_COMPILE_COMMANDS True)
if(NOT CMAKE_PREFIX_PATH)
  SET(CMAKE_PREFIX_PATH $ENV{HOME}/.local)
endif()

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem system program_options)

find_package(Threads REQUIRED)

find_package(OpenMP REQUIRED)

# Add macaron dependency.
file(GLOB MACARON_SOURCES dependencies/macaron/src/macaron/*.cpp)
add_library(macaron STATIC ${MACARON_SOURCES})
target_include_directories(macaron SYSTEM PUBLIC dependencies/macaron/include)

# Add despot dependency.
add_subdirectory(dependencies/despot)

# Add RVO2 dependency.
file(GLOB RVO2_SOURCES dependencies/RVO2/src/rvo2/*.cpp)
add_library(RVO2 STATIC ${RVO2_SOURCES})
target_include_directories(RVO2 SYSTEM PUBLIC dependencies/RVO2/include)

# Add summit dependency
add_subdirectory(dependencies/carla)

# Create macro-nav library.
file(GLOB SOURCES
  src/core/*.cpp
  src/core/random/*.cpp
  src/core/simulations/*.cpp)
add_library(macro_nav STATIC ${SOURCES})
target_include_directories(macro_nav PUBLIC include)
target_link_libraries(macro_nav ${OpenCV_LIBS} ${Boost_LIBRARIES} Threads::Threads OpenMP::OpenMP_CXX macaron despot RVO2 carla)

message("Adding POMCPOW environments")
foreach(TASK
    LightDark
    PuckPush
    VdpTag
    )
  message("  PomcpowEnv${TASK}")
  add_executable("PomcpowEnv${TASK}" src/experiments/PomcpowEnv.cpp)
  target_compile_definitions("PomcpowEnv${TASK}" PRIVATE -D"SIM_${TASK}")
  target_link_libraries("PomcpowEnv${TASK}" macro_nav)
endforeach()
# Add DriveHard task.
message("  PomcpowEnvDriveHard")
add_executable("PomcpowEnvDriveHard" src/experiments/PomcpowEnvDriveHard.cpp)
target_compile_definitions("PomcpowEnvDriveHard" PRIVATE -D"SIM_DriveHard")
target_link_libraries("PomcpowEnvDriveHard" macro_nav)

message("Adding DESPOT (Handcrafted) environments:")
# Adds standard tasks.
foreach(TASK
    LightDark
    PuckPush
    )
  message("  DespotHandcraftedEnv${TASK}")
  add_executable("DespotHandcraftedEnv${TASK}" src/experiments/DespotHandcraftedEnv.cpp)
  target_compile_definitions("DespotHandcraftedEnv${TASK}" PRIVATE -D"SIM_${TASK}")
  target_link_libraries("DespotHandcraftedEnv${TASK}" macro_nav)
endforeach()
# Add DriveHard task.
message("  DespotHandcraftedEnvDriveHard")
add_executable("DespotHandcraftedEnvDriveHard" src/experiments/DespotHandcraftedEnvDriveHard.cpp)
target_compile_definitions("DespotHandcraftedEnvDriveHard" PRIVATE -D"SIM_DriveHard")
target_link_libraries("DespotHandcraftedEnvDriveHard" macro_nav)


message("Adding DESPOT (MAGIC) environments:")
# Adds standard tasks.
foreach(TASK
    LightDark
    PuckPush
    )
  message("  DespotMagicEnv${TASK}")
  add_executable("DespotMagicEnv${TASK}" src/experiments/DespotMagicEnv.cpp)
  target_compile_definitions("DespotMagicEnv${TASK}" PRIVATE -D"SIM_${TASK}")
  target_link_libraries("DespotMagicEnv${TASK}" macro_nav)
endforeach()
# Add DriveHard task.
message("  DespotMagicEnvDriveHard")
add_executable("DespotMagicEnvDriveHard" src/experiments/DespotMagicEnvDriveHard.cpp)
target_compile_definitions("DespotMagicEnvDriveHard" PRIVATE -D"SIM_DriveHard")
target_link_libraries("DespotMagicEnvDriveHard" macro_nav)

message("Adding environment tests:")
foreach(TASK
    LightDark
    PuckPush
    )
  message("  TaskTest${TASK}")
  add_executable("TaskTest${TASK}" src/experiments/TaskTest.cpp)
  target_compile_definitions("TaskTest${TASK}" PRIVATE -D"SIM_${TASK}")
  target_link_libraries("TaskTest${TASK}" macro_nav)
endforeach()
message("  TaskTestDriveHard")
add_executable("TaskTestDriveHard" src/experiments/TaskTestDriveHard.cpp)
target_compile_definitions("TaskTestDriveHard" PRIVATE -D"SIM_DriveHard")
target_link_libraries("TaskTestDriveHard" macro_nav)


# message("Adding real world targets tests:")
#
# message("  RealWorldPuckPushDespot")
# add_executable("RealWorldPuckPushDespot" src/experiments/RealWorldPuckPushDespot.cpp)
# target_compile_definitions("RealWorldPuckPushDespot" PRIVATE -D"SIM_PuckPush")
# target_link_libraries("RealWorldPuckPushDespot" macro_nav)
#
# message("  RealWorldPuckPushPomcpow")
# add_executable("RealWorldPuckPushPomcpow" src/experiments/RealWorldPuckPushPomcpow.cpp)
# target_compile_definitions("RealWorldPuckPushPomcpow" PRIVATE -D"SIM_PuckPush")
# target_link_libraries("RealWorldPuckPushPomcpow" macro_nav)

message("Adding summit targets")
message("  SummitEnvDriveHard")
add_executable("SummitEnvDriveHard" src/experiments/SummitEnvDriveHard.cpp)
target_compile_definitions("SummitEnvDriveHard" PRIVATE -D"SIM_DriveHard")
target_link_libraries("SummitEnvDriveHard" macro_nav)
