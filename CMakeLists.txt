cmake_minimum_required(VERSION 3.15...3.22)

project(DemoARM64 LANGUAGES C)

enable_testing()

include(CheckIncludeFile)

message(STATUS "CPU ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR}")

check_include_file(arm_neon.h HAS_NEON_H)

if(NOT HAS_NEON_H)
  message(FATAL_ERROR "NEON not available, is host using an ARM CPU?")
endif()

add_executable(demo_neon demo_neon.c)
target_compile_options(demo_neon PRIVATE $<$<C_COMPILER_ID:GNU>:-mfpu=neon>)

add_test(NAME NEON COMMAND demo_neon)
