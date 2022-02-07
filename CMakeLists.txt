cmake_minimum_required(VERSION 3.15...3.22)

project(DemoARM64 LANGUAGES C)

enable_testing()

include(CheckIncludeFile)
include(CheckSymbolExists)

# --- Detect CPU

message(STATUS "CPU ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR}")

check_include_file(arm_neon.h HAS_NEON_H)
if(NOT HAS_NEON_H)
  message(FATAL_ERROR "NEON not available, is host using an ARM CPU?")
endif()

check_symbol_exists(__aarch64__ arm_neon.h AARCH64)

# --- ARM options

if(NOT AARCH64)
  add_compile_options($<$<C_COMPILER_ID:GNU>:-mfpu=neon>)
endif()

# --- examples

add_executable(demo_neon demo_neon.c)
add_test(NAME NEON COMMAND demo_neon)

if(AARCH64)
  add_executable(matrix_neon matrix_neon.c)
  add_test(NAME matrix_NEON COMMAND matrix_neon)
endif()
