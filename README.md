# GitHub Actions aarch64

Example of ARM64 image with [GitHub Actions](https://github.com/uraimo/run-on-arch-action) to setup QEMU enviornment, cache, install packages, and run tests.

We use CMake to build a C program using aarch64 NEON instructions, that would fail on default GitHub Actions x86 arch.
