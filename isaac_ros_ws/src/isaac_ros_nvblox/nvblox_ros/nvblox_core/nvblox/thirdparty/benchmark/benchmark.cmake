if (USE_SYSTEM_BENCHMARK)
  find_package(benchmark REQUIRED)
else()

include(FetchContent)

# Disable failure on compiler warnings that are out of our control.
option(BENCHMARK_ENABLE_WERROR OFF)
option(BENCHMARK_ENABLE_TESTING OFF)

FetchContent_Declare(
  ext_benchmark
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.0.tar.gz
  URL_HASH MD5=21a2604efeded8b4cbabc72f3e1c7a2a
)
FetchContent_MakeAvailable(ext_benchmark)

# Apply nvblox compile options to exported targets
set_nvblox_compiler_options_nowarnings(benchmark)
endif()
