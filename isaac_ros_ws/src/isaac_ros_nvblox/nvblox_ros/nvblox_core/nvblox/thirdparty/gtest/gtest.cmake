if (USE_SYSTEM_GTEST)
  find_package(GTest REQUIRED)
else()
include(FetchContent)

FetchContent_Declare(
  ext_gtest
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  URL_HASH MD5=7e11f6cfcf6498324ac82d567dcb891e
)
FetchContent_MakeAvailable(ext_gtest)

# Apply nvblox compile options to exported targets
set_nvblox_compiler_options_nowarnings(gtest)
endif()
