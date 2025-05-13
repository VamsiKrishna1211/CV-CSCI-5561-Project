if (USE_SYSTEM_SQLITE3)
  find_package(sqlite3 REQUIRED)
else()
include(FetchContent)

FetchContent_Declare(
  ext_sqlite3
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  URL https://github.com/alex85k/sqlite3-cmake/archive/refs/tags/v3.24.0.tar.gz
  URL_HASH MD5=15ab81bf8cfbdb9a9a3b3abd38c7598b
)

FetchContent_MakeAvailable(ext_sqlite3)
target_include_directories(sqlite3 PUBLIC ${sqlite3_SOURCE_DIR}/src)

# Apply nvblox compile options to exported targets
set_nvblox_compiler_options_nowarnings(sqlite3)
endif()
