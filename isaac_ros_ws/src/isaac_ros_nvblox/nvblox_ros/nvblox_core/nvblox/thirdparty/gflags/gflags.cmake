if (USE_SYSTEM_GFLAGS)
  find_package(gflags REQUIRED)
else()

include(FetchContent)
  FetchContent_Declare(
    ext_gflags
    SYSTEM
    OVERRIDE_FIND_PACKAGE
    URL https://github.com/gflags/gflags/archive/refs/tags/v2.2.0.tar.gz
    URL_HASH MD5=b99048d9ab82d8c56e876fb1456c285e
  )

  FetchContent_MakeAvailable(ext_gflags)

  # Apply nvblox compile options to exported targets
  set_nvblox_compiler_options_nowarnings(gflags_nothreads_static)
endif()
