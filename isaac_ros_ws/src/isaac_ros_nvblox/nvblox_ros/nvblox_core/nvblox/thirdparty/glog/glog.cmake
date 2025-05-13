if (USE_SYSTEM_GLOG)
  find_package(glog REQUIRED)
else()
  # Disable printing of stack trace for now. Note(dtingdahl): Might have something to do with unwind
  # library not found.
  set(PC_FROM_UCONTEXT false)

  # Make sure we're building a static lib
  option(BUILD_SHARED_LIBS "" OFF)

  # Disable testing and store the old state of the BUILD_TESTING option
  set(BUILD_TESTING_OLD ${BUILD_TESTING})
  set(BUILD_TESTING OFF)

  # Disable gflags to avoid system libs being picked up by find_package() inside glog's
  # CMakeLists.txt. TODO(dtingdahl): From cmake>=3.24 it is possible to use the OVERRIDE_FIND_PACKAGE
  # option to FetchContent_Declare() which make find_package() point use the sources specified by
  # FetchContent instead. Let's use that mechanism once we support a more recent version of cmake.
  option(WITH_GFLAGS "USE gflags" OFF)

  include(FetchContent)
  FetchContent_Declare(
    ext_glog
    OVERRIDE_FIND_PACKAGE
    SYSTEM
    URL https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz
    URL_HASH MD5=2368e3e0a95cce8b5b35a133271b480f
  )

  # Download the files
  FetchContent_MakeAvailable(ext_glog)

  # Apply nvblox compile options to exported targets
  set_nvblox_compiler_options_nowarnings(glog)

  # Restore state of the BUILD_TESTING flag
  set(BUILD_TESTING ${BUILD_TESTING_OLD})
endif()
