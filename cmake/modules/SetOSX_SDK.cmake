# The SDKROOT environment variable and/or CMAKE_OSX_SYSROOT CMake variable need to be set *before* any project(ROOT) call.
# This is necessary to find (and for cling to remember) the SDK, so don't remove this unless tested with all combinations of
# incremental builds, fresh builds, having SDKROOT set, having CMAKE_OSX_SYSROOT set, and also not having them set.
# The CMake 4 standard of leaving it to the compiler to figure out the SDK doesn't work for Cling.
# See https://github.com/root-project/root/pull/19718, https://github.com/root-project/root/pull/19855 for failed attempts to remove this.

# To choose an SDK, one has the following options:
# - Let xcrun choose the latest installed SDK: "cmake ..."
# - Configure an SDK using "SDKROOT=<path> cmake ..."
# - Set a cache variable: "cmake -DCMAKE_OSX_SYSROOT=<path> ..."

if(NOT IS_DIRECTORY "${CMAKE_OSX_SYSROOT}")
  unset(CMAKE_OSX_SYSROOT CACHE)
  unset(CMAKE_OSX_SYSROOT)
endif()

find_program(XCRUN_EXECUTABLE xcrun)
if(EXISTS ${XCRUN_EXECUTABLE})
  if(NOT DEFINED "${CMAKE_OSX_SYSROOT}")
    execute_process(COMMAND ${XCRUN_EXECUTABLE} --sdk macosx --show-sdk-path
      OUTPUT_VARIABLE SDK_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT IS_DIRECTORY "${SDK_PATH}")
      message(FATAL_ERROR "Could not detect macOS SDK path")
    endif()
    set(CMAKE_OSX_SYSROOT "${SDK_PATH}" CACHE PATH "MacOS SDK path" FORCE)
  endif()

  # Save the SDK version for ROOT to inspect it later. This is needed for LTS that live longer
  # then the SDK they were created with.
  execute_process(COMMAND ${XCRUN_EXECUTABLE} --sdk ${CMAKE_OSX_SYSROOT} --show-sdk-version
    OUTPUT_VARIABLE OSX_SDK_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  mark_as_advanced(OSX_SDK_VERSION)
endif()

if(NOT IS_DIRECTORY "${CMAKE_OSX_SYSROOT}")
  message(FATAL_ERROR "ROOT needs a path to an SDK to compile on Mac. Try setting CMAKE_OSX_SYSROOT or provide the xcrun executable.")
endif()
message(STATUS "Mac OS SDK version: '${OSX_SDK_VERSION}' ${CMAKE_OSX_SYSROOT}")
