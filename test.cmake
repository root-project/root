cmake_minimum_required(VERSION 3.16)

set(ENV{LANG} "C")
set(ENV{LC_ALL} "C")
# set(CTEST_USE_LAUNCHERS TRUE)

if(EXISTS "/etc/os-release")
  file(STRINGS "/etc/os-release" OS_NAME REGEX "^ID=.*$")
  string(REGEX REPLACE "ID=[\"']?([^\"']*)[\"']?$" "\\1" OS_NAME "${OS_NAME}")
  file(STRINGS "/etc/os-release" OS_VERSION REGEX "^VERSION_ID=.*$")
  string(REGEX REPLACE "VERSION_ID=[\"']?([^\"'.]*).*$" "\\1" OS_VERSION "${OS_VERSION}")
  file(STRINGS "/etc/os-release" OS_FULL_NAME REGEX "^PRETTY_NAME=.*$")
  string(REGEX REPLACE "PRETTY_NAME=[\"']?([^\"']*)[\"']?$" "\\1" OS_FULL_NAME "${OS_FULL_NAME}")
  string(REGEX REPLACE "[ ]*\\(.*\\)" "" OS_FULL_NAME "${OS_FULL_NAME}")
elseif(APPLE)
  set(OS_NAME "macOS")
  execute_process(COMMAND sw_vers -productVersion
    OUTPUT_VARIABLE OS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(OS_FULL_NAME "${OS_NAME} ${OS_VERSION}")
else()
  cmake_host_system_information(RESULT OS_NAME QUERY OS_NAME)
  cmake_host_system_information(RESULT OS_VERSION QUERY OS_VERSION)
  set(OS_FULL_NAME "${OS_NAME} ${OS_VERSION}")
endif()

cmake_host_system_information(RESULT
  NCORES QUERY NUMBER_OF_PHYSICAL_CORES)
cmake_host_system_information(RESULT
  NTHREADS QUERY NUMBER_OF_LOGICAL_CORES)

if(NOT DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
  set(ENV{CMAKE_BUILD_PARALLEL_LEVEL} ${NTHREADS})
endif()

if(NOT DEFINED ENV{CTEST_PARALLEL_LEVEL})
  set(ENV{CTEST_PARALLEL_LEVEL} ${NCORES})
endif()

if(NOT DEFINED CTEST_CONFIGURATION_TYPE)
  if(DEFINED ENV{CMAKE_BUILD_TYPE})
    set(CTEST_CONFIGURATION_TYPE $ENV{CMAKE_BUILD_TYPE})
  else()
    set(CTEST_CONFIGURATION_TYPE RelWithDebInfo)
  endif()
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --system-information
  OUTPUT_VARIABLE CMAKE_SYSTEM_INFORMATION ERROR_VARIABLE ERROR)

if(ERROR)
  message(FATAL_ERROR "Cannot detect system information")
endif()

string(REGEX REPLACE ".+CMAKE_CXX_COMPILER_ID \"([-0-9A-Za-z ]+)\".*$" "\\1"
  COMPILER_ID "${CMAKE_SYSTEM_INFORMATION}")
string(REPLACE "GNU" "GCC" COMPILER_ID "${COMPILER_ID}")

string(REGEX REPLACE ".+CMAKE_CXX_COMPILER_VERSION \"([^\"]+)\".*$" "\\1"
  COMPILER_VERSION "${CMAKE_SYSTEM_INFORMATION}")

set(CTEST_BUILD_NAME "${OS_FULL_NAME}")
string(APPEND CTEST_BUILD_NAME " ${COMPILER_ID} ${COMPILER_VERSION}")
string(APPEND CTEST_BUILD_NAME " ${CTEST_CONFIGURATION_TYPE}")

if(DEFINED ENV{CMAKE_GENERATOR})
  set(CTEST_CMAKE_GENERATOR $ENV{CMAKE_GENERATOR})
else()
  string(REGEX REPLACE ".+CMAKE_GENERATOR \"([-0-9A-Za-z ]+)\".*$" "\\1"
    CTEST_CMAKE_GENERATOR "${CMAKE_SYSTEM_INFORMATION}")
endif()

if(NOT CTEST_CMAKE_GENERATOR MATCHES "Makefile")
  string(APPEND CTEST_BUILD_NAME " ${CTEST_CMAKE_GENERATOR}")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  string(APPEND CTEST_BUILD_NAME " ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(DEFINED ENV{GITHUB_ACTIONS})
  set(CTEST_SITE "GitHub Actions ($ENV{GITHUB_REPOSITORY_OWNER})")

  if("$ENV{GITHUB_REPOSITORY_OWNER}" STREQUAL "root-project")
    set(CDASH TRUE)
    set(MODEL "Continuous")
  endif()

  if("$ENV{GITHUB_EVENT_NAME}" MATCHES "pull_request")
    set(GROUP "Pull Requests")
    set(ENV{BASE_REF} $ENV{GITHUB_SHA}^1)
    set(ENV{HEAD_REF} $ENV{GITHUB_SHA}^2)
    string(REGEX REPLACE "/merge" "" PR_NUMBER "$ENV{GITHUB_REF_NAME}")
    string(PREPEND CTEST_BUILD_NAME "#${PR_NUMBER} ($ENV{GITHUB_ACTOR})")
  else()
    set(ENV{HEAD_REF} $ENV{GITHUB_SHA})
    string(APPEND CTEST_BUILD_NAME " ($ENV{GITHUB_REF_NAME})")
  endif()

  if("$ENV{GITHUB_RUN_ATTEMPT}" GREATER 1)
    string(APPEND CTEST_BUILD_NAME " #$ENV{GITHUB_RUN_ATTEMPT}")
  endif()

  macro(section title)
      message("::group::${title}")
  endmacro()

  macro(endsection)
      message("::endgroup::")
  endmacro()
else()
  macro(section title)
  endmacro()
  macro(endsection)
  endmacro()
endif()

if(NOT DEFINED CTEST_SITE)
  site_name(CTEST_SITE)
endif()

if(NOT DEFINED CTEST_SOURCE_DIRECTORY)
  set(CTEST_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
endif()

if(NOT DEFINED CTEST_BINARY_DIRECTORY)
  get_filename_component(CTEST_BINARY_DIRECTORY "root_build" REALPATH)
endif()

if(NOT DEFINED MODEL)
  if(DEFINED CTEST_SCRIPT_ARG)
    set(MODEL ${CTEST_SCRIPT_ARG})
  else()
    set(MODEL Experimental)
  endif()
endif()

if(NOT DEFINED GROUP)
  set(GROUP ${MODEL})
endif()

set(CMAKE_ARGS $ENV{CMAKE_ARGS} ${CMAKE_ARGS})

if(COVERAGE)
  find_program(CTEST_COVERAGE_COMMAND NAMES gcov)
  list(PREPEND CMAKE_ARGS "-DCMAKE_C_FLAGS=--coverage -fprofile-update=atomic")
  list(PREPEND CMAKE_ARGS "-DCMAKE_CXX_FLAGS=--coverage -fprofile-update=atomic")
endif()

if(MEMCHECK)
  find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
endif()

if(STATIC_ANALYSIS)
  find_program(CMAKE_CXX_CLANG_TIDY NAMES clang-tidy)
  list(PREPEND CMAKE_ARGS "-DCMAKE_CXX_CLANG_TIDY=${CMAKE_CXX_CLANG_TIDY}")
endif()

foreach(FILENAME ${OS_NAME}${OS_VERSION}.cmake ${OS_NAME}.cmake config.cmake)
  if(EXISTS "${CTEST_SOURCE_DIRECTORY}/.ci/${FILENAME}")
    message(STATUS "Using CMake cache file ${FILENAME}")
    list(PREPEND CMAKE_ARGS -C ${CTEST_SOURCE_DIRECTORY}/.ci/${FILENAME})
    list(APPEND CTEST_NOTES_FILES ${CTEST_SOURCE_DIRECTORY}/.ci/${FILENAME})
    break()
  endif()
endforeach()

if(IS_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
  ctest_empty_binary_directory("${CTEST_BINARY_DIRECTORY}")
endif()

ctest_read_custom_files("${CTEST_SOURCE_DIRECTORY}")

if(IS_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/.git)
  find_program(CTEST_GIT_COMMAND NAMES git)
endif()

if(EXISTS "${CTEST_GIT_COMMAND}" AND DEFINED ENV{HEAD_REF} AND NOT DEFINED ENV{BASE_REF})
  set(CTEST_CHECKOUT_COMMAND
    "${CTEST_GIT_COMMAND} --git-dir ${CTEST_SOURCE_DIRECTORY}/.git checkout -f $ENV{HEAD_REF}")
endif()

ctest_start(${MODEL} GROUP "${GROUP}")

if(EXISTS "${CTEST_GIT_COMMAND}")
  if(DEFINED ENV{BASE_REF})
    execute_process(COMMAND ${CTEST_GIT_COMMAND} checkout -f $ENV{BASE_REF}
      WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY} ERROR_QUIET RESULT_VARIABLE GIT_STATUS)
    if(NOT ${GIT_STATUS} EQUAL 0)
      message(FATAL_ERROR "Could not checkout base ref: $ENV{BASE_REF}")
    endif()
  endif()
  if(DEFINED ENV{HEAD_REF})
    set(CTEST_GIT_UPDATE_CUSTOM
      ${CTEST_GIT_COMMAND} --git-dir ${CTEST_SOURCE_DIRECTORY}/.git checkout -f $ENV{HEAD_REF})
  else()
    set(CTEST_GIT_UPDATE_CUSTOM ${CTEST_GIT_COMMAND} --git-dir ${CTEST_SOURCE_DIRECTORY}/.git diff)
  endif()

  ctest_update()
endif()

section("Configure")
ctest_configure(OPTIONS "${CMAKE_ARGS}" RETURN_VALUE CONFIG_RESULT)

ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")
list(APPEND CTEST_NOTES_FILES ${CTEST_BINARY_DIRECTORY}/CMakeCache.txt)

if(NOT ${CONFIG_RESULT} EQUAL 0)
  if(CDASH OR (DEFINED ENV{CDASH} AND "$ENV{CDASH}"))
    ctest_submit()
  endif()
  message(FATAL_ERROR "Configuration failed")
endif()
endsection()

section("Build")
ctest_build(RETURN_VALUE BUILD_RESULT)

if(NOT ${BUILD_RESULT} EQUAL 0)
  if(CDASH OR (DEFINED ENV{CDASH} AND "$ENV{CDASH}"))
    ctest_submit()
  endif()
  message(FATAL_ERROR "Build failed")
endif()

if(INSTALL)
  set(ENV{DESTDIR} "${CTEST_BINARY_DIRECTORY}/install")
  ctest_build(TARGET install)
endif()
endsection()

section("Test")
ctest_test(PARALLEL_LEVEL $ENV{CTEST_PARALLEL_LEVEL} RETURN_VALUE TEST_RESULT)

if(NOT ${TEST_RESULT} EQUAL 0)
  message(SEND_ERROR "Tests failed")
endif()
endsection()

if(DEFINED CTEST_COVERAGE_COMMAND)
  section("Coverage")
  find_program(GCOVR NAMES gcovr)
  if(EXISTS ${GCOVR})
    execute_process(COMMAND
      ${GCOVR} --gcov-executable ${CTEST_COVERAGE_COMMAND}
        -r ${CTEST_SOURCE_DIRECTORY} ${CTEST_BINARY_DIRECTORY}
        --html-details ${CTEST_BINARY_DIRECTORY}/coverage/ ERROR_VARIABLE ERROR)
    if(ERROR)
      message(SEND_ERROR "Failed to generate coverage report")
    endif()
  endif()
  ctest_coverage()
  endsection()
endif()

if(DEFINED CTEST_MEMORYCHECK_COMMAND)
  section("Memcheck")
  ctest_memcheck()
  endsection()
endif()

if(CDASH OR (DEFINED ENV{CDASH} AND "$ENV{CDASH}"))
  ctest_submit()
endif()
