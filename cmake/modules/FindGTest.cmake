# Find the gtest and gmock includes and library.
#
# This module defines
# GTEST_LIBRARIES
# GTEST_MAIN_LIBRARIES
# GTEST_INCLUDE_DIRS
# GMOCK_LIBRARIES
# GMOCK_MAIN_LIBRARIES
# GMOCK_INCLUDE_DIRS
#
# GTEST_FOUND           true if all libraries present

find_package(Threads QUIET)

find_path(GTEST_INCLUDE_DIRS NAMES gtest/gtest.h)
find_library(GTEST_LIBRARIES NAMES gtest)
find_library(GTEST_MAIN_LIBRARIES NAMES gtest_main)

find_path(GMOCK_INCLUDE_DIRS NAMES gmock/gmock.h)
find_library(GMOCK_LIBRARIES NAMES gmock)
find_library(GMOCK_MAIN_LIBRARIES NAMES gmock_main)

# Special for EPEL 7's gmock
if(NOT GMOCK_LIBRARIES)
  find_path(GMOCK_SRC_DIR NAMES gmock-all.cc PATHS /usr/src/gmock)
endif()

if(NOT GMOCK_MAIN_LIBRARIES)
  find_path(GMOCK_MAIN_SRC_DIR NAMES gmock_main.cc PATHS /usr/src/gmock)
endif()

if (GTEST_INCLUDE_DIRS AND
    GTEST_LIBRARIES AND
    GTEST_MAIN_LIBRARIES AND
    GMOCK_INCLUDE_DIRS AND
    (GMOCK_LIBRARIES OR GMOCK_SRC_DIR) AND
    (GMOCK_MAIN_LIBRARIES OR GMOCK_MAIN_SRC_DIR))

  add_library(gtest UNKNOWN IMPORTED)
  set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION ${GTEST_LIBRARIES}
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIRS})
  target_link_libraries(gtest INTERFACE Threads::Threads)

  add_library(gtest_main UNKNOWN IMPORTED)
  set_target_properties(gtest_main PROPERTIES
    IMPORTED_LOCATION ${GTEST_MAIN_LIBRARIES})
  target_link_libraries(gtest_main INTERFACE gtest Threads::Threads)

  if(GMOCK_LIBRARIES)
    add_library(gmock UNKNOWN IMPORTED)
    set_target_properties(gmock PROPERTIES
      IMPORTED_LOCATION ${GMOCK_LIBRARIES}
      INTERFACE_INCLUDE_DIRECTORIES ${GMOCK_INCLUDE_DIRS})
  else()
    add_library(gmock STATIC ${GMOCK_SRC_DIR}/gmock-all.cc)
    target_include_directories(gmock PUBLIC ${GMOCK_INCLUDE_DIRS})
    set(GMOCK_LIBRARIES gmock)
  endif()
  target_link_libraries(gmock INTERFACE gtest Threads::Threads)

  if(GMOCK_MAIN_LIBRARIES)
    add_library(gmock_main UNKNOWN IMPORTED)
    set_target_properties(gmock_main PROPERTIES
      IMPORTED_LOCATION ${GMOCK_MAIN_LIBRARIES})
  else()
    add_library(gmock_main STATIC ${GMOCK_MAIN_SRC_DIR}/gmock_main.cc)
    set(GMOCK_MAIN_LIBRARIES gmock_main)
  endif()
  target_link_libraries(gmock_main INTERFACE gmock Threads::Threads)

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GTest DEFAULT_MSG
  GTEST_LIBRARIES
  GTEST_MAIN_LIBRARIES
  GTEST_INCLUDE_DIRS
  GMOCK_LIBRARIES
  GMOCK_MAIN_LIBRARIES
  GMOCK_INCLUDE_DIRS)
