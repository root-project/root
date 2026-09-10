# Check the way the dynamic (shared library) search path is assembled from
# ROOT_LIBRARY_PATH, the system loader path and the .rootrc resource.
#
# This used to be implemented by the shell script `test_dynpath_setup.sh`; it
# is now driven by the CMake script `dynamicPathSetup.cmake` so that the test
# also runs on Windows.

ROOTTEST_ADD_TEST(DynPathSetup
                  COMMAND ${CMAKE_COMMAND}
                          -DROOT_EXE=${ROOT_root_CMD}
                          -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}/rootlibpath_test
                          -P ${CMAKE_CURRENT_SOURCE_DIR}/dynamicPathSetup.cmake
                  ENVIRONMENT ROOTENV_NO_HOME=1)
