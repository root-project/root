# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_custom_target(cppyy_tests_dict ALL COMMAND make all WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bindings/pyroot/cppyy/cppyy/test)

file(GLOB CPPYY_TEST_SRC "test_*.py")

foreach(CPPYY_TEST_FILE ${CPPYY_TEST_SRC})
    get_filename_component(CPPYY_TEST_FILE_NAME ${CPPYY_TEST_FILE} NAME)
    string(REPLACE ".py" "" CPPYY_TEST_FILE_NO_EXT ${CPPYY_TEST_FILE_NAME})

    ROOT_ADD_PYUNITTEST(${CPPYY_TEST_FILE_NO_EXT} ${CPPYY_TEST_FILE_NAME} GENERIC PYTHON_DEPS pytest ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
