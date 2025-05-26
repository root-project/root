# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Dictionary list
set(CPPYY_TEST_DICTS
advancedcpp
advancedcpp2
conversions
cpp11features
crossinheritance
datatypes
doc_helper
example01
fragile
operators
overloads
pythonizables
std_streams
templates
stltypes)

foreach(DICT ${CPPYY_TEST_DICTS})
    # Generate dictionary for the tests
    ROOT_GENERATE_DICTIONARY(G__${DICT}Dict ${CMAKE_CURRENT_SOURCE_DIR}/${DICT}.h LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/${DICT}.xml)
    # Create necessary shared libraries for the tests
    add_library(${DICT}Dict SHARED ${CMAKE_CURRENT_SOURCE_DIR}/${DICT}.cxx G__${DICT}Dict.cxx)
    target_include_directories(${DICT}Dict PUBLIC ${Python3_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/include)
    target_link_libraries(${DICT}Dict PUBLIC Python3::Python ROOT::Core)
    target_compile_options(${DICT}Dict PRIVATE "-w")
endforeach()

# Tests list
set(CPPYY_TESTS
test_aclassloader
test_advancedcpp
test_api
test_boost
test_conversions
test_crossinheritance
test_datatypes
test_doc_features
test_eigen
test_fragile
# The leakcheck test is disabled due to it's sporadic nature, especially fragile on VMs
# test_leakcheck
test_lowlevel
test_numba
test_operators
test_overloads
test_pythonify
test_pythonization
test_regression
test_streams
test_templates
test_concurrent)

if(NOT CMAKE_CXX_STANDARD EQUAL 23)
    # test_cpp11features crashes when run with C++23
    list(APPEND CPPYY_TESTS test_cpp11features)
endif()

if(NOT APPLE)
    # test_stltypes completely fails on OSX
    # because it is unable to load the necessary shared libraries
    list(APPEND CPPYY_TESTS test_stltypes)
endif()

# Configure the tests
foreach(TEST ${CPPYY_TESTS})
    ROOT_ADD_PYUNITTEST(${TEST} ${TEST}.py GENERIC ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR} PYTHON_DEPS pytest)
endforeach()
