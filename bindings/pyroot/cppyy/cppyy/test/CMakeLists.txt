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
    ROOT_GENERATE_DICTIONARY(G__${DICT} ${CMAKE_CURRENT_SOURCE_DIR}/${DICT}.h ${CMAKE_CURRENT_SOURCE_DIR}/${DICT}.xml NOINSTALL)
    # Create shared necessary libraries for the tests
    ROOT_LINKER_LIBRARY(${DICT} ${CMAKE_CURRENT_SOURCE_DIR}/${DICT}.cxx NOINSTALL LIBRARIES Python3::Python)
endforeach()

# Tests list
set(CPPYY_TESTS
test_aclassloader
test_advancedcpp
test_api
test_boost
test_conversions
test_cpp11features
test_crossinheritance
test_datatypes
test_doc_features
test_eigen
test_fragile
test_leakcheck
test_lowlevel
test_numba
test_operators
test_overloads
test_pythonify
test_pythonization
test_regression
test_stltypes
test_streams
test_templates
test_concurrent)

# Configure the tests
foreach(TEST ${CPPYY_TESTS})
    ROOT_ADD_PYUNITTEST(${TEST} ${TEST}.py GENERIC ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_CURRENT_SOURCE_DIR} PYTHON_DEPS pytest)
endforeach()
