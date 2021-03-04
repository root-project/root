# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Find if pyspark is installed on the system. It depends on:
# 1. Java, minimum required 1.8
# 2. py4j
#
# Java is a hard requirement in pyspark but still doing `import pyspark` does not
# report an error if it is not installed or found. For that reason we look for Java
# on the system with REQUIRED flag.
#
# On the other hand, py4j should be present on the system if pyspark was installed
# either through pip, conda or via downloading the official binaries. Thus, we can
# check for errors in `import pyspark` to catch also the case that py4j is missing somehow.
#
# Only the main Python executable that is used in the ROOT build will be used to find pyspark.
# This module sets the following variables
#  PySpark_FOUND - system has pyspark and it is usable
#  PySpark_VERSION_STRING - pyspark version string

message(STATUS "Looking for PySpark dependency: Java")
find_package(Java 1.8 REQUIRED COMPONENTS Runtime)

message(STATUS "Found Java ${Java_JAVA_EXECUTABLE}")
message(STATUS "Java version ${Java_VERSION_STRING}")

# Import pyspark using the main Python executable, print its version and path to the __init__.py file
execute_process(
    COMMAND ${PYTHON_EXECUTABLE_Development_Main}
            -c "import re, pyspark; print(pyspark.__version__); print(re.compile('/__init__.py.*').sub('',pyspark.__file__))" 
    RESULT_VARIABLE _PYSPARK_IMPORT_SUCCESS
    OUTPUT_VARIABLE _PYSPARK_VALUES_OUTPUT
    ERROR_VARIABLE _PYSPARK_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(_PYSPARK_IMPORT_SUCCESS EQUAL 0)
    # Convert the process output into a list
    string(REGEX REPLACE "\n" ";" _PYSPARK_VALUES ${_PYSPARK_VALUES_OUTPUT})
    # Just in case there is unexpected output from the Python command.
    list(GET _PYSPARK_VALUES -2 _PYSPARK_VERSION)
    list(GET _PYSPARK_VALUES -1 PySpark_HOME)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PySpark_VERSION_STRING "${_PYSPARK_VERSION}")
else()
    message(STATUS "Python package 'pyspark' could not be imported with ${PYTHON_EXECUTABLE_Development_Main}\n"
                   "${_PYSPARK_ERROR_VALUE}"
    )
endif()

find_package_handle_standard_args(PySpark
    REQUIRED_VARS PySpark_HOME
    VERSION_VAR PySpark_VERSION_STRING
)
