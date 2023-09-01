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
# report an error if it is not installed or found.
#
# On the other hand, py4j should be present on the system if pyspark was installed
# either through pip, conda or via downloading the official binaries. Thus, we can
# check for errors in `import pyspark` to catch also the case that py4j is missing somehow.
#
# Only the main Python executable that is used in the ROOT build will be used to find pyspark.
# This module sets the following variables
#  PySpark_FOUND - system has pyspark and it is usable
#  PySpark_DEPENDENCIES_READY - the environment could import pyspark and Java runtime was found
#  PySpark_VERSION_STRING - pyspark version string

message(STATUS "Looking for PySpark dependency: Java")
if(PySpark_FIND_REQUIRED)
    find_package(Java 1.8 REQUIRED COMPONENTS Runtime)
else()
    find_package(Java 1.8 COMPONENTS Runtime)
endif()

if(Java_FOUND)
    message(STATUS "Found Java ${Java_JAVA_EXECUTABLE}")
    message(STATUS "Java version ${Java_VERSION_STRING}")

    # Import pyspark using the main Python executable, print its version and path to the __init__.py file
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE_Development_Main} -c "import pyspark; print(pyspark.__version__)" 
        RESULT_VARIABLE _PYSPARK_IMPORT_EXIT_STATUS
        OUTPUT_VARIABLE _PYSPARK_VALUES_OUTPUT
        ERROR_VARIABLE _PYSPARK_ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Exit status equal to zero means success
    if(_PYSPARK_IMPORT_EXIT_STATUS EQUAL 0)
        # Build the version string
        string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PySpark_VERSION_STRING "${_PYSPARK_VALUES_OUTPUT}")
        # Signal to CMake that the environment could import pyspark and Java runtime was found
        set(PySpark_DEPENDENCIES_READY TRUE)
    else()
        message(STATUS "Python package 'pyspark' could not be imported with ${PYTHON_EXECUTABLE_Development_Main}\n"
                    "${_PYSPARK_ERROR_VALUE}"
        )
    endif()

find_package_handle_standard_args(PySpark
    REQUIRED_VARS PySpark_DEPENDENCIES_READY
    VERSION_VAR PySpark_VERSION_STRING
)
endif()
