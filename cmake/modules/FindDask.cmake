# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Find if dask is installed on the environment of the main Python executable
# used to build ROOT.
#
# Installing the package `dask` through conda always provides the `distributed`
# The same is not true for pip, where the two packages have to be installed
# separately.
#
# This module sets the following variables
#  Dask_FOUND - system has dask and it is usable
#  Dask_DEPENDENCIES_READY - the environment could import the `dask` and `distributed` packages
#  Dask_VERSION_STRING - Dask version string

# Import `dask` and `distributed` using the main Python executable, print dask version
execute_process(
    COMMAND ${PYTHON_EXECUTABLE_Development_Main} -c "import distributed; import dask; print(dask.__version__)"
    RESULT_VARIABLE _DASK_IMPORT_EXIT_STATUS
    OUTPUT_VARIABLE _DASK_VALUES_OUTPUT
    ERROR_VARIABLE _DASK_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Exit status equal to zero means success
if(_DASK_IMPORT_EXIT_STATUS EQUAL 0)
    # Build the version string
    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" Dask_VERSION_STRING "${_DASK_VALUES_OUTPUT}")
    # Signal to CMake that the environment could import `dask` and `distributed` packages
    set(Dask_DEPENDENCIES_READY TRUE)
else()
    message(STATUS "Python package 'dask' could not be imported with ${PYTHON_EXECUTABLE_Development_Main}\n"
                   "${_DASK_ERROR_VALUE}"
    )
endif()

find_package_handle_standard_args(Dask
    REQUIRED_VARS Dask_DEPENDENCIES_READY
    VERSION_VAR Dask_VERSION_STRING
)
