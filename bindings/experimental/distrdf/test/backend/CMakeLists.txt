# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Don't bother testing the distributed layer of RDataFrame if it's not available
# in the build
if (dataframe AND NOT MSVC)

file(COPY test_headers DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 2clusters.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 4clusters.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 1000clusters.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY emptytree.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY Slimmed_ntuple.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

ROOT_ADD_PYUNITTEST(distrdf_unit_backend_test_common test_common.py)
ROOT_ADD_PYUNITTEST(distrdf_unit_backend_test_dist test_dist.py)

# pyspark is required to run this test
if(test_distrdf_pyspark)
    # Define environment variables needed in all pyspark tests
    set(PYSPARK_ENV_VARS PYSPARK_PYTHON=${PYTHON_EXECUTABLE_Development_Main})

    if(MACOSX_VERSION VERSION_GREATER_EQUAL 10.13)
        # MacOS has changed rules about forking processes after 10.13
        # Running pyspark tests with XCode Python3 throws crashes with errors like:
        # `objc[17271]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called.`
        # This issue should have been fixed after Python 3.8 (see https://bugs.python.org/issue33725)
        # Indeed, any other Python 3.8+ executable does not show this crash. It is
        # specifically the XCode Python executable that triggers this.
        # For now, there seems no other way than this workaround,
        # which effectively sets the behaviour of `fork` back to MacOS 10.12
        list(APPEND PYSPARK_ENV_VARS OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES)
    endif()

    ROOT_ADD_PYUNITTEST(distrdf_unit_backend_test_spark test_spark.py ENVIRONMENT ${PYSPARK_ENV_VARS})
endif()

endif()
