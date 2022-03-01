# Test only if:
# * RDataFrame is available
# * PyROOT is available
# * The Python version is greater than 3.7
if (ROOT_dataframe_FOUND AND
    ROOT_pyroot_FOUND AND
    PYTHON_VERSION_STRING_Development_Main VERSION_GREATER_EQUAL 3.7
    AND NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES arm64))

    file(COPY test_headers DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY test_shared_libraries DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY emptytree.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

    ROOTTEST_ADD_TESTDIRS()

endif()
