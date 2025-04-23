# Test only if:
# * RDataFrame is available
# * PyROOT is available
if (ROOT_dataframe_FOUND AND
    ROOT_pyroot_FOUND)

    file(COPY test_headers DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY test_shared_libs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY test_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

    ROOTTEST_ADD_TESTDIRS()

endif()
