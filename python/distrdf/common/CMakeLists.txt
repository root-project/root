# Test only if:
# * pyspark is installed on that label for that python version
# * `dask` and `distributed` are installed on that label for that python version
if (ROOT_test_distrdf_pyspark_FOUND AND ROOT_test_distrdf_dask_FOUND)

    # Define environment variables needed in all pyspark tests
    set(PYSPARK_ENV_VARS PYSPARK_PYTHON=${PYTHON_EXECUTABLE_Development_Main})

    ROOTTEST_ADD_TEST(common_test_rungraphs
                    MACRO test_rungraphs.py
                    ENVIRONMENT ${PYSPARK_ENV_VARS})

endif()
