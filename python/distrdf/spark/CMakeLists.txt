# Test only if:
# * pyspark is installed on that label for that python version
if (ROOT_test_distrdf_pyspark_FOUND)

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

    # We use the PROCESSORS property to tell cmake how many cores we will be
    # using in the following test. In this folder, a Dask cluster object is
    # created on the local machine. It is going to use 2 cores with
    # multiprocessing, so the property is set to 2 in this case.
    ROOTTEST_ADD_TEST(test_all
                      MACRO test_all.py
                      ENVIRONMENT ${PYSPARK_ENV_VARS}
                      PROPERTIES PROCESSORS 2)

endif()
