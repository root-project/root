# Test only if:
# * pyspark is installed on that label for that python version
# * `dask` and `distributed` are installed on that label for that python version
if (ROOT_test_distrdf_pyspark_FOUND AND ROOT_test_distrdf_dask_FOUND)

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
    # using in the following test. In this folder, both a Dask and a Spark
    # cluster objects are created on the local machine. Each object will use 2
    # cores with multiprocessing, 4 in total. We give it more room by setting
    # the property to 6.
    # The test also locks all resources related to creation of cluster objects,
    # which are shared also with the respective folders of this project and the
    # tutorials of the root repository.
    ROOTTEST_ADD_TEST(test_all
                      MACRO test_all.py
                      ENVIRONMENT ${PYSPARK_ENV_VARS})

    # This test has to take multiple resource locks. This means that they should
    # be passed as a cmake list (semi-colon separated strings) after the
    # RESOURCE_LOCK argument to `set_tests_properties`. The issue lies in
    # forwarding this argument to ROOTTEST_ADD_TEST and then ROOT_ADD_TEST.
    # Argument forwarding and list parsing in cmake is utterly broken.
    # The safest and most reliable thing to do is to call the final cmake
    # function directly here, so we can be sure that the PROPERTIES argument
    # will be properly parsed.
    set_tests_properties(roottest-python-distrdf-common-test_all PROPERTIES
                                                                 RESOURCE_LOCK "dask_resource_lock;spark_resource_lock"
                                                                 PROCESSORS 6)
endif()
