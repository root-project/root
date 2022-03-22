# Test only if:
# * `dask` and `distributed` are installed on that label for that python version
if (ROOT_test_distrdf_dask_FOUND)

    # We use the PROCESSORS property to tell cmake how many cores we will be
    # using in the following test. In this folder, a Dask cluster object is
    # created on the local machine. It is going to use 2 cores with
    # multiprocessing, so the property is set to 2 in this case.
    ROOTTEST_ADD_TEST(test_all
                      MACRO test_all.py
                      TIMEOUT 420
                      PROPERTIES PROCESSORS 2)

endif()
