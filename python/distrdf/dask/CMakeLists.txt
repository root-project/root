# Test only if:
# * `dask` and `distributed` are installed on that label for that python version
if (ROOT_test_distrdf_dask_FOUND)

    # We use the PROCESSORS property to tell cmake how many cores we will be
    # using in the following test. In this folder, a Dask cluster object is
    # created on the local machine. It is going to use 2 cores with
    # multiprocessing, we give it more room by setting the property to 4.
    # The test also locks a resource for the creation of a Dask cluster, which
    # is shared with the "common" folder and the tutorials of the root
    # repository.
    ROOTTEST_ADD_TEST(test_all
                      MACRO test_all.py
                      TIMEOUT 420
                      PROPERTIES PROCESSORS 4 RESOURCE_LOCK dask_resource_lock)

endif()
