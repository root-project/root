if(NOT ROOT_pyroot_FOUND)
    return()
endif()

ROOTTEST_ADD_TEST(compression_settings
                  MACRO test_compression_settings.py)