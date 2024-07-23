if(NOT ROOT_pyroot_FOUND)
    return()
endif()

ROOTTEST_ADD_TEST(compression_settings
                  MACRO test_compression_settings.py)

ROOTTEST_ADD_TEST(input_validation
                  MACRO input_validation.py)
