if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(api
                    MACRO runPyAPITest.C
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyAPITest.ref
                    ENVIRONMENT CLING_STANDARD_PCH=none
                                CPPYY_BACKEND_LIBRARY=${CMAKE_BINARY_DIR}/lib/libcppyy_backend.so
                    ${PYTESTS_WILLFAIL})

  ROOTTEST_ADD_TEST(class MACRO
                    runPyClassTest.C
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyClassTest.ref
                    ENVIRONMENT CLING_STANDARD_PCH=none
                    CPPYY_BACKEND_LIBRARY=${CMAKE_BINARY_DIR}/lib/libcppyy_backend.so
                    ${PYTESTS_WILLFAIL})

  ROOTTEST_ADD_TEST(cling
                    MACRO PyROOT_clingtests.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_clingtests.ref
                    ${PYTESTS_WILLFAIL})
endif()
