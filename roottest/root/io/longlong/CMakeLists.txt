ROOTTEST_GENERATE_DICTIONARY(libTestIoLonglong
                             Class.h
                             LINKDEF linkdef.h
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-lonlong-libTestIoLonglong-fixture)

# probably need before grep -v G__letLonglong
if(MSVC)
  set(_cmd findstr "long[ ]long")
else()
  set(_cmd grep "long long")
endif()

ROOTTEST_ADD_TEST(longlong
                  COMMAND ${_cmd} libTestIoLonglong.cxx
                  PASSRC 1
                  OUTREF longlong.ref
                  FIXTURES_REQUIRED root-io-lonlong-libTestIoLonglong-fixture
                  WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR})
