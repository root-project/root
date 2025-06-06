ROOTTEST_COMPILE_MACRO(other.C
                       FIXTURES_SETUP other-fixture)

# all offsets calculated for 64 bit, therefore on other platforms it will not work
if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(willf WILLFAIL)
endif()

ROOTTEST_ADD_TEST(template32
                  MACRO runtemplate32.C+
                  OUTREF template32.ref
                  ${willf}
                  FIXTURES_REQUIRED other-fixture)

ROOTTEST_GENERATE_DICTIONARY(Dict32
                  runtemplate32.h
                  LINKDEF linkdef32.h
                  NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_GENERATE_DICTIONARY(libTwoInOne
                  runtemplate32.h
                  LINKDEF linkdef.h
                  SOURCES Dict32.cxx
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_REQUIRED Dict32_fixture)
