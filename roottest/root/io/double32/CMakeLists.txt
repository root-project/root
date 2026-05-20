ROOTTEST_COMPILE_MACRO(other.C
                       FIXTURES_SETUP root-io-double32-other-fixture)

ROOTTEST_ADD_TEST(template32
                  MACRO runtemplate32.C+
                  OUTREF template32.ref
                  OUTCNVCMD sed -e s,offset=[[:space:]]*[0-9]*,offset=__deleted__,g
                  FIXTURES_REQUIRED root-io-double32-other-fixture)

ROOTTEST_GENERATE_DICTIONARY(Dict32
                  runtemplate32.h
                  LINKDEF linkdef32.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP root-io-double32-Dict32-fixture)

# test building of library with two dictionaries
ROOTTEST_GENERATE_DICTIONARY(libTwoInOne
                  runtemplate32.h
                  LINKDEF linkdef.h
                  SOURCES Dict32.cxx
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_REQUIRED root-io-double32-Dict32-fixture)
