#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
ROOTTEST_ADD_TEST(interpreted
                  MACRO runTCut.C
                  OUTREF TCut.ref)

ROOTTEST_ADD_TEST(compiled
                  MACRO runTCut.C+
                  OUTREF TCutcomp.ref)
