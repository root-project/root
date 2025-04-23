#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
ROOTTEST_ADD_OLDTEST()
ROOTTEST_ADD_TESTDIRS()

# ROOT-10046
ROOTTEST_ADD_TEST(test_oddName
                  COPY_TO_BUILDDIR root-10046.root
                  MACRO oddName.C)

# ROOT-9757
ROOTTEST_ADD_TEST(test_numberBranchesRead
                  MACRO  test_numberBranchesRead.C
                  MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}/../../dataframe/Slimmed_TotemNTuple_9883.040.ntuple.root\"")

# ROOT-6286
ROOTTEST_ADD_TEST(test_runcleanupChain
                  MACRO runcleanupChain.C)
