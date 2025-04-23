ROOTTEST_ADD_TEST(runownership MACRO runownership.C+ OUTREF ownership.ref)
ROOTTEST_ADD_TEST(testSparse MACRO testSparse.cxx OUTREF testSparse.ref)
ROOT_ADD_GTEST(gh15874 test_gh15874.cxx
               COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/test_gh15874.root
               LIBRARIES ROOT::Core ROOT::Hist ROOT::RIO)
