ROOTTEST_ADD_TEST(mergeTH2Poly
                  MACRO th2PolyMerge.C
                  ${WILLFAIL_ON_WIN32})

ROOTTEST_ADD_TEST(assertSparseToTHn
                  MACRO assertSparseToTHn.C
                  COPY_TO_BUILDDIR effectiveCorrection.root)

if(fortran AND CMAKE_Fortran_COMPILER AND NOT CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
  ROOTTEST_ADD_TEST(h2root
                    COMMAND h2root mb4i1.hbook
                    COPY_TO_BUILDDIR mb4i1.hbook
                    OUTREF h2root.ref)
  ROOT_ADD_GTEST(THbookFile THbookFile.cxx LIBRARIES Hbook)
  set_tests_properties(roottest-root-hist-h2root PROPERTIES FIXTURES_SETUP hbookFile)
  set_tests_properties(gtest-roottest-root-hist-THbookFile PROPERTIES FIXTURES_REQUIRED hbookFile)
endif()

ROOTTEST_ADD_TESTDIRS()
