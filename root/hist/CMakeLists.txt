ROOTTEST_ADD_TEST(assertSparseToTHn
                  MACRO assertSparseToTHn.C
                  COPY_TO_BUILDDIR effectiveCorrection.root)
if(ROOT_fortran_FOUND)
  ROOTTEST_ADD_TEST(h2root
                    COMMAND h2root mb4i1.hbook
                    COPY_TO_BUILDDIR mb4i1.hbook
                    OUTREF h2root.ref)
endif()

ROOTTEST_ADD_TESTDIRS()
