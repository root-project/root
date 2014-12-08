ROOTTEST_ADD_TEST(roottest-root-hist-assertSparseToTHn
                  MACRO assertSparseToTHn.C
                  COPY_TO_BUILDDIR effectiveCorrection.root)

ROOTTEST_ADD_TEST(roottest-root-hist-h2root
                  COMMAND h2root mb4i1.hbook
                  COPY_TO_BUILDDIR mb4i1.hbook
                  OUTREF h2root.ref)

ROOTTEST_ADD_TESTDIRS()
