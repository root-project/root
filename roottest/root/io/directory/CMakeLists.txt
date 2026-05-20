ROOTTEST_ADD_TEST(assertCycleParsing
                  MACRO assertCycleParsing.C
                  ERRREF assertCycleParsing.ref)

ROOTTEST_ADD_TEST(assertSubdirAndTree
                  MACRO assertSubdirAndTree.C
                  COPY_TO_BUILDDIR Collision12-ANNPID.root
                  ERRREF assertSubdirAndTree.ref)

ROOTTEST_ADD_TEST(cd
                  MACRO runcd.C
                  OUTREF cd.ref)

ROOTTEST_ADD_TEST(execKeyOrder
                  MACRO execKeyOrder.cxx+
                  OUTREF execKeyOrder.ref)

ROOTTEST_ADD_TEST(execmanydirs
                  MACRO execmanydirs.cxx+
                  OUTREF execmanydirs.ref)

ROOTTEST_ADD_TEST(execOverWriteDir
                  MACRO execOverWriteDir.C+
                  OUTREF execOverWriteDir.ref)

ROOTTEST_ADD_TEST(execRecursiveGet
                  MACRO execRecursiveGet.C+
                  COPY_TO_BUILDDIR objstring.root
                  OUTREF execRecursiveGet.ref)

ROOTTEST_ADD_TEST(getobj
                  MACRO rungetobj.C
                  OUTREF getobj.ref)

ROOTTEST_ADD_TEST(testFindObjectAny
                  MACRO testFindObjectAny.C
                  OUTREF testFindObjectAny.ref)

ROOTTEST_ADD_TEST(withautoadd
                  MACRO withautoadd.C+
                  OUTREF withautoadd.ref)
