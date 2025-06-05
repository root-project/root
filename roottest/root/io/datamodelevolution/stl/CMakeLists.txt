ROOTTEST_GENERATE_DICTIONARY(libDataModelCond
                  libcond/DataHeaderElement_p5.h
                  libcond/DataHeaderForm_p5.h
                  libcond/DataHeader_p5.h
                  libcond/DetCondKeyTrans.h
                  libcond/HepGeom__Transform3D.h
                  libcond/libcondProjectHeaders.h
                  libcond/libcondProjectInstances.h
                  LINKDEF libcond/libcondLinkDef.h
                  SOURCES libcond/libcondProjectSource.cxx
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP libDataModelCond_fixture)

ROOTTEST_ADD_TEST(MapContent
                  MACRO execMapContent.C+
                  OUTREF execMapContent.ref
                  COPY_TO_BUILDDIR cond09_mc.000029.gen.COND._0002.pool.root
                  LABELS longtest
                  FIXTURES_REQUIRED libDataModelCond_fixture)

ROOTTEST_ADD_TEST(WriteFile
                  MACRO execWriteFile.C
                  OUTREF execWriteFile.ref
                  LABELS longtest
                  FIXTURES_SETUP WriteFile_fixture)

ROOTTEST_ADD_TEST(ReadFile
                  MACRO execReadFile.C
                  OUTREF execReadFile.ref
                  LABELS longtest
                  FIXTURES_REQUIRED WriteFile_fixture)

