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
                  FIXTURES_SETUP root-io-datamodelevolution-stl-libDataModelCond-fixture)


ROOTTEST_ADD_TEST(MapContent
                  MACRO execMapContent.C+
                  OUTREF execMapContent.ref
                  COPY_TO_BUILDDIR cond09_mc.000029.gen.COND._0002.pool.root
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-stl-libDataModelCond-fixture)

ROOTTEST_COMPILE_MACRO(writeFile.C
                  FIXTURES_SETUP root-io-datamodelevolution-stl-writeFile-compile-fixture)

ROOTTEST_COMPILE_MACRO(readFile.C
                  FIXTURES_SETUP root-io-datamodelevolution-stl-readFile-compile-fixture)

ROOTTEST_ADD_TEST(WriteFile
                  MACRO execWriteFile.C
                  OUTREF execWriteFile.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-stl-writeFile-compile-fixture
                  FIXTURES_SETUP root-io-datamodelevolution-stl-WriteFile-fixture)

ROOTTEST_ADD_TEST(ReadFile
                  MACRO execReadFile.C
                  OUTREF execReadFile.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-stl-WriteFile-fixture
                                    root-io-datamodelevolution-stl-readFile-compile-fixture)

