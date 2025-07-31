ROOTTEST_GENERATE_EXECUTABLE(readFromCin readFromCin.cxx
                             LIBRARIES Core Hist RIO Net Graf Graf3d Gpad Tree Rint Postscript Matrix Physics MathCore Thread ${OptionalMultiProc}
                             FIXTURES_SETUP root-tree-readcin-executable-fixture)

ROOTTEST_ADD_TEST(readFromCin
                  COMMAND ./readFromCin ${CMAKE_CURRENT_SOURCE_DIR}/test.csv
                  OUTREF readcin.ref
                  FIXTURES_REQUIRED root-tree-readcin-executable-fixture)

ROOTTEST_ADD_TEST(parseCin
                  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/parseCin.sh ${CMAKE_CURRENT_SOURCE_DIR}/test.csv
                  OUTREF readcin.ref
                  FIXTURES_REQUIRED root-tree-readcin-executable-fixture)

ROOTTEST_ADD_TEST(parseStream
                  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/parseStream.sh ${CMAKE_CURRENT_SOURCE_DIR}/test.csv ${CMAKE_CURRENT_SOURCE_DIR}/readFromStream.cc
                  OUTREF readcinpiped.ref)

ROOTTEST_ADD_TEST(readcin
                  MACRO runreadcin.C
                  COPY_TO_BUILDDIR test.csv
                  OUTREF readcin.ref-cling)
