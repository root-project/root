ROOTTEST_GENERATE_EXECUTABLE(readFromCin readFromCin.cxx LIBRARIES Core Hist RIO Net Graf Graf3d Gpad Tree Rint Postscript Matrix Physics MathCore Thread MultiProc)

ROOTTEST_ADD_TEST(roottest-root-tree-readcin-readFromCin
                  COMMAND readFromCin ${CMAKE_CURRENT_SOURCE_DIR}/test.csv
                  OUTREF ${CMAKE_CURRENT_SOURCE_DIR}/readcin.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_ADD_TEST(roottest-root-tree-readcin-parseCin
                  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/parseCin.sh ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF ${CMAKE_CURRENT_SOURCE_DIR}/readcin.ref )

ROOTTEST_ADD_TESTDIRS()

