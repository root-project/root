ROOTTEST_GENERATE_EXECUTABLE(readFromCin readFromCin.cxx LIBRARIES Core Hist RIO Net Graf Graf3d Gpad Tree Rint Postscript Matrix Physics MathCore Thread ${OptionalMultiProc})

ROOTTEST_ADD_TEST(readFromCin
                  COMMAND ./readFromCin ${CMAKE_CURRENT_SOURCE_DIR}/test.csv
                  OUTREF readcin.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_ADD_TEST(parseCin
                  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/parseCin.sh ${CMAKE_CURRENT_SOURCE_DIR}/test.csv
                  OUTREF readcin.ref 
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_ADD_TESTDIRS()

