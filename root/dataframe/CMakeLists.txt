ROOTTEST_ADD_TESTDIRS()

# Workaround (k)ubuntu bug: runtime issue when using pthread and gcc48
if(ROOTTEST_OS_ID MATCHES Ubuntu)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

set(DFLIBRARIES Core TreePlayer Hist Tree RIO MathCore)

ROOTTEST_GENERATE_EXECUTABLE(typeguessing test_typeguessing.cxx LIBRARIES ${DFLIBRARIES})
ROOTTEST_ADD_TEST(typeguessing
                  EXEC typeguessing
                  OUTREF test_typeguessing.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(misc test_misc.cxx LIBRARIES ${DFLIBRARIES})
ROOTTEST_ADD_TEST(misc
                  EXEC misc
                  OUTREF test_misc.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(functiontraits test_functiontraits.cxx LIBRARIES ${DFLIBRARIES})
ROOTTEST_ADD_TEST(functiontraits
                  EXEC functiontraits
                  OUTREF test_functiontraits.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_ADD_TEST(ctors
                  MACRO test_ctors.C
                  OUTREF test_ctors.ref)

ROOTTEST_GENERATE_EXECUTABLE(branchoverwrite test_branchoverwrite.cxx LIBRARIES ${DFLIBRARIES})
ROOTTEST_ADD_TEST(branchoverwrite
                  EXEC branchoverwrite
                  OUTREF test_branchoverwrite.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(regression_multipletriggerrun regression_multipletriggerrun.cxx LIBRARIES ${DFLIBRARIES})
ROOTTEST_ADD_TEST(regression_multipletriggerrun
                  EXEC regression_multipletriggerrun
                  OUTREF regression_multipletriggerrun.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(regression_invalidref regression_invalidref.cxx LIBRARIES ${DFLIBRARIES})
ROOTTEST_ADD_TEST(regression_invalidref
                  EXEC regression_invalidref
                  OUTREF regression_invalidref.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})


if(ROOT_imt_FOUND)

   ROOTTEST_GENERATE_EXECUTABLE(regression_zeroentries regression_zeroentries.cxx LIBRARIES ${DFLIBRARIES})
   ROOTTEST_ADD_TEST(regression_zeroentries
                     EXEC regression_zeroentries
                     OUTREF regression_zeroentries.ref
                     DEPENDS ${GENERATE_EXECUTABLE_TEST})


ROOTTEST_GENERATE_EXECUTABLE(foreach test_foreach.cxx LIBRARIES ${DFLIBRARIES})
ROOTTEST_ADD_TEST(foreach
                  EXEC foreach
                  OUTREF test_foreach.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(reports test_reports.cxx LIBRARIES ${DFLIBRARIES})
ROOTTEST_ADD_TEST(reports
                  EXEC reports
                  OUTREF test_reports.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

endif()
