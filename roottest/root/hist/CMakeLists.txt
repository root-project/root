ROOTTEST_ADD_TEST(mergeTH2Poly
                  MACRO th2PolyMerge.C
                  ${WILLFAIL_ON_WIN32})

ROOTTEST_ADD_TEST(assertSparseToTHn
                  MACRO assertSparseToTHn.C
                  COPY_TO_BUILDDIR effectiveCorrection.root)

if(fortran AND CMAKE_Fortran_COMPILER AND NOT CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
  ROOTTEST_ADD_TEST(h2root
                    COMMAND ${CMAKE_COMMAND} -E env h2root mb4i1.hbook
                    COPY_TO_BUILDDIR mb4i1.hbook
                    OUTREF h2root.ref)

  # Mask the known failure due to a read error in the file
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin" AND CMAKE_HOST_SYSTEM_VERSION VERSION_GREATER_EQUAL 25.4)
    set_tests_properties(roottest-root-hist-h2root PROPERTIES
      SKIP_REGULAR_EXPRESSION "Error cannot open input file: mb4i1\.hbook"
      WILL_FAIL True)
  endif()
endif()

ROOTTEST_ADD_TESTDIRS()
