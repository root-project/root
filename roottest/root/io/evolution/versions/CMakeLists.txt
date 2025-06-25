ROOTTEST_ADD_TESTDIRS()

ROOTTEST_ADD_TEST(compile-1
                  MACRO compile.cxx
                  MACROARG "\"1\""
                  LABELS longtest io
                  COPY_TO_BUILDDIR myclass.h myclass.cxx veclong64.h
                  FIXTURES_SETUP root-io-evolution-versions-1-compile)
list(APPEND fixtures root-io-evolution-versions-1-compile)

foreach(indx RANGE 2 7)
  ROOTTEST_ADD_TEST(compile-${indx}
                    MACRO compile.cxx
                    MACROARG "\"${indx}\""
                    LABELS longtest io
                    FIXTURES_REQUIRED root-io-evolution-versions-1-compile
                    FIXTURES_SETUP root-io-evolution-versions-${indx}-compile)
  list(APPEND fixtures root-io-evolution-versions-${indx}-compile)
endforeach()

foreach(indx 1 2 3 6 7)
  ROOTTEST_ADD_TEST(file-${indx}
                    MACRO wcomp.cxx
                    MACROARG "\"${indx}\""
                    LABELS longtest io
                    FIXTURES_REQUIRED root-io-evolution-versions-${indx}-compile
                    FIXTURES_SETUP root-io-evolution-versions-${indx}-file)
  list(APPEND fixtures root-io-evolution-versions-${indx}-file)
endforeach()

foreach(indx RANGE 1 6)
  ROOTTEST_ADD_TEST(AddVersion-1-${indx}
                    MACRO runAddVersion.cxx
                    MACROARG "1, \"${indx}\""
                    OUTCNVCMD sed -e s,offset=[[:space:]]*[0-9]*,offset=na,g
                    OUTREF AddVersion-1-${indx}.ref
                    LABELS longtest io
                    FIXTURES_REQUIRED ${fixtures})
  ROOTTEST_ADD_TEST(AddVersion-2-${indx}
                    MACRO runAddVersion.cxx
                    MACROARG "2, \"${indx}\""
                    OUTCNVCMD sed -e s,offset=[[:space:]]*[0-9]*,offset=na,g
                    OUTREF AddVersion-2-${indx}.ref
                    LABELS longtest io
                    FIXTURES_REQUIRED ${fixtures})
endforeach()

foreach(indx RANGE 6 7)
  ROOTTEST_ADD_TEST(AddVersion-3-${indx}
                    MACRO runAddVersion.cxx
                    MACROARG "3, \"${indx}\""
                    OUTCNVCMD sed -e s,offset=[[:space:]]*[0-9]*,offset=na,g
                    OUTREF AddVersion-3-${indx}.ref
                    LABELS longtest io
                    FIXTURES_REQUIRED ${fixtures})
endforeach()

foreach(indx RANGE 1 2)
  ROOTTEST_ADD_TEST(veclong64-${indx}
                    MACRO compile.cxx
                    MACROARG "\"${indx}\", \"veclong64.h\""
                    LABELS longtest io
                    FIXTURES_REQUIRED root-io-evolution-versions-1-compile
                    FIXTURES_SETUP root-io-evolution-versions-veclong64-${indx})
  ROOTTEST_ADD_TEST(veclong64-${indx}-file
                    MACRO wcomp.cxx
                    MACROARG "\"${indx}\", \"veclong64.h\""
                    LABELS longtest io
                    FIXTURES_REQUIRED root-io-evolution-versions-veclong64-${indx}
                    FIXTURES_SETUP root-io-evolution-versions-veclong64-${indx}-file)
  list(APPEND fixtures_veclong64 root-io-evolution-versions-veclong64-${indx}-file)
endforeach()

# test uses long which defined different on Windows, therefore it fails
if(NOT MSVC OR win_broken_tests)
  foreach(indx RANGE 1 2)
    ROOTTEST_ADD_TEST(veclong64-${indx}-run
                      MACRO runVecLong.cxx
                      MACROARG "1, \"${indx}\""
                      LABELS longtest io
                      FIXTURES_REQUIRED ${fixtures_veclong64}
                      OUTREF VecLong-${indx}.ref)
  endforeach()
endif()

ROOTTEST_ADD_TEST(ROOT7500
                  MACRO execROOT7500.cxx+
                  COPY_TO_BUILDDIR file7500.root
                  OUTREF execROOT7500.ref
                  LABELS longtest io)

ROOTTEST_ADD_TEST(attmarker
                  COMMAND ${ROOT_hadd_CMD} -f result.root hsimple14.root hsimple20.root
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COPY_TO_BUILDDIR hsimple14.root hsimple20.root
                  OUTREF attmarker.ref
                  LABELS longtest io)
