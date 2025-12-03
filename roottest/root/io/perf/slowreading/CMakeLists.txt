
ROOTTEST_ADD_TEST(files
                  COMMAND ${CMAKE_COMMAND} -E echo "copy files"
                  COPY_TO_BUILDDIR rootfiles/DST_60.root
                                   rootfiles/DST_61.root
                                   rootfiles/DST_62.root
                  FIXTURES_SETUP root-io-perf-slowreading-files-fixture)

foreach(macro 60 61 62)
  ROOTTEST_COMPILE_MACRO(NuEvent_${macro}.cxx
                         FIXTURES_SETUP root-io-perf-slowreading-${macro}-fixture)

  if(macro EQUAL 60)
    # measure reading time and write file
    ROOTTEST_ADD_TEST(mark
                      MACRO Read.C
                      MACROARG "\"NuEvent_60_cxx\", \"DST_60.root\", 1"
                      FIXTURES_REQUIRED root-io-perf-slowreading-files-fixture
                                        root-io-perf-slowreading-60-fixture
                      FIXTURES_SETUP root-io-perf-slowreading-mark-fixture)
  endif()

  foreach(file 60 61 62)
    # in Makefile test was running twice if first time performance was too slow
    # one could instead run performance code several times and choose best value
    ROOTTEST_ADD_TEST(check-macro${macro}-file${file}
                      MACRO Read.C
                      MACROARG "\"NuEvent_${macro}_cxx\", \"DST_${file}.root\""
                      OUTREF Read.ref
                      FIXTURES_REQUIRED root-io-perf-slowreading-files-fixture
                                        root-io-perf-slowreading-${macro}-fixture
                                        root-io-perf-slowreading-mark-fixture)
  endforeach()

endforeach()
