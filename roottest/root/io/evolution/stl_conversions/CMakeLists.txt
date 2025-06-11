set(NAMESGOOD map vector multimap list)

# first compile map and copy necessary files
ROOTTEST_ADD_TEST(compile-map
                  MACRO compile.C
                  COPY_TO_BUILDDIR maptovector.h maptovector.cxx
                  MACROARG "\"map\""
                  LABELS longtest io
                  FIXTURES_SETUP root-io-evolution-map-compile)

foreach(_name ${NAMESGOOD})
  if(NOT "${_name}" STREQUAL "map")
    ROOTTEST_ADD_TEST(compile-${_name}
                      MACRO compile.C
                      MACROARG "\"${_name}\""
                      LABELS longtest io
                      FIXTURES_REQUIRED root-io-evolution-map-compile
                      FIXTURES_SETUP root-io-evolution-${_name}-compile)
  endif()
  ROOTTEST_ADD_TEST(write-${_name}
                    MACRO run.C
                    ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L compile.C\")"
                    MACROARG "\"${_name}\""
                    LABELS longtest io
                    FIXTURES_REQUIRED root-io-evolution-${_name}-compile
                    FIXTURES_SETUP root-io-evolution-${_name}-write)
  list(APPEND fixtures root-io-evolution-${_name}-write)
endforeach()

foreach(_name ${NAMESGOOD})
  set(goodfiles "${goodfiles} ${_name}.root")
endforeach()

foreach(_name ${NAMESGOOD})
  ROOTTEST_ADD_TEST(read-${_name}
                    MACRO read.C
                    ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L compile.C\")"
                    MACROARG "\"${goodfiles}\", \"${_name}\""
                    LABELS longtest io
                    OUTREF check.stl.ref
                    FIXTURES_REQUIRED ${fixtures})
endforeach()
