set(NAMESGOOD int float16 double32 regular char
              short long longlong uchar ushort
              uint ulong ulonglong float double
              double32enough double32mantis
              float16enough float16mantis)
set(NAMESFAIL double32tooshort float16tooshort)

# first compile int to copy also float.h and float16.cxx
ROOTTEST_ADD_TEST(compile-int
                  MACRO compile.C
                  COPY_TO_BUILDDIR float16.h float16.cxx
                  MACROARG "\"int\""
                  LABELS longtest io
                  FIXTURES_SETUP root-io-evolution-int-compile)

foreach(_name ${NAMESGOOD} ${NAMESFAIL})
  if(NOT "${_name}" STREQUAL "int")
    ROOTTEST_ADD_TEST(compile-${_name}
                      MACRO compile.C
                      MACROARG "\"${_name}\""
                      LABELS longtest io
                      FIXTURES_REQUIRED root-io-evolution-int-compile
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

foreach(_name ${NAMESFAIL})
  set(failfiles "${failfiles} ${_name}.root")
endforeach()

foreach(_name ${NAMESGOOD} ${NAMESFAIL})
  ROOTTEST_ADD_TEST(read-${_name}
                    MACRO read.C
                    ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L compile.C\")"
                    MACROARG "\"${goodfiles}\", \"${failfiles}\", \"${_name}\""
                    LABELS longtest io
                    OUTREF read.ref
                    FIXTURES_REQUIRED ${fixtures})
endforeach()
