ROOTTEST_ADD_TEST(execROOT8804
                  MACRO execROOT8804.C
                  COPY_TO_BUILDDIR lariat-si.root
                  OUTREF execROOT8804.ref)

ROOTTEST_COMPILE_MACRO(classes.h
                       FIXTURES_SETUP root-io-emulated-classes-fixtures)

ROOTTEST_COMPILE_MACRO(marker.h
                       FIXTURES_SETUP root-io-emulated-marker-fixtures)

if(ClingWorkAroundMissingSmartInclude)
  set(_opts_classes -e "(void)gROOT->ProcessLine(\".L classes.h+\")")
  set(_opts_marker -e "(void)gROOT->ProcessLine(\".L marker.h+\")")
endif()

ROOTTEST_ADD_TEST(write
                  MACRO write.C
                  ROOTEXE_OPTS ${_opts_classes} ${_opts_marker}
                  FIXTURES_REQUIRED root-io-emulated-classes-fixtures root-io-emulated-marker-fixtures
                  FIXTURES_SETUP root-io-emulated-write-fixtures)

ROOTTEST_ADD_TEST(read
                  MACRO read.C
                  OUTREF read.ref
                  ROOTEXE_OPTS ${_opts_marker}
                  FIXTURES_REQUIRED root-io-emulated-marker-fixtures root-io-emulated-write-fixtures
                  FIXTURES_SETUP root-io-emulated-read-fixtures)

ROOTTEST_ADD_TEST(rewrite
                  MACRO rewrite.C
                  OUTREF rewrite.ref
                  ROOTEXE_OPTS ${_opts_marker}
                  FIXTURES_REQUIRED root-io-emulated-marker-fixtures root-io-emulated-write-fixtures)

ROOTTEST_ADD_TEST(writeAbstract
                  MACRO writeAbstract.cxx+
                  FIXTURES_SETUP root-io-emulated-writeAbstract-fixtures)

ROOTTEST_ADD_TEST(readAbstract
                  MACRO readAbstract.cxx+
                  OUTREF readAbstract.ref
                  FIXTURES_REQUIRED root-io-emulated-writeAbstract-fixtures)
