ROOTTEST_COMPILE_MACRO(foo.C
                       FIXTURES_SETUP root-io-arrayobject-foo-fixture)

ROOTTEST_COMPILE_MACRO(bar.C
                       BUILDLIB foo_C
                       FIXTURES_REQUIRED root-io-arrayobject-foo-fixture
                       FIXTURES_SETUP root-io-arrayobject-bar-fixture)

ROOTTEST_COMPILE_MACRO(main.C
                       BUILDLIB bar_C
                       FIXTURES_REQUIRED root-io-arrayobject-bar-fixture
                       FIXTURES_SETUP root-io-arrayobject-main-fixture)

ROOTTEST_ADD_TEST(RealArraySave
                  MACRO RealArraySave.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L main.C+\")"
                  PASSRC 1
                  OUTREF RealArraySave.ref
                  FIXTURES_REQUIRED root-io-arrayobject-main-fixture
                  FIXTURES_SETUP root-io-arrayobject-write-fixture)

ROOTTEST_ADD_TEST(ReadBar
                  MACRO Run.C
                  MACROARG "\"Event.root\""
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L bar.C+\")"
                  OUTREF ReadBar.ref
                  COPY_TO_BUILDDIR Event.3.2.0.root
                  FIXTURES_REQUIRED root-io-arrayobject-bar-fixture
                                    root-io-arrayobject-write-fixture)

# reading old version of bar class fails - while class not correctly equipt
ROOTTEST_ADD_TEST(ReadBarOld
                  MACRO Run.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L bar.C+\")"
                  OUTREF ReadBarOld.ref
                  COPY_TO_BUILDDIR Event.3.2.0.root
                  WILLFAIL
                  FIXTURES_REQUIRED root-io-arrayobject-bar-fixture)
