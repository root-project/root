ROOTTEST_COMPILE_MACRO(${CMAKE_SOURCE_DIR}/roottest/scripts/utils.cc
                       FIXTURES_SETUP root-treedraw-utils-fixture)

ROOTTEST_ADD_TEST(create
                  MACRO ${CMAKE_SOURCE_DIR}/tutorials/hsimple.C
                  PASSRC 255
                  FIXTURES_SETUP root-treedraw-create-fixture)

ROOTTEST_ADD_TEST(simple
                  MACRO runsimple.C
                  ROOTEXE_OPTS -e "(void)gSystem->Load(\"utils_cc\")"
                  COPY_TO_BUILDDIR hsimple.ps.ref
                  OUTREF simple.ref
                  FIXTURES_REQUIRED root-treedraw-utils-fixture root-treedraw-create-fixture)
