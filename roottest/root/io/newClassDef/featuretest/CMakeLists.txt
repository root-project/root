ROOTTEST_COMPILE_MACRO(typeidtest.C
                       FIXTURES_SETUP root-io-newClassDef-featuretest-typeidtest-fixture)

ROOTTEST_COMPILE_MACRO(Class.C
                       FIXTURES_SETUP root-io-newClassDef-featuretest-Class-fixture)

ROOTTEST_COMPILE_MACRO(ClassTrick.C
                       FIXTURES_SETUP root-io-newClassDef-featuretest-ClassTrick-fixture
                       FIXTURES_REQUIRED root-io-newClassDef-featuretest-Class-fixture)

# in the ClassTrick.C commants stated that
# trick does not works on Windows.
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(run
                    MACRO Run.C
                    OUTREF run.ref
                    LABELS longtest
                    FIXTURES_REQUIRED root-io-newClassDef-featuretest-typeidtest-fixture
                                      root-io-newClassDef-featuretest-Class-fixture
                                      root-io-newClassDef-featuretest-ClassTrick-fixture)
endif()
