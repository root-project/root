ROOTTEST_COMPILE_MACRO(TOffset.cxx
                       FIXTURES_SETUP root-aclic-offset-TOffset-fixture)

if(MSVC)
  SET(_ref_name offsetwin.ref)
else()
  SET(_ref_name offset.ref)
endif()

ROOTTEST_ADD_TEST(offset
                  MACRO runoffset.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L TOffset.cxx+\")"
                  OUTREF ${_ref_name}
                  FIXTURES_REQUIRED root-aclic-offset-TOffset-fixture)
