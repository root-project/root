ROOTTEST_ADD_TEST(MakeProxy
                  MACRO createAliasStringMakeProxy.C
                  OUTREF createAliasStringMakeProxy.ref
                  COPY_TO_BUILDDIR test.txt scriptAliasString.C
                  FIXTURES_SETUP root-treeformula-string-MakeProxy-fixture)

ROOTTEST_ADD_TEST(AliasString
                  MACRO execAliasString.C
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF execAliasString.ref
                  FIXTURES_REQUIRED root-treeformula-string-MakeProxy-fixture)

ROOTTEST_ADD_TEST(cernbuild
                  MACRO ${CMAKE_SOURCE_DIR}/tutorials/io/tree/tree500_cernbuild.C
                  MACROARG 1,0
                  PASSRC 255
                  FIXTURES_SETUP root-treeformula-string-cernbuild-fixture)

ROOTTEST_ADD_TEST(DrawString
                  MACRO execDrawString.C
                  OUTREF execDrawString.ref
                  FIXTURES_REQUIRED root-treeformula-string-cernbuild-fixture)

ROOTTEST_COMPILE_MACRO(runstring.C
                       FIXTURES_SETUP root-treeformula-string-runstring_C-fixture)

ROOTTEST_ADD_TEST(string
                  MACRO runstring.C
                  OUTREF string.ref
                  OUTCNVCMD sed -e s:0x[0-9a-fA-F]*:0xRemoved:g
                  FIXTURES_REQUIRED root-treeformula-string-runstring_C-fixture)
