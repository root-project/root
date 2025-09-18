ROOTTEST_ADD_TESTDIRS()

ROOTTEST_GENERATE_DICTIONARY(testDict
                             test.h
                             LINKDEF testLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_COMPILE_MACRO(cltest.C
                       FIXTURES_SETUP root-io-rootcint-cltest-fixture)

ROOTTEST_ADD_TEST(cltest
                  MACRO cltest.C+
                  OUTREF cltest.ref
                  FIXTURES_REQUIRED root-io-rootcint-cltest-fixture)

ROOTTEST_ADD_TEST(TemplateExpr
                  MACRO execTemplateExpr.C
                  OUTREF execTemplateExpr.ref)

ROOTTEST_COMPILE_MACRO(virtualbase.h
                       FIXTURES_SETUP root-io-rootcint-virtualbase-fixture)

ROOTTEST_ADD_TEST(nestedtemplate
                  MACRO runnestedtemplate.C
                  OUTREF nestedtemplate.ref)

ROOTTEST_COMPILE_MACRO(templateKeyword.C
                       FIXTURES_SETUP root-io-rootcint-templateKeyword-fixture)

ROOTTEST_ADD_TEST(templateKeyword
                  MACRO runtemplateKeyword.C
                  OUTREF templateKeyword.ref
                  FIXTURES_REQUIRED root-io-rootcint-templateKeyword-fixture)

ROOTTEST_ADD_TEST(virtualbase
                  MACRO runvirtualbase.C
                  OUTREF virtualbase.ref
                  FIXTURES_REQUIRED root-io-rootcint-virtualbase-fixture)

ROOTTEST_GENERATE_DICTIONARY(weirdpair
                             weirdpair.h
                             LINKDEF weirdpair_LinkDef.h
                             NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_GENERATE_DICTIONARY(foo
                             foo.hh
                             LINKDEF foo_linkdef.h
                             NO_ROOTMAP NO_CXXMODULE)
