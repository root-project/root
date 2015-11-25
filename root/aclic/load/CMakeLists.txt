ROOTTEST_ADD_TEST(autoload
                  COPY_TO_BUILDDIR theRootmap.rootmap
                  MACRO execautoload.C
                  OUTREF execautoload.ref
                  DEPENDS autoloadtest.C)

ROOTTEST_ADD_TEST(reload
                  MACRO assertReload.C)

ROOTTEST_ADD_TEST(linktest
                  MACRO execlinktest.C
                  OUTREF execlinktest.ref)

ROOTTEST_ADD_TEST(autoloadNamespace
                  MACRO execautoloadNamespace.C
                  OUTREF execautoloadNamespace.ref
                  DEPENDS MyClass.cxx autoload)
