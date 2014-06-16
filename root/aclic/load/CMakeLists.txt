ROOTTEST_ADD_TEST(roottest-root-aclic-load-autoload
                  MACRO execautoload.C
                  OUTREF execautoload.ref
                  DEPENDS autoloadtest.C)

ROOTTEST_ADD_TEST(roottest-root-aclic-load-reload
                  MACRO assertReload.C)

ROOTTEST_ADD_TEST(roottest-root-aclic-load-linktest
                  MACRO execlinktest.C
                  OUTREF execlinktest.ref)

ROOTTEST_ADD_TEST(roottest-root-aclic-load-autoloadNamespace
                  MACRO execautoloadNamespace.C
                  OUTREF execautoloadNamespace.ref
                  DEPENDS MyClass.cxx)
