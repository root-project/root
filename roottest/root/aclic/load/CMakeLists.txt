ROOTTEST_ADD_TEST(autoload
                  COPY_TO_BUILDDIR theRootmap.rootmap
                  MACRO execautoload.C
                  OUTREF execautoload.ref
                  DEPENDS autoloadtest.C
                  FIXTURES_SETUP root-aclic-load-autoload-fixture)

if(NOT coverage)
    ROOTTEST_ADD_TEST(reload
                      COPY_TO_BUILDDIR Reload.root
                      MACRO assertReload.C
                      OUTREF assertReload.ref)
endif()

ROOTTEST_ADD_TEST(linktest
                  MACRO execlinktest.C
                  OUTREF execlinktest.ref)

if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(autoloadNamespace
                      MACRO execautoloadNamespace.C
                      OUTREF execautoloadNamespace.ref
                      DEPENDS MyClass.cxx
                      FIXTURES_REQUIRED root-aclic-load-autoload-fixture)
endif()
