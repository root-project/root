ROOTTEST_COMPILE_MACRO(autoloadtest.C
                       FIXTURES_SETUP root-aclic-load-autoloadtest-fixture)

ROOTTEST_ADD_TEST(autoload
                  MACRO execautoload.C
                  OUTREF execautoload.ref
                  FIXTURES_REQUIRED root-aclic-load-autoloadtest-fixture)

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
   ROOTTEST_COMPILE_MACRO(MyClass.cxx
                          FIXTURES_SETUP root-aclic-load-MyClass-fixture)

   ROOTTEST_ADD_TEST(autoloadNamespace
                     MACRO execautoloadNamespace.C
                     OUTREF execautoloadNamespace.ref
                     COPY_TO_BUILDDIR theRootmap.rootmap
                     FIXTURES_REQUIRED root-aclic-load-MyClass-fixture)
endif()
