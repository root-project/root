# Add defines.
add_definitions(-DClingWorkAroundUnloadingIOSTREAM)

ROOTTEST_COMPILE_MACRO(autoloadtest.C)
ROOTTEST_ADD_TEST(roottest-root-aclic-load-autoload
                  MACRO execautoload.C
                  OUTREF execautoload.ref
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(roottest-root-aclic-load-reload
                  MACRO assertReload.C)

ROOTTEST_ADD_TEST(roottest-root-aclic-load-linktest
                  MACRO execlinktest.C
                  OUTREF execlinktest.ref)

ROOTTEST_COMPILE_MACRO(MyClass.cxx)

ROOTTEST_ADD_TEST(roottest-root-aclic-load-autoloadNamespace
                  MACRO execautoloadNamespace.C
                  OUTREF execautoloadNamespace.ref
                  DEPENDS ${COMPILE_MACRO_TEST})
