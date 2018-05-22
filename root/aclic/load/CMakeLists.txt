ROOTTEST_ADD_TEST(autoload
                  COPY_TO_BUILDDIR theRootmap.rootmap
                  MACRO execautoload.C
                  OUTREF execautoload.ref
                  DEPENDS autoloadtest.C)

# FIXME: Temporary workaround for runtime_cxxmodules
if(NOT ROOT_runtime_cxxmodules_FOUND)
ROOTTEST_ADD_TEST(reload
                  COPY_TO_BUILDDIR Reload.root
                  MACRO assertReload.C
                  OUTREF assertReload.ref)
endif()

ROOTTEST_ADD_TEST(linktest
                  MACRO execlinktest.C
                  OUTREF execlinktest.ref)

ROOTTEST_ADD_TEST(autoloadNamespace
                  MACRO execautoloadNamespace.C
                  OUTREF execautoloadNamespace.ref
                  DEPENDS MyClass.cxx autoload)
