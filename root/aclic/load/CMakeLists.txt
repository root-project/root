# Add defines.
add_definitions(-DClingWorkAroundUnloadingIOSTREAM)

ROOT_COMPILE_MACRO(autoloadtest.C)
ROOT_COMPILE_MACRO(MyClass.cxx)

ROOT_ADD_ROOTTEST(root-aclic-load-autoload MACRO execautoload.C OUTREF execautoload.ref)
ROOT_ADD_ROOTTEST(root-aclic-load-reload   MACRO assertReload.C)
ROOT_ADD_ROOTTEST(root-aclic-load-linktest MACRO execlinktest.C OUTREF execlinktest.ref)
ROOT_ADD_ROOTTEST(root-aclic-load-autoloadNamespace MACRO execautoloadNamespace.C OUTREF execautoloadNamespace.ref)
