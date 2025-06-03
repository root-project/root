ROOTTEST_GENERATE_DICTIONARY(customConstructor
                             customConstructor.C
                             LINKDEF linkdef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP customConstructor_fixture)

if(ClingReinstateTemplateRootIOCtor)
  set(testref customConstructor.ref)
else()
  set(testref customConstructor.ref-notemplateCtor)
endif()

ROOTTEST_ADD_TEST(customConstructorLib
                  MACRO runcustomConstructor.C
                  OUTREF ${testref}
                  FIXTURES_REQUIRED customConstructor_fixture)

ROOTTEST_ADD_TEST(BuildCustomLoader
                  MACRO ${CMAKE_SOURCE_DIR}/roottest/scripts/build.C
                  MACROARG "\"customLoader.C\""
                  COPY_TO_BUILDDIR customLoader.C customConstructor.C linkdef.h
                  FIXTURES_SETUP BuildCustomLoader-fixture)

# add dependency from dictionary lib to prevent running ACLiC in parallel
ROOTTEST_ADD_TEST(customConstructorACLiC
                  MACRO runcustomConstructorACLiC.C
                  OUTREF customConstructorACLiC.ref
                  FIXTURES_REQUIRED BuildCustomLoader-fixture)
