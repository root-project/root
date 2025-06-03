ROOTTEST_GENERATE_DICTIONARY(customConstructor
                             customConstructor.C
                             LINKDEF linkdef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-io-alloc-customConstructor-fixture)

if(ClingReinstateTemplateRootIOCtor)
  set(testref customConstructor.ref)
else()
  set(testref customConstructor.ref-notemplateCtor)
endif()

ROOTTEST_ADD_TEST(customConstructorLib
                  MACRO runcustomConstructor.C
                  OUTREF ${testref}
                  FIXTURES_REQUIRED root-io-alloc-customConstructor-fixture)

ROOTTEST_COMPILE_MACRO(customLoader.C
                       FIXTURES_SETUP root-io-alloc-customLoader-fixture)

# add dependency from dictionary lib to prevent running ACLiC in parallel
ROOTTEST_ADD_TEST(customConstructorACLiC
                  MACRO runcustomConstructorACLiC.C
                  OUTREF customConstructorACLiC.ref
                  FIXTURES_REQUIRED root-io-alloc-customLoader-fixture)
