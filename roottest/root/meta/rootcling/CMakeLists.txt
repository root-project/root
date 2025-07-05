ROOTTEST_ADD_TESTDIRS()

# issue #13543
ROOTTEST_GENERATE_DICTIONARY(xmlusedByRootclingDict
                             Classy.h
                             LINKDEF xmlusedByRootcling_sel.xml
                             FIXTURES_SETUP root-meta-roocling-xmlusedByRootclingDict-fixture)

ROOTTEST_ADD_TEST(xmlusedByRootcling
                  MACRO xmlusedByRootclingDict.C
                  FIXTURES_REQUIRED root-meta-roocling-xmlusedByRootclingDict-fixture)

# ROOT-9335
ROOTTEST_ADD_TEST(classDelCtor
                  MACRO classDelCtor.C+)

# ROOT-7765
ROOTTEST_GENERATE_DICTIONARY(includeInLinkdefDict
                             LINKDEF classInLinkdef.h
                             FIXTURES_SETUP root-meta-roocling-includeInLinkdefDict-fixture)

ROOTTEST_ADD_TEST(includeInLinkdef
                  MACRO includeInLinkdef.C
                  COPY_TO_BUILDDIR headerForLdef.h
                  FIXTURES_REQUIRED root-meta-roocling-includeInLinkdefDict-fixture)

# ROOT-8492
ROOTTEST_ADD_TEST(selectUnion
                  COMMAND ${ROOT_rootcling_CMD} -f NestedClassOff.Dict.cc ${CMAKE_CURRENT_SOURCE_DIR}/NestedClassOff.h ${CMAKE_CURRENT_SOURCE_DIR}/NestedClassOff.LinkDef.h
                  OUTREF selectUnion.ref
                  PASSRC 1)

# ROOT-10798
ROOTTEST_ADD_TEST(ROOT10798
                  COMMAND ${ROOT_rootcling_CMD} -f ROOT10798Dict.cxx ${CMAKE_CURRENT_SOURCE_DIR}/ROOT10798LinkDef.h)

# Issue #18833
ROOTTEST_GENERATE_DICTIONARY(streamerInfoStdFunctionDict
                             streamerInfoStdFunction.h
                             LINKDEF streamerInfoStdFunction.xml
                             FIXTURES_SETUP root-meta-roocling-streamerInfoStdFunctionDict-fixture)

ROOTTEST_ADD_TEST(streamerInfoStdFunction
                  MACRO streamerInfoStdFunction.C
                  FIXTURES_REQUIRED root-meta-roocling-streamerInfoStdFunctionDict-fixture)

# Issue #18982
ROOTTEST_ADD_TEST(selectTemplateInvalidArg
                  COMMAND ${ROOT_rootcling_CMD} -f selectTemplateInvalidArg.Dict.cc ${CMAKE_CURRENT_SOURCE_DIR}/selectTemplateInvalidArg.h ${CMAKE_CURRENT_SOURCE_DIR}/selectTemplateInvalidArg.LinkDef.h
                  ERRREF selectTemplateInvalidArg.ref)
