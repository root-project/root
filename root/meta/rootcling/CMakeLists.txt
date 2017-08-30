ROOTTEST_ADD_TESTDIRS()

# ROOT-7765
ROOTTEST_GENERATE_DICTIONARY(includeInLinkdefDict LINKDEF classInLinkdef.h)
ROOTTEST_ADD_TEST(includeInLinkdef
                  MACRO includeInLinkdef.C
                  COPY_TO_BUILDDIR headerForLdef.h
                  DEPENDS ${GENERATE_DICTIONARY_TEST})

# ROOT-8492
ROOTTEST_ADD_TEST(selectUnion
                  COMMAND ${ROOT_rootcling_CMD} -f NestedClassOff.Dict.cc ${CMAKE_CURRENT_SOURCE_DIR}/NestedClassOff.h ${CMAKE_CURRENT_SOURCE_DIR}/NestedClassOff.LinkDef.h
                  OUTREF selectUnion.ref
                  PASSRC 1)
