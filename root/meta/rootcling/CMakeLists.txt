ROOTTEST_ADD_TESTDIRS()

# ROOT-8492

ROOTTEST_ADD_TEST(selectUnion
                  COMMAND ${ROOT_rootcling_CMD} -f NestedClassOff.Dict.cc ${CMAKE_CURRENT_SOURCE_DIR}/NestedClassOff.h ${CMAKE_CURRENT_SOURCE_DIR}/NestedClassOff.LinkDef.h
                  OUTREF selectUnion.ref)
