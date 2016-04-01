ROOTTEST_ADD_TESTDIRS()

ROOTTEST_ADD_TEST(customDeleteError
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classesError1.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/selError1.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  ERRREF  customDeleteError.eref
                  PASSRC 1)

ROOTTEST_ADD_TEST(vectorOfUniquePointersError
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classesError2.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/selError2.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  ERRREF  vectorOfUniquePointersError.eref
                  PASSRC 1)

ROOTTEST_ADD_TEST(simpleRead
                  MACRO  rw.C
                  MACROARG "0"
                  OUTREF simpleRead.ref
                  DEPENDS simpleWriteRead)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h  SELECTION sel.xml)

ROOTTEST_ADD_TEST(simpleWriteRead
                  MACRO  rw.C
                  OUTREF simpleWriteRead.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
