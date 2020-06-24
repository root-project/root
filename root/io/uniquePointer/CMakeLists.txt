ROOTTEST_ADD_TESTDIRS()

# Cross-compatibility of unique_ptr and ptr

ROOTTEST_ADD_TEST(writeUniqPtrColls
                  MACRO writeUniqPtrColls.C+)

ROOTTEST_ADD_TEST(writePtrColls
                  MACRO writePtrColls.C+)

ROOTTEST_ADD_TEST(readUniqPtrColls
                  MACRO readUniqPtrColls.C+
                  DEPENDS writeUniqPtrColls)

ROOTTEST_ADD_TEST(readPtrColls
                  MACRO readPtrColls.C+
                  DEPENDS writePtrColls)

ROOTTEST_ADD_TEST(crossReadPtrColls
                  MACRO readPtrColls.C+
                  MACROARG "\"uniqPtrColls.root\""
                  DEPENDS readPtrColls)

ROOTTEST_ADD_TEST(crossReadUniqPtrColls
                  MACRO readUniqPtrColls.C+
                  MACROARG "\"ptrColls.root\""
                  DEPENDS readUniqPtrColls)

# END Cross-compatibility of unique_ptr and ptr

ROOTTEST_ADD_TEST(checkNamesForIO
                  MACRO  checkNameForIO.C)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes3 classes3.h  SELECTION sel3.xml)

ROOTTEST_ADD_TEST(checkDictionaries
                  MACRO  checkDictionaries.C
                  ${WILLFAIL_ON_WIN32}
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(aclicDataMemberSelection
                  MACRO  aclic01.C+
                  OUTREF aclic01.ref)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes2 classes2.h  SELECTION sel2.xml)

ROOTTEST_ADD_TEST(streamerInfoError
                  MACRO  streamerInfoError.C
                  OUTREF streamerInfoError.eref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(customDeleteError
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classesError1.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/selError1.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  ERRREF  customDeleteError.eref
                  PASSRC 1)

ROOTTEST_ADD_TEST(vectorOfUniquePointers
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classesError2.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/selError2.xml -I ${CMAKE_CURRENT_SOURCE_DIR})

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

ROOTTEST_ADD_TEST(simpleReadConst
                  MACRO  rwconst.C
                  MACROARG "0"
                  OUTREF simpleReadconst.ref
                  DEPENDS simpleWriteReadConst)

ROOTTEST_ADD_TEST(simpleWriteReadConst
                  MACRO  rwconst.C
                  OUTREF simpleWriteReadconst.ref
                  DEPENDS simpleWriteRead)