# Cross-compatibility of unique_ptr and ptr

ROOTTEST_ADD_TEST(writeUniqPtrColls
                  MACRO writeUniqPtrColls.C+
                  FIXTURES_SETUP root-io-uniquePointer-writeUniqPtrColls-fixture)

ROOTTEST_ADD_TEST(writePtrColls
                  MACRO writePtrColls.C+
                  FIXTURES_SETUP root-io-uniquePointer-writePtrColls-fixture)

ROOTTEST_COMPILE_MACRO(readUniqPtrColls.C
                       FIXTURES_SETUP root-io-uniquePointer-readUniqPtrColls-compile-fixture)

ROOTTEST_ADD_TEST(readUniqPtrColls
                  MACRO readUniqPtrColls.C+
                  FIXTURES_REQUIRED root-io-uniquePointer-readUniqPtrColls-compile-fixture
                                    root-io-uniquePointer-writeUniqPtrColls-fixture)

ROOTTEST_COMPILE_MACRO(readPtrColls.C
                       FIXTURES_SETUP root-io-uniquePointer-readPtrColls-compile-fixture)

ROOTTEST_ADD_TEST(readPtrColls
                  MACRO readPtrColls.C+
                  FIXTURES_REQUIRED root-io-uniquePointer-readPtrColls-compile-fixture
                                    root-io-uniquePointer-writePtrColls-fixture)

ROOTTEST_ADD_TEST(crossReadPtrColls
                  MACRO readPtrColls.C+
                  MACROARG "\"uniqPtrColls.root\""
                  FIXTURES_REQUIRED root-io-uniquePointer-readPtrColls-compile-fixture
                                    root-io-uniquePointer-writeUniqPtrColls-fixture)

ROOTTEST_ADD_TEST(crossReadUniqPtrColls
                  MACRO readUniqPtrColls.C+
                  MACROARG "\"ptrColls.root\""
                  FIXTURES_REQUIRED root-io-uniquePointer-readUniqPtrColls-compile-fixture
                                    root-io-uniquePointer-writePtrColls-fixture)

# END Cross-compatibility of unique_ptr and ptr

ROOTTEST_ADD_TEST(checkNamesForIO
                  MACRO checkNameForIO.C)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes3 classes3.h  SELECTION sel3.xml)

ROOTTEST_ADD_TEST(checkDictionaries
                  MACRO checkDictionaries.C
                  ${WILLFAIL_ON_WIN32}
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(aclicDataMemberSelection
                  MACRO aclic01.C+
                  OUTREF aclic01.ref)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes2 classes2.h  SELECTION sel2.xml)

ROOTTEST_ADD_TEST(streamerInfoError
                  MACRO streamerInfoError.C
                  OUTREF streamerInfoError.eref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(customDeleteError
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classesError1.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/selError1.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                          -o ${CMAKE_CURRENT_BINARY_DIR}/classesError1_rflx.cpp
                  ERRREF customDeleteError.eref
                  PASSRC 1)

ROOTTEST_ADD_TEST(vectorOfUniquePointers
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classesError2.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/selError2.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                          -o ${CMAKE_CURRENT_BINARY_DIR}/classesError2_rflx.cpp)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h  SELECTION sel.xml)

ROOTTEST_ADD_TEST(simpleWriteRead
                  MACRO rw.C
                  OUTREF simpleWriteRead.ref
                  DEPENDS ${GENERATE_REFLEX_TEST}
                  FIXTURES_SETUP root-io-uniquePointer-simpleWriteRead-fixture)

ROOTTEST_ADD_TEST(simpleRead
                  MACRO rw.C
                  MACROARG "0"
                  OUTREF simpleRead.ref
                  FIXTURES_REQUIRED root-io-uniquePointer-simpleWriteRead-fixture)

ROOTTEST_ADD_TEST(simpleWriteReadConst
                  MACRO  rwconst.C
                  OUTREF simpleWriteReadconst.ref
                  DEPENDS ${GENERATE_REFLEX_TEST}
                  FIXTURES_SETUP root-io-uniquePointer-simpleWriteReadConst-fixture)

ROOTTEST_ADD_TEST(simpleReadConst
                  MACRO  rwconst.C
                  MACROARG "0"
                  OUTREF simpleReadconst.ref
                  FIXTURES_REQUIRED root-io-uniquePointer-simpleWriteReadConst-fixture)

