#####################   sub dirs  ####################

ADD_SUBDIRECTORY(testDict1)
ADD_SUBDIRECTORY(testDict2)

#####################    tests    ####################

INCLUDE_DIRECTORIES(${REFLEX_INCLUDE_DIR} ${CPPUNIT_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

REFLEX_ADD_EXECUTABLE(test_Reflex_generate TEST test_Reflex_generate.cxx)
TARGET_LINK_LIBRARIES(test_Reflex_generate Reflex ${DL_LIBRARY})
REFLEX_ADD_TEST("test_Reflex_generate" test_Reflex_generate
                DICTIONARIES test_Class2DictRflx)

REFLEX_ADD_EXECUTABLE(test_Reflex_unit TEST test_Reflex_unit.cxx)
TARGET_LINK_LIBRARIES(test_Reflex_unit Reflex ${CPPUNIT_LIBRARY})
REFLEX_ADD_TEST("test_Reflex_unit" test_Reflex_unit)

REFLEX_ADD_EXECUTABLE(test_ReflexBuilder_unit TEST test_ReflexBuilder_unit.cxx)
TARGET_LINK_LIBRARIES(test_ReflexBuilder_unit Reflex ${CPPUNIT_LIBRARY})
REFLEX_ADD_TEST("test_ReflexBuilder_unit" test_ReflexBuilder_unit)

REFLEX_ADD_EXECUTABLE(test_Reflex_simple1 TEST test_Reflex_simple1.cxx)
TARGET_LINK_LIBRARIES(test_Reflex_simple1 Reflex ${CPPUNIT_LIBRARY} ${DL_LIBRARY})
REFLEX_ADD_TEST("test_Reflex_simple1" test_Reflex_simple1
                DICTIONARIES test_ReflexRflx)

REFLEX_ADD_EXECUTABLE(test_Reflex_simple2 TEST test_Reflex_simple2.cxx)
TARGET_LINK_LIBRARIES(test_Reflex_simple2 Reflex ${CPPUNIT_LIBRARY} ${DL_LIBRARY})
REFLEX_ADD_TEST("test_Reflex_simple2" test_Reflex_simple2
                DICTIONARIES test_Class2DictRflx)
