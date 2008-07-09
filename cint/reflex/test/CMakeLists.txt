FIND_PACKAGE(CppUnit REQUIRED)
FIND_PACKAGE(PythonInterp REQUIRED)
FIND_PACKAGE(GCCXML REQUIRED)
FIND_PACKAGE(DL)

# add CppUnit and the current dirs to the include path
INCLUDE_DIRECTORIES(${CPPUNIT_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

# generate and build the dictionary for testDict1/Reflex.h
GENREFLEX(testDict1/Reflex.h GENREFLEX_TGT)
ADD_LIBRARY(test_ReflexRflx SHARED ${GENREFLEX_TGT})
TARGET_LINK_LIBRARIES(test_ReflexRflx Reflex)

# generate and build the dictionary for testDict2/Class2Dict.h
GENREFLEX(testDict2/Class2Dict.h GENREFLEX_TGT)
ADD_LIBRARY(test_Class2DictRflx SHARED ${GENREFLEX_TGT})
TARGET_LINK_LIBRARIES(test_Class2DictRflx Reflex)

#####################   tests     ####################

ADD_EXECUTABLE(test_Reflex_generate test_Reflex_generate.cxx)
TARGET_LINK_LIBRARIES(test_Reflex_generate test_ReflexRflx Reflex ${DL_LIBRARY})
ADD_TEST("test_Reflex_generate" test_Reflex_generate)

ADD_EXECUTABLE(test_Reflex_unit test_Reflex_unit.cxx)
TARGET_LINK_LIBRARIES(test_Reflex_unit Reflex ${CPPUNIT_LIBRARY})
ADD_TEST("test_Reflex_unit" test_Reflex_unit)

ADD_EXECUTABLE(test_ReflexBuilder_unit test_ReflexBuilder_unit.cxx)
TARGET_LINK_LIBRARIES(test_ReflexBuilder_unit Reflex ${CPPUNIT_LIBRARY})
ADD_TEST("test_ReflexBuilder_unit" test_ReflexBuilder_unit)

ADD_EXECUTABLE(test_Reflex_simple1 test_Reflex_simple1.cxx)
TARGET_LINK_LIBRARIES(test_Reflex_simple1 test_ReflexRflx Reflex ${CPPUNIT_LIBRARY} ${DL_LIBRARY})
ADD_TEST("test_Reflex_simple1" test_Reflex_simple1)

ADD_EXECUTABLE(test_Reflex_simple2 test_Reflex_simple2.cxx)
TARGET_LINK_LIBRARIES(test_Reflex_simple2 test_ReflexRflx Reflex ${CPPUNIT_LIBRARY} ${DL_LIBRARY})
ADD_TEST("test_Reflex_simple2" test_Reflex_simple2)

