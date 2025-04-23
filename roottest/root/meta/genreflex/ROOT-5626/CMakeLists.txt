#ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes_rflx classes.h SELECTION classes_selection.xml)

#ROOTTEST_ADD_TEST(execdummy
#                  MACRO execdummy.C
#                  OUTREF execdummy.ref
#                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(5626
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classes.h -o  classes.cxx
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/classes_selection.xml -I ${CMAKE_CURRENT_SOURCE_DIR})
