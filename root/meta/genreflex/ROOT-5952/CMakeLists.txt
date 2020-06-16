ROOTTEST_ADD_TEST(5952
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classes.h --verbose --select=${CMAKE_CURRENT_SOURCE_DIR}/classes_selection.xml
                  ${MIGHT_FAIL}
                  OUTREF selLog.ref)
