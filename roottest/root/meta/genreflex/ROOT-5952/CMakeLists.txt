ROOTTEST_ADD_TEST(5952
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classes.h --verbose
                           --select=${CMAKE_CURRENT_SOURCE_DIR}/classes_selection.xml
                           --output=${CMAKE_CURRENT_BINARY_DIR}/classes_rflx.cxx
                  ${WILLFAIL_ON_WIN32}
                  OUTREF selLog.ref)
