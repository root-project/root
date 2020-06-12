# ROOT-10232

file(MAKE_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/libsubdir)

ROOTTEST_GENERATE_EXECUTABLE(clingDictInBinary
                             main.cxx
                             ${CMAKE_CURRENT_BINARY_DIR}/libsubdir/bindict_dict.cxx
                             LIBRARIES Core)

add_custom_command(OUTPUT libsubdir/bindict_dict.cxx
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libsubdir
                   COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classes.h -s ${CMAKE_CURRENT_SOURCE_DIR}/selection.xml -o bindict_dict.cxx
                   )

add_custom_command(TARGET clingDictInBinary
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/clingDictInBinary${exesuffix} ${CMAKE_CURRENT_BINARY_DIR}/libsubdir/clingDictInBinary${exesuffix}
                   COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/clingDictInBinary${exesuffix}
                   )

if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(clingDictInBinary
                      COMMAND ${CMAKE_COMMAND} -E env "PATH=${CMAKE_CURRENT_BINARY_DIR}/libsubdir:$ENV{PATH}" clingDictInBinary${exesuffix}
                      OUTREF check_dict_in_binary.ref
                      LABELS roottest regression cling
                      DEPENDS ${GENERATE_EXECUTABLE_TEST})
endif()

