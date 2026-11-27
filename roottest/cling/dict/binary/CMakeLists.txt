# ROOT-10232

file(MAKE_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/libsubdir)

ROOTTEST_GENERATE_EXECUTABLE(clingDictInBinary
                             main.cxx
                             ${CMAKE_CURRENT_BINARY_DIR}/libsubdir/bindict_dict.cxx
                             LIBRARIES Core
                             FIXTURES_SETUP clingDictInBinary-createExe)
# Next to the dictionary payload
set_target_properties(clingDictInBinary PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_BINARY_DIR}/libsubdir>)

add_custom_command(OUTPUT libsubdir/bindict_dict.cxx
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libsubdir
                   COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classes.h -s ${CMAKE_CURRENT_SOURCE_DIR}/selection.xml -o bindict_dict.cxx
                   )

if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(clingDictInBinary
                      COMMAND ${CMAKE_COMMAND} -E env "PATH=${CMAKE_CURRENT_BINARY_DIR}/libsubdir:$ENV{PATH}" clingDictInBinary${exesuffix}
                      OUTREF check_dict_in_binary.ref
                      LABELS roottest regression cling
                      FIXTURES_REQUIRED clingDictInBinary-createExe)
endif()

