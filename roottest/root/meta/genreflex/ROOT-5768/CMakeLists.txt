# windows fails with message
# PyCool_rflx.cxx(824,1): error C2026: string too big, trailing characters truncated

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(PyCool
                    COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/PyCool.h
                            --interpreteronly -I ${CMAKE_CURRENT_SOURCE_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR}/CoralBase
                            --select=${CMAKE_CURRENT_SOURCE_DIR}/PyCool_selection.xml
                            --output=${CMAKE_CURRENT_BINARY_DIR}/PyCool_rflx.cxx
                    FIXTURES_SETUP PyCool-reflex-dict)


  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/PyCool_rflx.cxx
                     COMMAND ${CMAKE_COMMAND} -E echo "do nothing, dictionary should be generated already")


  # build static library only to ensure that generated dictionary can be compiled
  ROOTTEST_LINKER_LIBRARY(PyCoolLib
                          TEST
                          ${CMAKE_CURRENT_BINARY_DIR}/PyCool_rflx.cxx
                          TYPE STATIC
                          LIBRARIES ${ROOT_LIBRARIES})

  target_include_directories(PyCoolLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  # target_include_directories(PyCoolLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/CoralBase)

  ROOTTEST_ADD_TEST(PyCoolLib-build
                    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} ${build_config} --target PyCoolLib${fast} -- ${always-make}
                    FIXTURES_REQUIRED PyCool-reflex-dict)
endif()