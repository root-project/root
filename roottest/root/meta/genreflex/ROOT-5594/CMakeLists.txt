#ROOTTEST_GENERATE_REFLEX_DICTIONARY(AliaODPid AliAODPid.h SELECTION AliAODPid_selection.xml)
#
#ROOTTEST_ADD_TEST(execdummy
#                  MACRO execdummy.C
#                  OUTREF execdummy.ref
#                  DEPENDS ${GENERATE_REFLEX_TEST})

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(5594
                    COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/AliAODPid.h
                                 -o  AliAODPid.cxx
                                 --select=${CMAKE_CURRENT_SOURCE_DIR}/AliAODPid_selection.xml)
endif()
