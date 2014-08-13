#ROOTTEST_GENERATE_REFLEX_DICTIONARY(AliaODPid AliAODPid.h SELECTION AliAODPid_selection.xml)
#
#ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-ROOT-5594-execdummy
#                  MACRO execdummy.C
#                  OUTREF execdummy.ref
#                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-ROOT-5594
                  COMMAND ${ROOT_genreflex_cmd} ${CMAKE_CURRENT_SOURCE_DIR}/AliAODPid.h
                               -o  AliAODPid.cxx 
                               --select=${CMAKE_CURRENT_SOURCE_DIR}/AliAODPid_selection.xml)