ROOTTEST_GENERATE_REFLEX_DICTIONARY(AliaODPid AliAODPid.h SELECTION AliAODPid_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-ROOT-5594-execdummy
                  MACRO execdummy.C OUTREF execdummy.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
