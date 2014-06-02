ROOTTEST_GENERATE_REFLEX_DICTIONARY(classesFindBackPCMNames_rflx classesFindBackPCMNames.h SELECTION classesFindBackPCMNames_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-rootPCMs-execFindBackPCMNames
                  MACRO execFindBackPCMNames.C
                  OUTREF execFindBackPCMNames.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
