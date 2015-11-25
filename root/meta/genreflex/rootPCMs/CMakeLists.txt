ROOTTEST_GENERATE_REFLEX_DICTIONARY(classesFindBackPCMNames_rflx classesFindBackPCMNames.h SELECTION classesFindBackPCMNames_selection.xml OPTIONS -m pcm1.pcm -m pcm2.pcm -m pcm3.pcm -m pcm4.pcm -m pcm5.pcm)

ROOTTEST_ADD_TEST(execFindBackPCMNames
                  MACRO execFindBackPCMNames.C
                  OUTREF execFindBackPCMNames.ref
                  WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS ${GENERATE_REFLEX_TEST})
