ROOTTEST_GENERATE_REFLEX_DICTIONARY(IOWithoutDictionaries
                                    IOWithoutDictionaries.h
                                    SELECTION IOWithoutDictionaries_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-io-withoutDictionaries-IOWithoutDictionaries
                  MACRO execIOWithoutDictionaries.C
                  OUTREF execIOWithoutDictionaries.oref
                  ERRREF execIOWithoutDictionaries.eref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(RowWiseNoProxyWarning
                                    RowWiseNoProxyWarning.h
                                    SELECTION RowWiseNoProxyWarning_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-io-withoutDictionaries-RowWiseNoProxyWarning
                  MACRO execRowWiseNoProxyWarning.C
                  OUTREF execRowWiseNoProxyWarning.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
