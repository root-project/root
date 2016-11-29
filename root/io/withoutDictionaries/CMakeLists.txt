# ROOT-8478
ROOTTEST_GENERATE_REFLEX_DICTIONARY(VectorDMWriteWithoutDictionary
                                    VectorDMWriteWithoutDictionary.h
                                    SELECTION VectorDMWriteWithoutDictionary_selection.xml)

ROOTTEST_ADD_TEST(VectorDMWriteWithoutDictionary
                  OUTREF execVectorDMWriteWithoutDictionary.ref
                  MACRO execVectorDMWriteWithoutDictionary.C
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(CheckMissingDictionaries
                  MACRO checkMissingDictionaries.C
                  DEPENDS VectorDMWriteWithoutDictionary)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(IOWithoutDictionaries
                                    IOWithoutDictionaries.h
                                    SELECTION IOWithoutDictionaries_selection.xml)

ROOTTEST_ADD_TEST(IOWithoutDictionaries
                  MACRO execIOWithoutDictionaries.C
                  OUTREF execIOWithoutDictionaries.oref
                  ERRREF execIOWithoutDictionaries.eref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(RowWiseNoProxyWarning
                                    RowWiseNoProxyWarning.h
                                    SELECTION RowWiseNoProxyWarning_selection.xml)

ROOTTEST_ADD_TEST(RowWiseNoProxyWarning
                  MACRO execRowWiseNoProxyWarning.C
                  OUTREF execRowWiseNoProxyWarning.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
