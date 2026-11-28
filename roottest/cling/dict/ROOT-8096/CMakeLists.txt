# ROOT-8096

ROOTTEST_GENERATE_REFLEX_DICTIONARY(dict definitions.h
                                    SELECTION selection.xml
                                    LIBNAME libROOT-8096
                                    NO_ROOTMAP
                                    OPTIONS --rootmap=libROOT-8096.rootmap --rootmap-lib=libROOT-8096${libsuffix})

ROOTTEST_ADD_TEST(load
                  MACRO load.C
                  OUTREF load.ref
                  FIXTURES_REQUIRED ${GENERATE_REFLEX_TEST}-fixture)
