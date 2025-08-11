ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes
                                    classes.h
                                    SELECTION classes_selection.xml
                                    FIXTURES_SETUP root-meta-iotypeioname-dict-fixture)

ROOTTEST_ADD_TEST(ionameiotype
                  MACRO execionameiotype.C
                  COPY_TO_BUILDDIR example.xml
                  OUTREF execionameiotype.ref
                  FIXTURES_REQUIRED root-meta-iotypeioname-dict-fixture)
