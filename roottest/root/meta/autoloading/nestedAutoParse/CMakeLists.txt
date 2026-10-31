ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedAutoParse
                                    nestedAutoParse.h
                                    SELECTION selection.xml
                                    LIBNAME libnestedAutoParse_dictrflx
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-autoloading-nestedAutoParse-fixture)

ROOTTEST_ADD_TEST(nestedAutoParse
                  MACRO execNestedAutoParse.C
                  COPY_TO_BUILDDIR useNestedAutoParse.C
                  FIXTURES_REQUIRED root-meta-autoloading-nestedAutoParse-fixture)
