ROOTTEST_GENERATE_REFLEX_DICTIONARY(templateDict
                                    Objects.h
                                    SELECTION selection.xml
                                    FIXTURES_SETUP root-meta-naming-18363-templateDict-fixture)

ROOTTEST_ADD_TEST(templateName
                  MACRO templateNameTest.C
                  OUTREF templateNameTest.ref
		            FIXTURES_REQUIRED root-meta-naming-18363-templateDict-fixture)
