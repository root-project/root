ROOTTEST_GENERATE_REFLEX_DICTIONARY(templateDict Objects.h SELECTION selection.xml FIXTURES_SETUP template_lib)

ROOTTEST_ADD_TEST(templateName
                  MACRO templateNameTest.C
                  OUTREF templateNameTest.ref
		  FIXTURES_REQUIRED template_lib)
