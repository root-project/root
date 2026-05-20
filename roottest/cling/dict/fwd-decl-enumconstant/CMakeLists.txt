ROOTTEST_GENERATE_REFLEX_DICTIONARY(fwdDeclEnumconstantDict
  OPTIONS --fail_on_warnings
  SELECTION selection.xml
  classes.h 
  )

ROOTTEST_ADD_TEST(fwdDeclEnumconstant
                  MACRO execfwdDeclEnumconstant.C
                  DEPENDS fwdDeclEnumconstantDict-libgen-build
                  LABELS roottest regression cling)
