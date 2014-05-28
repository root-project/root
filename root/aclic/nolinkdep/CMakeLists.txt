# Tests are actually not activated, they are failing when run, but 
# the results are not validated so it claims the tests passed.

#ROOTTEST_ADD_TEST(root-aclic-nolinkdep-link    MACRO link.C\(1\))
#ROOTTEST_ADD_TEST(root-aclic-nolinkdep-single  MACRO single.C+  DEPENDS root-aclic-nolinkdep-link)
#ROOTTEST_ADD_TEST(root-aclic-nolinkdep-script1 MACRO script1.C+ DEPENDS root-aclic-nolinkdep-single)
#ROOTTEST_ADD_TEST(root-aclic-nolinkdep-script2 MACRO script2.C+ DEPENDS root-aclic-nolinkdep-single)
