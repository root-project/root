# Tests are actually not activated, they are failing when run, but 
# the results are not validated so it claims the tests passed.

#ROOT_ADD_ROOTTEST(root-aclic-nolinkdep-link    MACRO link.C\(1\))
#ROOT_ADD_ROOTTEST(root-aclic-nolinkdep-single  MACRO single.C+  DEPENDS root-aclic-nolinkdep-link)
#ROOT_ADD_ROOTTEST(root-aclic-nolinkdep-script1 MACRO script1.C+ DEPENDS root-aclic-nolinkdep-single)
#ROOT_ADD_ROOTTEST(root-aclic-nolinkdep-script2 MACRO script2.C+ DEPENDS root-aclic-nolinkdep-single)
