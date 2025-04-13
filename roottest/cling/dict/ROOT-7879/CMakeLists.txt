# ROOT-7879
ROOTTEST_ADD_TEST(skipLinkDefDict
                  COMMAND ${ROOT_rootcling_CMD} -f libSkippingDict.cxx -rmf libSkipping.rootmap -rml libSkipping${libsuffix} ${CMAKE_CURRENT_SOURCE_DIR}/skippingHeader.h ${CMAKE_CURRENT_SOURCE_DIR}/skipLinkDef.h
                  OUTREF skipLinkDefDict.ref)

ROOTTEST_ADD_TEST(skipLinkDef
                  COMMAND ${grep_cmd} clingAutoload libSkippingDict.cxx
                  DEPENDS skipLinkDefDict
                  OUTREF skippingDict${ref_suffix})

ROOTTEST_ADD_TESTDIRS()
