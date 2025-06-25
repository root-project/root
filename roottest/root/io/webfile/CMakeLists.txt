if(geom AND davix)
  ROOTTEST_ADD_TEST(CloseTWebFile
                    MACRO runCloseTWebFile.C
                    OUTREF CloseTWebFile.ref)
endif()
