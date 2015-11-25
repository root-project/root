ROOTTEST_ADD_TEST(7695
                  COMMAND ${ROOT_rootcling_CMD} -f AliEveITSDigitsInfo_cling.cxx
                          ${CMAKE_CURRENT_SOURCE_DIR}/AliEveITSDigitsInfo.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/AliEveITSDigitsInfo_linkdef.h)
