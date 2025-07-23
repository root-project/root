ROOTTEST_ADD_TEST(textcopy MACRO  textcopy.C
                           OUTREF textcopy.ref)
if(TARGET Gui)
  ROOT_ADD_GTEST(testImageFormats testImageFormats.cxx LIBRARIES Gui)
endif()
