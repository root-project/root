ROOTTEST_ADD_TEST(textcopy MACRO  textcopy.C
                           OUTREF textcopy.ref)

if(asimage)
  ROOTTEST_ADD_TEST(simpleImages MACRO simpleImages.C
                                OUTREF simpleImages.ref
                                ERREF simpleImages.eref)
endif()

if(TARGET Gui)
  ROOT_ADD_GTEST(testImageFormats testImageFormats.cxx LIBRARIES Gui)
endif()
