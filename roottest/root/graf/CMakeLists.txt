ROOTTEST_ADD_TEST(textcopy MACRO  textcopy.C
                           OUTREF textcopy.ref)

# Test mitigation for #14156
if(asimage)
    ROOTTEST_ADD_TEST(tasimage_frompad MACRO tasimage_frompad.C)
endif()
