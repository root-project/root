foreach( t LoadAllLibs LoadAllLibsAZ LoadAllLibsZA)
  ROOTTEST_ADD_TEST(${t} MACRO assert${t}.C FAILREGEX "Error in|rc = -1")
endforeach()
