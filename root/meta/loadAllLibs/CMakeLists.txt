foreach( t LoadAllLibs LoadAllLibsAZ LoadAllLibsZA)
  ROOTTEST_ADD_TEST(roottest-root-meta-${t} MACRO assert${t}.C FAILREGEX "Error in|rc = -1")
endforeach()
