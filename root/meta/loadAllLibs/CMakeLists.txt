
if(ROOTTEST_OS_ID MATCHES Scientific OR ROOTTEST_OS_ID MATCHES CentOS)
  set(ROOTTEST_ENV_EXTRA LD_PRELOAD=/usr/lib64/libGL.so)
endif()

foreach( t LoadAllLibs LoadAllLibsAZ LoadAllLibsZA)
  ROOTTEST_ADD_TEST(${t} MACRO assert${t}.C FAILREGEX "Error in|rc = -1")
endforeach()
