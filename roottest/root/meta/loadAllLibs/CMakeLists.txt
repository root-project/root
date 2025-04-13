
if(OPENGL_gl_LIBRARY AND ROOTTEST_OS_ID MATCHES Scientific|CentOS|Ubuntu|Fedora)
  set(ROOTTEST_ENV_EXTRA LD_PRELOAD=${OPENGL_gl_LIBRARY})
endif()

foreach( t LoadAllLibs LoadAllLibsAZ LoadAllLibsZA)
  ROOTTEST_ADD_TEST(${t} MACRO assert${t}.C FAILREGEX "Error in|rc = -1")
endforeach()
