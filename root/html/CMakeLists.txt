ROOTTEST_ADD_TEST(runClassInModule
                  MACRO runClassInModule.C
                  WILLFAIL
                  OUTREF ClassInModule.ref)

ROOTTEST_ADD_TEST(runFindSource
                  MACRO runFindSource.C
                  WILLFAIL
                  OUTREF FindSource.ref)
                  
if(OPENGL_gl_LIBRARY AND ROOTTEST_OS_ID MATCHES Scientific|CentOS|Ubuntu|Fedora)
  set(ROOTTEST_ENV_EXTRA LD_PRELOAD=${OPENGL_gl_LIBRARY})
endif()

# FIXME: Temporary workaround for runtime_cxxmodule
if(NOT ROOT_runtime_cxxmodules_FOUND)
ROOTTEST_ADD_TEST(runMakeIndex
                  MACRO runMakeIndex.C
                  OUTCNV MakeIndex_convert.sh
                  OUTREF MakeIndex.oref
                  ERRREF MakeIndex.eref)
endif()
