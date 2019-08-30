ROOTTEST_ADD_TEST(cxxmodules-implicit-build-error
                  COMMAND ${ROOT_rootcling_CMD} -f ${CMAKE_CURRENT_BINARY_DIR}/out.cxx -cxxmodule
                          implicit_build.h
                          implicit_build_linkdef.h
                  POSTCMD ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/out.pcm ${CMAKE_CURRENT_BINARY_DIR}/out_rdict.pcm ${CMAKE_CURRENT_BINARY_DIR}/A.pcm
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  PASSREGEX "Had to build non-system module A implicitly")
