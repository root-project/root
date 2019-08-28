ROOTTEST_ADD_TEST(cxxmodules-implicit-build-error
                  COMMAND ${ROOT_rootcling_CMD} -f ${CMAKE_CURRENT_BINARY_DIR}/out.cxx -cxxmodule
                          implicit_build.h
                          implicit_build_linkdef.h
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  PASSREGEX "Had to build non-system module A implicitly")
# Remove the the implicitly built module to make the test work for incremental builds and testing.
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/out.pcm
            ${CMAKE_CURRENT_BINARY_DIR}/out_rdict.pcm
            ${CMAKE_CURRENT_BINARY_DIR}/A.pcm)
