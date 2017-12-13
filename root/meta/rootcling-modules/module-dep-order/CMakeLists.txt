set(ENV{ROOT_MODULES} 1)
ROOTTEST_ADD_TEST(cxxmodules-implicit-build-error
                  COMMAND ${ROOT_rootcling_CMD} -f ${CMAKE_CURRENT_BINARY_DIR}/out.cxx -cxxmodule
                          implicit_build.h
                          implicit_build_linkdef.h
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  PASSREGEX "Had to build non-system module A implicitly")
