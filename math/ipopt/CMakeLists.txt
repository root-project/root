############################################################################
# CMakeLists.txt file for building ROOT math/ipopt package
############################################################################



ROOT_STANDARD_LIBRARY_PACKAGE(Ipopt
                              HEADERS 
                                Math/IpoptMinimizer.h
                              LINKDEF 
                                Math/LinkDef.h
                              LIBRARIES ${IPOPT_LIBRARIES}
                              DEPENDENCIES MathCore MathMore)

target_compile_definitions(Ipopt PUBLIC USE_ROOT_ERROR ${IPOPT_DEFINITIONS})
target_include_directories(Ipopt PUBLIC ${IPOPT_INCLUDE_DIRS})
ROOT_ADD_TEST_SUBDIRECTORY(test)
