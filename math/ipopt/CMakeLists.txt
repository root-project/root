############################################################################
# CMakeLists.txt file for building ROOT math/ipopt package
############################################################################

include_directories(${IPOPT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
add_definitions(-DUSE_ROOT_ERROR ${IPOPT_DEFINITIONS} )


set(headers Math/IpoptMinimizer.h )
set(linkdefs Math/LinkDef.h )

ROOT_STANDARD_LIBRARY_PACKAGE(Ipopt
                              HEADERS ${headers}
                              LINKDEF ${linkdefs}
                              LIBRARIES ${IPOPT_LIBRARIES}
                              DEPENDENCIES MathCore MathMore)

#ROOT_ADD_TEST_SUBDIRECTORY(test)




