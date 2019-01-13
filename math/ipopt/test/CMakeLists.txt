project(ipopt-tests)
find_package(ROOT REQUIRED)

include_directories(${ROOT_INCLUDE_DIRS})

set(Libraries Core RIO Net Hist Graf Graf3d Gpad Tree
              Rint Postscript Matrix Physics MathCore Thread MathMore Ipopt)


include_directories(${CMAKE_SOURCE_DIR})

ROOT_ADD_GTEST(minimizer testIpoptMinimizer.cxx LIBRARIES ${IPOPT_LIBRARIES} ${Libraries})
