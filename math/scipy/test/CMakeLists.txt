project(scipy-tests)
find_package(ROOT REQUIRED)

include_directories(${ROOT_INCLUDE_DIRS})

set(Libraries Core RIO Net Hist Graf Graf3d Gpad Tree
              Rint Postscript Matrix MathCore Thread Scipy)


include_directories(${CMAKE_SOURCE_DIR})
ROOT_ADD_GTEST(scipy-minimizer testScipyMinimizer.cxx LIBRARIES  ${Libraries})
