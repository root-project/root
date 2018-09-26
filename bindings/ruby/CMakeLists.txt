############################################################################
# CMakeLists.txt file for building ROOT bindings/ruby package
############################################################################
include_directories(${RUBY_INCLUDE_DIRS})

ROOT_STANDARD_LIBRARY_PACKAGE(Ruby
                              LIBRARIES ${RUBY_LIBRARY} ${CMAKE_DL_LIBS}
                              DEPENDENCIES Hist MathCore)
