############################################################################
# CMakeLists.txt file for building ROOT math/vecops package
############################################################################

set(VECOPS_HEADERS ROOT/TVec.hxx )

ROOT_STANDARD_LIBRARY_PACKAGE(VecOps HEADERS ${VECOPS_HEADERS}
                              DEPENDENCIES Core
                              SOURCES *.cxx)

ROOT_ADD_TEST_SUBDIRECTORY(test)
