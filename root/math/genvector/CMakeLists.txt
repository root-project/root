ROOTTEST_ADD_TEST(coordinates3D
                  MACRO ${ROOT_SOURCE_DIR}/math/genvector/test/coordinates3D.cxx+)

ROOTTEST_ADD_TEST(coordinates4D
                  MACRO ${ROOT_SOURCE_DIR}/math/genvector/test/coordinates4D.cxx+)

ROOTTEST_ADD_TEST(rotationApplication
                  MACRO ${ROOT_SOURCE_DIR}/math/genvector/test/rotationApplication.cxx+)

# FIXME: Temporary workaround for runtime_cxxmodules;
if(NOT ROOT_runtime_cxxmodules_FOUND)
ROOTTEST_ADD_TEST(testGenVector
                  MACRO ${ROOT_SOURCE_DIR}/math/genvector/test/testGenVector.cxx+)
endif()
