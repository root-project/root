############################################################################
# CMakeLists.txt file for building ROOT net/net package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(math/mathcore)

ROOT_STANDARD_LIBRARY_PACKAGE(Net DEPENDENCIES RIO MathCore)

