############################################################################
# CMakeLists.txt file for building ROOT math/rtools package
# @author Pere Mato, CERN
############################################################################

include_directories(${R_INCLUDE_DIRS})

ROOT_GENERATE_DICTIONARY(G__Rtools Math/RMinimizer.h MODULE Rtools LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(Rtools *.cxx G__Rtools.cxx LIBRARIES Core ${R_LIBRARIES} DEPENDENCIES MathCore RInterface)
ROOT_INSTALL_HEADERS()

FILE(COPY "${CMAKE_SOURCE_DIR}/etc/plugins/ROOT@@Math@@Minimizer/P090_RMinimizer.C" DESTINATION "${CMAKE_BINARY_DIR}/etc/plugins/ROOT@@Math@@Minimizer/")

#ROOT_ADD_TEST_SUBDIRECTORY(test)
