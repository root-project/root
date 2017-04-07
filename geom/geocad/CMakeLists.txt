############################################################################
# CMakeLists.txt file for building ROOT geom/geocad package
# @author Pere Mato, CERN
############################################################################

include_directories(${OCC_INCLUDE_DIR})

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-ignored-qualifiers)

ROOT_GENERATE_DICTIONARY(G__GeoCad *.h MODULE GeoCad LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(GeoCad *.cxx G__GeoCad.cxx LIBRARIES Core ${OCC_LIBRARIES} DEPENDENCIES Geom)

ROOT_INSTALL_HEADERS()
