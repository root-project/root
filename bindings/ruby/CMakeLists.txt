############################################################################
# CMakeLists.txt file for building ROOT bindings/ruby package
############################################################################
ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(tree/tree)
include_directories(${RUBY_INCLUDE_DIRS})

ROOT_GENERATE_DICTIONARY(G__Ruby *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Ruby LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(Ruby *.cxx G__Ruby.cxx LIBRARIES ${RUBY_LIBRARY} ${CMAKE_DL_LIBS} DEPENDENCIES Hist MathCore)
if(MACOSX_MINOR EQUAL 5)
  ROOT_EXECUTABLE(ruby64 ruby64.c LIBRARIES  ${RUBY_LIBRARY})
endif()

#---Install headers----------------------------------------------------------
ROOT_INSTALL_HEADERS()

