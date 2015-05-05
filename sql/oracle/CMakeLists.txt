############################################################################
# CMakeLists.txt file for building ROOT sql/oracle package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)

include_directories(${ORACLE_INCLUDE_DIR})
add_definitions(${ORACLE_DEFINITIONS})

ROOT_GENERATE_DICTIONARY(G__Oracle *.h MODULE Oracle LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(Oracle *.cxx G__Oracle.cxx LIBRARIES Core ${ORACLE_LIBRARIES} DEPENDENCIES Net RIO)
if(GCC_MAJOR GREATER 4)
  file(GLOB srcs src/TOracle*.cxx)
  set_source_files_properties(${srcs} PROPERTIES COMPILE_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=0)
endif()

ROOT_INSTALL_HEADERS()
