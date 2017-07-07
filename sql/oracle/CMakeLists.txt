############################################################################
# CMakeLists.txt file for building ROOT sql/oracle package
############################################################################

include_directories(${ORACLE_INCLUDE_DIR})
add_definitions(${ORACLE_DEFINITIONS})

ROOT_GENERATE_DICTIONARY(G__Oracle *.h MODULE Oracle LINKDEF LinkDef.h DEPENDENCIES Net RIO)

ROOT_LINKER_LIBRARY(Oracle *.cxx G__Oracle.cxx LIBRARIES Core ${ORACLE_LIBRARIES} DEPENDENCIES Net RIO)

# Check if libstdc++ uses the new string ABI and fall back to the old one if yes.
# We have to check for this because on the nightly builds we have an old occi
# installation that uses the old ABI, which results in linking errors.
if(${GLIBCXX_USE_CXX11_ABI})
  file(GLOB srcs src/TOracle*.cxx)
  set_source_files_properties(${srcs} PROPERTIES COMPILE_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=0)
endif()

ROOT_INSTALL_HEADERS()
