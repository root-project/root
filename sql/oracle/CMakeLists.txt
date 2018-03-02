############################################################################
# CMakeLists.txt file for building ROOT sql/oracle package
############################################################################

include_directories(${ORACLE_INCLUDE_DIR})
add_definitions(${ORACLE_DEFINITIONS})

# Check if libstdc++ uses the new string ABI and fall back to the old one if yes.
# We have to check for this because on the nightly builds we have an old occi
# installation that uses the old ABI, which results in linking errors.
if(${GLIBCXX_USE_CXX11_ABI})
  file(GLOB srcs src/TOracle*.cxx)
  set_source_files_properties(${srcs} PROPERTIES COMPILE_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=0)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Oracle
                              LIBRARIES ${ORACLE_LIBRARIES}
                              DEPENDENCIES Core Net RIO)
