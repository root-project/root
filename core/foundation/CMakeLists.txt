############################################################################
# CMakeLists.txt file for building ROOT core/foundation package
############################################################################

set(sources
  src/RConversionRuleParser.cxx
  src/TClassEdit.cxx
)

ROOT_OBJECT_LIBRARY(Foundation ${sources})
target_include_directories(Foundation PRIVATE res)

ROOT_OBJECT_LIBRARY(Foundation_Stage1 ${sources}) # used by rootcling_stage1
target_include_directories(Foundation_Stage1 PRIVATE res)
set_target_properties(Foundation_Stage1 PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} ${CLING_CXXFLAGS}")

ROOT_INSTALL_HEADERS()
ROOT_ADD_TEST_SUBDIRECTORY(test)
