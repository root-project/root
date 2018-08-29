############################################################################
# CMakeLists.txt file for building ROOT core/foundation package
############################################################################

set(headers ESTLType.h
   RStringView.h
   TClassEdit.h
   ROOT/RIntegerSequence.hxx
   ROOT/RMakeUnique.hxx
   ROOT/RNotFn.hxx
   ROOT/RSpan.hxx
   ROOT/RStringView.hxx
   ROOT/span.hxx
   ROOT/TypeTraits.hxx
)

set(sources src/RConversionRuleParser.cxx
   src/TClassEdit.cxx
)

set(Foundation_dict_headers ${headers} PARENT_SCOPE)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res)

ROOT_OBJECT_LIBRARY(Foundation ${sources})

ROOT_OBJECT_LIBRARY(Foundation_Stage1 ${sources}) # used by rootcling_stage1
set_target_properties(Foundation_Stage1 PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} ${CLING_CXXFLAGS}")

ROOT_INSTALL_HEADERS()

ROOT_ADD_TEST_SUBDIRECTORY(test)
