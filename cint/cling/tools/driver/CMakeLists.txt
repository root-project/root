#set( LLVM_LINK_COMPONENTS
#  ${LLVM_TARGETS_TO_BUILD}
#  bitreader
#  bitwriter
#  codegen
#  instrumentation
#  ipo
#  linker
#  selectiondag
#  )

add_cling_executable(cling
  cling.cpp)

target_link_libraries(cling
  clingUserInterface
  clingMetaProcessor
  clingInterpreter
  clangFrontend
  clangSerialization
  clangDriver
  clangCodeGen
  clangParse
  clangSema
  clangAnalysis
  clangRewrite
  clangAST
  clangLex
  clangBasic
  )

install(TARGETS cling
  RUNTIME DESTINATION bin)
