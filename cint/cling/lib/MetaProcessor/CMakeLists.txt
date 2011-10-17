#set(LLVM_LINK_COMPONENTS mc)
#set(LLVM_USED_LIBS clangBasic clangAST clangParse)


add_cling_library(clingMetaProcessor
  MetaProcessor.cpp
  InputValidator.cpp)

#add_dependencies(clangDriver ClangAttrList ClangDiagnosticDriver
#                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)
