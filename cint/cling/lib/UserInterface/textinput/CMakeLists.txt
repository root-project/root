#set(LLVM_LINK_COMPONENTS mc)
#set(LLVM_USED_LIBS clangBasic clangAST clangParse)

include_directories("..")

add_cling_library(clingUITextInput
  Editor.cpp
  History.cpp
  KeyBinding.cpp
  Range.cpp
  SignalHandler.cpp
  StreamReader.cpp
  StreamReaderUnix.cpp
  StreamReaderWin.cpp
  TerminalConfigUnix.cpp
  TerminalDisplay.cpp
  TerminalDisplayUnix.cpp
  TerminalDisplayWin.cpp
  TextInput.cpp
  TextInputContext.cpp)

#add_dependencies(clangDriver ClangAttrList ClangDiagnosticDriver
#                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)

if(MSVC)
  # windows.h doesn't compile with /Za
  get_target_property(NON_ANSI_COMPILE_FLAGS clingUITextInput COMPILE_FLAGS)
  string(REPLACE "/Za" "" NON_ANSI_COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
  set_target_properties(clingUITextInput PROPERTIES
    COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
endif()
