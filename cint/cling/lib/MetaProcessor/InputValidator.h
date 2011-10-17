//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_INPUT_VALIDATOR_H
#define CLING_INPUT_VALIDATOR_H

#include "llvm/ADT/StringRef.h"

#include <stack>

namespace clang {
  class LangOptions;
}

namespace cling {
  class InputValidator {
  public:
    enum Result {
      kIncomplete,
      kComplete,
      kMismatch,
      kNumResults
    };
    
    InputValidator();
    ~InputValidator();
    
    Result Validate(llvm::StringRef input_line, clang::LangOptions& LO);
    std::string& TakeInput() {
      return m_Input;
    }
    int getExpectedIndent() { return m_ParenStack.size(); }
    void Reset();
  private:
    std::string m_Input;
    std::stack<int> m_ParenStack;
  };
}
#endif // CLING_INPUT_VALIDATOR_H
