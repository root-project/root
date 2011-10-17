//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_METAPROCESSOR_H
#define CLING_METAPROCESSOR_H

#include "llvm/ADT/OwningPtr.h"

#include <string>

namespace cling {

  class Interpreter;
  class InputValidator;

  class MetaProcessorOpts {
  public:
    bool Quitting : 1; // is quit requested
    bool PrintingAST : 1; // is printAST requested;
    bool RawInput : 1; // is using wrappers requested;
    bool DynamicLookup : 1; // is using dynamic lookup
    
    MetaProcessorOpts() {
      Quitting = 0;
      PrintingAST = 0;
      RawInput = 0;
      DynamicLookup = 0;
    }
  };

  //---------------------------------------------------------------------------
  // Class for the user interaction with the interpreter
  //---------------------------------------------------------------------------
  class MetaProcessor {
  private:
    Interpreter& m_Interp; // the interpreter
    llvm::OwningPtr<InputValidator> m_InputValidator; // balanced paren etc
    MetaProcessorOpts m_Options;
  private:
    bool ProcessMeta(const std::string& input_line);
  public:
    MetaProcessor(Interpreter& interp);
    ~MetaProcessor();
    int process(const char* input_line);
    MetaProcessorOpts& getMetaProcessorOpts();
  };
} // end namespace cling

#endif // CLING_METAPROCESSOR_H


