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
  class Value;

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
    bool ProcessMeta(const std::string& input_line, cling::Value* result);
  public:
    MetaProcessor(Interpreter& interp);
    ~MetaProcessor();

    MetaProcessorOpts& getMetaProcessorOpts();

    ///\brief Process the input coming from the prompt and possibli returns
    /// result of the execution of the last statement
    /// @param[in] input_line - the user input
    /// @param[out] result - the cling::Value as result of the execution of the
    ///             last statement
    ///
    ///\returns 0 on success
    ///
    int process(const char* input_line, cling::Value* result = 0);

    ///\brief Executes a file given the CINT specific rules. Mainly used as:
    /// .x filename[(args)], which in turn #include-s the filename and runs a
    /// function with signature void filename(args)
    /// @param[in] fileWithArgs - the filename(args)
    /// @param[out] result - the cling::Value as result of the execution of the
    ///             last statement
    ///
    ///\returns true on success
    ///
    bool executeFile(const std::string& fileWithArgs, cling::Value* result = 0);

  };
} // end namespace cling

#endif // CLING_METAPROCESSOR_H


