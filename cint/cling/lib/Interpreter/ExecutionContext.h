//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_EXECUTIONCONTEXT_H
#define CLING_EXECUTIONCONTEXT_H

#include "llvm/ADT/StringRef.h"

#include <vector>

namespace llvm {
  class Module;
  class ExecutionEngine;
  struct GenericValue;
}

namespace clang {
  class CompilerInstance;
  class CodeGenerator;
}

namespace cling {
  class Interpreter;
  class Value;
  
  class ExecutionContext {
  public:
    typedef void* (*LazyFunctionCreatorFunc_t)(const std::string&);
    
    ExecutionContext();
    ~ExecutionContext();
        
    void installLazyFunctionCreator(LazyFunctionCreatorFunc_t fp);
    
    void runStaticInitializersOnce(llvm::Module* m);
    void runStaticDestructorsOnce(llvm::Module* m);
    
    void executeFunction(llvm::StringRef function, 
                         llvm::GenericValue* returnValue = 0);
    
  private:
    static void* HandleMissingFunction(const std::string&);
    static void* NotifyLazyFunctionCreators(const std::string&);
    
    int verifyModule(llvm::Module* m);
    void printModule(llvm::Module* m);
    void InitializeBuilder(llvm::Module* m);
    
    static std::vector<std::string> m_vec_unresolved;
    static std::vector<LazyFunctionCreatorFunc_t> m_vec_lazy_function;
    
    llvm::ExecutionEngine* m_engine; // Owned by JIT
    unsigned m_posInitGlobals; // position (global idx in out module) of next global to be initialized in m_ASTCI's AST
    bool m_RunningStaticInits; // prevent the recursive run of the static inits
  };
} // end cling
#endif // CLING_EXECUTIONCONTEXT_H

