//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_INTERPRETER_H
#define CLING_INTERPRETER_H

#include "cling/Interpreter/InvocationOptions.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <string>

namespace llvm {
  class raw_ostream;
  struct GenericValue;
}

namespace clang {
  class ASTConsumer;
  class ASTContext;
  class CompilerInstance;
  class CompoundStmt;
  class Decl;
  class DeclContext;
  class Expr;
  class NamedDecl;
}

namespace cling {
  class DynamicExprInfo;
  namespace runtime {
    namespace internal {
      template <typename T>
      T EvaluateT(DynamicExprInfo* ExprInfo, clang::DeclContext* DC);
      class LifetimeHandler;
    }
  }
  class ExecutionContext;
  class IncrementalParser;
  class InterpreterCallbacks;
  class Value;

  ///\brief Helper structure used to provide specific context of the evaluated
  /// expression, when needed.
  ///
  /// Consider:
  /// @code
  /// int a = 5;
  /// const char* b = dep->Symbol(a);
  /// @endcode
  /// In the particular case we need to pass a context to the evaluator of the
  /// unknown symbol. The addresses of the items in the context are not known at
  /// compile time, so they cannot be embedded directly. Instead of that we
  /// need to create an array of addresses of those context items (mainly
  /// variables) and insert them into the evaluated expression at runtime
  /// This information is kept using the syntax: "dep->Symbol(*(int*)@)",
  /// where @ denotes that the runtime address the variable "a" is needed.
  ///
  class DynamicExprInfo {
  private:

    /// \brief The expression template.
    const char* m_Template;

    std::string m_Result;

    /// \brief The variable list.
    void** m_Addresses;

    /// \brief The variable is set if it is required to print out the result of
    /// the dynamic expression after evaluation
    bool m_ValuePrinterReq;
  public:
    DynamicExprInfo(const char* templ, void* addresses[], bool valuePrinterReq):
      m_Template(templ), m_Result(templ), m_Addresses(addresses), 
      m_ValuePrinterReq(valuePrinterReq){}

    ///\brief Performs the insertions of the context in the expression just
    /// before evaluation. To be used only at runtime.
    ///
    const char* getExpr();
    bool isValuePrinterRequested() { return m_ValuePrinterReq; }
  };

  //---------------------------------------------------------------------------
  //! Class for managing many translation units supporting automatic
  //! forward declarations and linking
  //---------------------------------------------------------------------------
  class Interpreter {
  public:

    ///\brief Implements named parameter idiom - allows the idiom 
    /// LookupDecl().LookupDecl()...
    /// 
    class NamedDeclResult {
    private:
      Interpreter* m_Interpreter;
      clang::ASTContext& m_Context;
      const clang::DeclContext* m_CurDeclContext;
      clang::NamedDecl* m_Result;
      NamedDeclResult(llvm::StringRef Decl, Interpreter* interp, 
                      const clang::DeclContext* Within = 0);
    public:
      NamedDeclResult& LookupDecl(llvm::StringRef);
      operator clang::NamedDecl* () const { return getSingleDecl(); }
      clang::NamedDecl* getSingleDecl() const;
      template<class T> T* getAs(){
        return llvm::dyn_cast<T>(getSingleDecl());
      }
      
      friend class Interpreter;
    };

    enum CompilationResult {
      kSuccess,
      kFailure,
      kMoreInputExpected
    };

    //---------------------------------------------------------------------
    //! Constructor
    //---------------------------------------------------------------------
    Interpreter(int argc, const char* const *argv, const char* llvmdir = 0);
    
    //---------------------------------------------------------------------
    //! Destructor
    //---------------------------------------------------------------------
    virtual ~Interpreter();

    const InvocationOptions& getOptions() const { return m_Opts; }
    InvocationOptions& getOptions() { return m_Opts; }

    const char* getVersion() const;
    std::string createUniqueName();
    void AddIncludePath(const char *incpath);
    void DumpIncludePath();
 
    ///\brief Compiles input line. 
    ///
    /// This is top most interface, which helps running statements and 
    /// expressions on the global scope. If rawInput mode disabled the
    /// input will be wrapped into wrapper function. Declaration extraction
    /// will be enabled and all declarations will be extracted as global.
    /// After compilation the wrapper will be executed.
    /// 
    /// If rawInput enabled no execution or declaration extraction is done
    ///
    /// @param[in] input_line - the input to be compiled
    /// @param[in] rawInput - turns on or off the wrapping of the input
    /// @param[out] D - returns the first declaration that was parsed from the
    ///                 input
    ///
    CompilationResult processLine(const std::string& input_line, 
                                  bool rawInput = false,
                                  clang::Decl** D = 0);
    
    bool loadFile(const std::string& filename,
                  const std::string* trailcode = 0,
                  bool allowSharedLib = true);
    
    bool executeFile(const std::string& fileWithArgs);
    
    void enableDynamicLookup(bool value = true);
    bool isDynamicLookupEnabled();

    bool isPrintingAST() { return m_PrintAST; }
    void enablePrintAST(bool print = true);
    
    void dumpAST(bool showAST = true, int last = -1);
    
    clang::CompilerInstance* getCI() const;

    void installLazyFunctionCreator(void* (*fp)(const std::string&));
    
    llvm::raw_ostream& getValuePrinterStream() const { return *m_ValuePrintStream; }

    void runStaticInitializersOnce() const;

    int CXAAtExit(void (*func) (void*), void* arg, void* dso);
    
  private:
    InvocationOptions m_Opts; // Interpreter options
    llvm::OwningPtr<ExecutionContext> m_ExecutionContext;
    llvm::OwningPtr<IncrementalParser> m_IncrParser; // incremental AST and its parser
    unsigned long long m_UniqueCounter; // number of generated call wrappers
    bool m_PrintAST; // whether to print the AST to be processed
    bool m_ValuePrinterEnabled; // whether the value printer is loaded
    llvm::OwningPtr<llvm::raw_ostream> m_ValuePrintStream; // stream to dump values into
    clang::Decl *m_LastDump; // last dump point

    struct CXAAtExitElement {
      CXAAtExitElement(void (*func) (void*), void* arg, void* dso,
                       clang::Decl* fromTLD):
        m_Func(func), m_Arg(arg), m_DSO(dso), m_FromTLD(fromTLD) {}
      void (*m_Func)(void*);
      void* m_Arg;
      void* m_DSO;
      clang::Decl* m_FromTLD; // when unloading this top level decl, call this atexit func.
    };

    llvm::SmallVector<CXAAtExitElement, 20> m_AtExitFuncs;

  private:
    void handleFrontendOptions();
    CompilationResult handleLine(llvm::StringRef Input,
                                 llvm::StringRef FunctionName,
                                 clang::Decl** D);
    void WrapInput(std::string& input, std::string& fname);
    bool RunFunction(llvm::StringRef fname, llvm::GenericValue* res = 0);
    friend class runtime::internal::LifetimeHandler;
    
  public:
    ///\brief Evaluates given expression within given declaration context.
    ///
    /// @param[in] expr The expression.
    /// @param[in] DC The declaration context in which the expression is going
    /// to be evaluated.
    Value Evaluate(const char* expr, clang::DeclContext* DC, 
                   bool ValuePrinterReq = false);

    ///\brief Looks up declaration within given declaration context. Does top
    /// down lookup.
    ///
    ///@param[in] Decl Declaration name.
    ///@param[in] Within Starting declaration context.
    ///
    NamedDeclResult LookupDecl(llvm::StringRef Decl, 
                               const clang::DeclContext* Within = 0);

    ///\brief Sets callbacks needed for the dynamic lookup.
    void setCallbacks(InterpreterCallbacks* C);
  };
  
} // namespace cling

#endif // CLING_INTERPRETER_H
