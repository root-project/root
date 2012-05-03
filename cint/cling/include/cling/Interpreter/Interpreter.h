//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_INTERPRETER_H
#define CLING_INTERPRETER_H

#include "cling/Interpreter/InvocationOptions.h"

#include "clang/AST/Type.h"

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
  class Parser;
}

namespace cling {
  class CompilationOptions;

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

  ///\brief Class that implements the interpreter-like behavior. It manages the
  /// incremental compilation.
  ///
  class Interpreter {

  public:

    ///\brief Describes the return result of the different routines that do the
    /// incremental compilation.
    ///
    enum CompilationResult {
      kSuccess,
      kFailure,
      kMoreInputExpected
    };

  private:

    ///\brief Interpreter invocation options.
    ///
    InvocationOptions m_Opts;

    ///\brief Cling's execution engine - a well wrapped llvm execution engine.
    ///
    llvm::OwningPtr<ExecutionContext> m_ExecutionContext;

    ///\brief Cling's worker class implementing the incremental compilation.
    ///
    llvm::OwningPtr<IncrementalParser> m_IncrParser;

    ///\brief Counter used when we need unique names.
    ///
    unsigned long long m_UniqueCounter;

    ///\brief Flag toggling the AST printing on or off.
    ///
    bool m_PrintAST;

    bool m_ValuePrinterEnabled; // whether the value printer is loaded

    ///\brief Stream to dump values into.
    ///
    /// TODO: Since it is only used by the ValuePrinterSynthesizer it should be
    /// somewhere else.
    ///
    llvm::OwningPtr<llvm::raw_ostream> m_ValuePrintStream;

    ///\breif Helper that manages when the destructor of an object to be called.
    ///
    /// The object is registered first as an CXAAtExitElement and then cling
    /// takes the control of it's destruction.
    ///
    struct CXAAtExitElement {
      ///\brief Constructs an element, whose destruction time will be managed by
      /// the interpreter. (By registering a function to be called by exit 
      /// or when a shared library is unloaded.)
      ///
      /// Registers destructors for objects with static storage duration with 
      /// the _cxa atexit function rather than the atexit function. This option 
      /// is required for fully standards-compliant handling of static 
      /// destructors(many of them created by cling), but will only work if 
      /// your C library supports __cxa_atexit (means we have our own work 
      /// around for Windows). More information about __cxa_atexit could be 
      /// found in the Itanium C++ ABI spec.
      ///
      ///\param [in] func - The function to be called on exit or unloading of 
      ///                   shared lib.(The destructor of the object.)
      ///\param [in] arg - The argument the func to be called with.
      ///\param [in] dso - The dynamic shared object handle.
      ///\param [in] fromTLD - The unloading of this top level declaration will
      ///                      trigger the atexit function.
      ///
      CXAAtExitElement(void (*func) (void*), void* arg, void* dso,
                       clang::Decl* fromTLD):
        m_Func(func), m_Arg(arg), m_DSO(dso), m_FromTLD(fromTLD) {}
   
      ///\brief The function to be called.
      ///
      void (*m_Func)(void*);

      ///\brief The single argument passed to the function.
      ///
      void* m_Arg;

      /// \brief The DSO handle.
      ///
      void* m_DSO;

      ///\brief Clang's top level declaration, whose unloading will trigger the 
      /// call this atexit function.
      ///
      clang::Decl* m_FromTLD;
    };
 
    ///\brief Static object, which are bound to unloading of certain declaration
    /// to be destructed.
    ///
    llvm::SmallVector<CXAAtExitElement, 20> m_AtExitFuncs;

    ///\brief Processes the invocation options.
    ///
    void handleFrontendOptions();

    ///\brief Worker function, building block for interpreter's public 
    /// interfaces.
    ///
    ///\param [in] input - The input being compiled.
    ///\param [in] CompilationOptions - The option set driving the compilation.
    ///\param [out] D - The first declaration of the compiled input.
    ///
    ///\returns Whether the operation was fully successful.
    ///
    CompilationResult Declare(const std::string& input, 
                              const CompilationOptions& CO,
                              const clang::Decl** D = 0);

    ///\brief Worker function, building block for interpreter's public 
    /// interfaces.
    ///
    ///\param [in] input - The input being compiled.
    ///\param [in] CompilationOptions - The option set driving the compilation.
    ///\param [out] V - The result of the evaluation of the input.
    ///
    ///\returns Whether the operation was fully successful.
    ///
    CompilationResult Evaluate(const std::string& input, 
                               const CompilationOptions& CO,
                               Value* V = 0);

    ///\brief Wraps a given input.
    ///
    /// The interpreter must be able to run statements on the fly, which is not
    /// C++ standard-compliant operation. In order to do that we must wrap the
    /// input into a artificial function, containing the statements and run it.
    ///\param [out] input - The input to wrap.
    ///\param [out] fname - The wrapper function's name.
    ///
    void WrapInput(std::string& input, std::string& fname);

    ///\brief Runs given function.
    ///
    ///\param [in] fname - The function name.
    ///\param [out] res - The return result of the run function.
    ///
    ///\returns true if successful otherwise false.
    ///
    bool RunFunction(llvm::StringRef fname, llvm::GenericValue* res = 0);

    ///\brief Super efficient way of creating unique names, which will be used 
    /// as a part of the compilation process.
    ///
    /// Creates the name directly in the compiler's identifier table, so that
    /// next time the compiler looks for that name it will find it directly
    /// there.
    ///
    llvm::StringRef createUniqueWrapper();

    ///\brief Forwards to cling::ExecutionContext::addSymbol.
    ///
    bool addSymbol(const char* symbolName,  void* symbolAddress);

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
        clang::NamedDecl *result = getSingleDecl();
        if (result) {
           return llvm::dyn_cast<T>(result);
        } else {
           return 0;
        }
      }
      
      friend class Interpreter;
    };

    Interpreter(int argc, const char* const *argv, const char* llvmdir = 0);
    virtual ~Interpreter();

    const InvocationOptions& getOptions() const { return m_Opts; }
    InvocationOptions& getOptions() { return m_Opts; }

    ///\brief Shows the current version of the project.
    ///
    ///\returns The current svn revision (svn Id). 
    ///
    const char* getVersion() const;

    ///\brief Creates unique name that can be used for various aims.
    ///
    void createUniqueName(std::string& out);

    ///\brief Adds an include path (-I).
    ///
    void AddIncludePath(llvm::StringRef incpath);

    ///\brief Prints the current include paths that are used.
    ///
    void DumpIncludePath();
 
    ///\brief Compiles the given input.
    ///
    /// This interface helps to run everything that cling can run. From
    /// declaring header files to running or evaluating single statements.
    /// Note that this should be used when there is no idea of what kind of 
    /// input is going to be processed. Otherwise if is known, for example 
    /// only header files are going to be processed it is much faster to run the
    /// specific interface for doing that - in the particular case - declare().
    ///
    ///\param [in] input - The input to be compiled.
    ///\param [out] V - The result of the evaluation of the input.
    ///\param [out] D - The first declaration of the compiled input.
    ///
    ///\returns Whether the operation was fully successful.
    ///
    CompilationResult process(const std::string& input, Value* V = 0,
                              const clang::Decl** D = 0);

    ///\brief Compiles input line, which doesn't contain statements.
    ///
    /// The interface circumvents the most of the extra work necessary to 
    /// compile and run statements.
    ///
    /// @param[in] input - The input containing only declarations (aka 
    ///                    Top Level Declarations)
    /// @param[out] D - The first compiled declaration from the input
    ///
    ///\returns Whether the operation was fully successful.
    ///
    CompilationResult declare(const std::string& input, 
                              const clang::Decl** D = 0);

    ///\brief Compiles input line, which contains only expressions.
    ///
    /// The interface circumvents the most of the extra work necessary extract
    /// the declarations from the input.
    ///
    /// @param[in] input - The input containing only expressions
    /// @param[out] V - The value of the executed input
    ///
    ///\returns Whether the operation was fully successful.
    ///
    CompilationResult evaluate(const std::string& input, 
                               Value* V = 0);

    ///\brief Compiles input line, which contains only expressions and prints out
    /// the result of its execution.
    ///
    /// The interface circumvents the most of the extra work necessary extract
    /// the declarations from the input.
    ///
    /// @param[in] input - The input containing only expressions.
    /// @param[out] V - The value of the executed input.
    ///
    ///\returns Whether the operation was fully successful.
    ///
    CompilationResult echo(const std::string& input, Value* V = 0);

    ///\brief Loads header file or shared library.
    ///
    ///\param [in] filename - The file to loaded.
    ///\param [in] allowSharedLib - Whether to try to load the file as shared
    ///                             library.
    ///
    ///\returns true for happiness.
    ///
    bool loadFile(const std::string& filename, bool allowSharedLib = true);
    
    ///\brief Lookup a type by name, starting from the global
    /// namespace.
    ///
    /// \param [in] typeName - The type to lookup.
    ///
    /// \retval retval - On a failed lookup retval.isNull() will be true.
    ///
    clang::QualType lookupType(const std::string& typeName);

    ///\brief Lookup a class declaration by name, starting from the global
    /// namespace, also handles struct, union, namespace, and enum.
    ///
    ///\param [in] className - The name of the class, struct, union,
    ///                        namespace, or enum to lookup.
    ///\returns The found declaration or null.
    ///
    clang::Decl* lookupClass(const std::string& className);

    clang::Decl* lookupFunctionProto(clang::Decl* classDecl,
                                     const std::string& funcName,
                                     const std::string& funcProto);

    clang::Decl* lookupFunctionArgs(clang::Decl* classDecl,
                                    const std::string& funcName,
                                    const std::string& funcArgs);

    void enableDynamicLookup(bool value = true);
    bool isDynamicLookupEnabled();

    bool isPrintingAST() { return m_PrintAST; }
    void enablePrintAST(bool print = true);
    
    clang::CompilerInstance* getCI() const;
    clang::Parser* getParser() const;

    void installLazyFunctionCreator(void* (*fp)(const std::string&));
    
    llvm::raw_ostream& getValuePrinterStream() const { return *m_ValuePrintStream; }

    void runStaticInitializersOnce() const;

    int CXAAtExit(void (*func) (void*), void* arg, void* dso);    

    ///\brief Evaluates given expression within given declaration context.
    ///
    ///\param[in] expr - The expression.
    ///\param[in] DC - The declaration context in which the expression is going
    ///                to be evaluated.
    ///\param[in] ValuePrinterReq - Whether the value printing is requested.
    ///
    ///\returns The result of the evaluation if the expression.
    ///
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
    ///
    void setCallbacks(InterpreterCallbacks* C);

    friend class runtime::internal::LifetimeHandler;
  };
  
} // namespace cling

#endif // CLING_INTERPRETER_H
