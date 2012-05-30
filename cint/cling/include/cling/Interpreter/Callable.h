//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_CALLABLE_H
#define CLING_CALLABLE_H

#include <vector>

namespace llvm {
  class Function;
  class GenericValue;
}

namespace cling {
   class FunctionDecl;
}

namespace cling {
  class Interpreter;
  class Value;

  ///\brief A representation of a function.
  //
  /// A callable combines a clang::FunctionDecl for AST-based reflection
  /// with the llvm::Function to call the function. It uses the
  /// interpreter's ExecutionContext to place the call passing in the
  /// normalized arguments. As no lookup is performed after constructing
  /// a Callable, calls are very efficient.
  class Callable {
  protected:
    /// \brief declaration; can also be a clang::CXXMethodDecl
    const clang::FunctionDecl* decl;
    /// \brief llvm::ExecutionEngine's function representation
    const llvm::Function* func;

  public:
    /// \brief Default constructor, creates a Callable that IsInvalid().
    Callable(): decl(0), func(0) {}
    /// \brief Construct a valid Value.
    Callable(const clang::FunctionDecl& Decl,
             const cling::Interpreter& Interp);

    /// \brief Determine whether the Callable can be invoked.
    //
    /// Determine whether the Callable can be invoked. Checking
    /// whether the llvm::Function pointer is valid.
    bool isValid() const { return func; }

    /// \brief Determine whether the Value is unset.
    //
    /// Determine whether the Callable cannot be invoked. Check
    /// whether the llvm::Function pointer is invalid.
    bool isInvalid() const { return !func; }

    /// \brief Invoke a free-standing (i.e. non-CXXMethod) function.
    //
    /// Invoke the function which must not be a CXXMethod. Pass in
    /// the parameters ArgValues; the return value of the function
    /// call ends up in Result.
    /// \return true if the call was successful.
    bool Invoke(Value& Result,
                const std::vector<llvm::GenericValue>& ArgValues) const;

    /// \brief Invoke a CXXMethod i.e. member function.
    //
    /// Invoke the function which must be a CXXMethod. Pass in the
    /// This pointer to the object on which to invoke the function and
    /// the parameters ArgValues; the return  value of the function call
    /// ends up in Result.
    /// \return true if the call was successful.
    bool InvokeThis(Value& Result, const llvm::GenericValue& This,
                    const std::vector<llvm::GenericValue>& ArgValues) const;

    /// \brief Invoke a free-standing function after checking parameter types.
    //
    /// Invoke the function which must not be a CXXMethod. If the parameters 
    /// passed in ArgValues match the expected types, call passing these
    /// parameters and put the return value of the call into Result.
    /// \return true if the call was successful.
    bool CheckedInvoke(Value& Result,
                       const std::vector<cling::Value>& ArgValues) const;

    /// \brief Invoke a member function after checking parameter types.
    //
    /// Invoke the function which must be a CXXMethod. Pass in the
    /// This pointer to the object on which to invoke the function and
    /// the parameters ArgValues. If the parameters 
    /// passed in ArgValues match the expected types, call passing these
    /// parameters and put the return value of the call into Result.
    /// \return true if the call was successful.
    bool CheckedInvokeThis(Value& Result, const llvm::GenericValue& This,
                           const std::vector<cling::Value>& ArgValues) const;
  };

} // end namespace cling

#endif // CLING_CALLABLE_H
