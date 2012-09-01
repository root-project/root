//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: Interpreter.h 45404 2012-08-06 08:30:06Z vvassilev $
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_LOOKUP_HELPER_H
#define CLING_LOOKUP_HELPER_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
  class Decl;
  class FunctionDecl;
  class Parser;
  class QualType;
  class Type;
}

namespace cling {

  ///\brief Reflection information query interface. The class performs lookups
  /// in the currently loaded information in the AST, using the same Parser, Sema
  /// and Preprocessor objects.
  ///
  class LookupHelper {
  private:
    clang::Parser* m_Parser; //doesn't own it.
  public:

    LookupHelper(clang::Parser* P) : m_Parser(P) {}

    ///\brief Lookup a type by name, starting from the global
    /// namespace.
    ///
    /// \param [in] typeName - The type to lookup.
    ///
    /// \retval retval - On a failed lookup retval.isNull() will be true.
    ///
    clang::QualType findType(llvm::StringRef typeName) const;

    ///\brief Lookup a class declaration by name, starting from the global
    /// namespace, also handles struct, union, namespace, and enum.
    ///
    ///\param [in] className   - The name of the class, struct, union,
    ///                          namespace, or enum to lookup.
    ///\param [out] resultType - The type of the class, struct, union,
    ///                          or enum to lookup; NULL otherwise.
    ///\returns The found declaration or null.
    ///
    const clang::Decl* findScope(llvm::StringRef className,
                                 const clang::Type** resultType = 0) const;

     
    const clang::FunctionDecl* findFunctionProto(const clang::Decl* scopeDecl,
                                                 llvm::StringRef funcName,
                                                llvm::StringRef funcProto) const;

    const clang::FunctionDecl* findFunctionArgs(const clang::Decl* scopeDecl,
                                                llvm::StringRef funcName,
                                                llvm::StringRef funcArgs) const;

  };

} // end namespace

#endif // CLING_LOOKUP_HELPER_H
