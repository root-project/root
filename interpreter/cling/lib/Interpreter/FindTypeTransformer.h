//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Brock Mammen <brockmammen@gmail.com>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_FIND_TYPE_TRANSFORMER_H
#define CLING_FIND_TYPE_TRANSFORMER_H

#include "ASTTransformer.h"

namespace clang {
  class Decl;
  class Sema;
}

namespace cling {

  /// \brief Enables the use of cling::LookupHelper::findType<T>()
  /// by fixing template specializations for each usage of the function.
  class FindTypeTransformer : public ASTTransformer {
  public:
    FindTypeTransformer(clang::Sema* S);

    virtual ~FindTypeTransformer();

    Result Transform(clang::Decl* D) override;
  };

} // namespace cling

#endif // CLING_FIND_TYPE_TRANSFORMER_H
