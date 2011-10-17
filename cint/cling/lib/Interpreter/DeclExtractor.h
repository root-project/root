//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_DECL_EXTRACTOR_H
#define CLING_DECL_EXTRACTOR_H

#include "VerifyingSemaConsumer.h"

namespace clang {
  class Decl;
  class DeclContext;
  class NamedDecl;
  class Scope;
}

namespace cling {
  class DeclExtractor : public VerifyingSemaConsumer {

  public:
    DeclExtractor();
    virtual ~DeclExtractor();
    void TransformTopLevelDecl(clang::DeclGroupRef DGR);

  private:
    void ExtractDecl(clang::Decl* D);
    bool CheckForClashingNames(const llvm::SmallVector<clang::NamedDecl*, 4>& Decls, 
                               clang::DeclContext* DC, clang::Scope* S);

  };

} // namespace cling

#endif // CLING_DECL_EXTRACTOR_H
