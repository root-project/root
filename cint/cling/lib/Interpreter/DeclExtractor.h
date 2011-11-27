//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_DECL_EXTRACTOR_H
#define CLING_DECL_EXTRACTOR_H

#include "VerifyingSemaConsumer.h"

#include "clang/Sema/Lookup.h"

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

    ///\brief Checks for clashing names when trying to extract a declaration.
    ///
    /// Returns true if there is another declaration with the same name
    bool CheckForClashingNames(const llvm::SmallVector<clang::NamedDecl*, 4>& Decls, 
                               clang::DeclContext* DC, clang::Scope* S);

    ///\brief Performs semantic checking on a newly-extracted tag declaration.
    ///
    /// This routine performs all of the type-checking required for a tag 
    /// declaration once it has been built. It is used both to check tags before
    /// they have been moved onto the global scope.
    ///
    /// Sets NewTD->isInvalidDecl if an error was encountered.
    ///
    /// Returns true if the tag declaration is redeclaration.
    bool CheckTagDeclaration(clang::TagDecl* NewTD, 
                             clang::LookupResult& Previous);
  };

} // namespace cling

#endif // CLING_DECL_EXTRACTOR_H
