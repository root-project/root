//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ASTUtils.h"

#include "clang/AST/DeclarationName.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Lookup.h"

using namespace clang;

namespace cling {
  Expr* Synthesize::CStyleCastPtrExpr(Sema* S, QualType Ty, uint64_t Ptr) {
    ASTContext& Ctx = S->getASTContext();
    if (!Ty->isPointerType())
      Ty = Ctx.getPointerType(Ty);
    TypeSourceInfo* TSI = Ctx.CreateTypeSourceInfo(Ty);
    const llvm::APInt Addr(8 * sizeof(void *), Ptr);
    
    Expr* Result = IntegerLiteral::Create(Ctx, Addr, Ctx.UnsignedLongTy, 
                                          SourceLocation());
    Result = S->BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(),
                                         Result).take();
    assert(Result && "Cannot create CStyleCastPtrExpr");
    return Result;

  }

  NamespaceDecl* Lookup::Namespace(Sema* S, const char* Name,
                                   DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    LookupResult R(*S, DName, SourceLocation(), 
                   Sema::LookupNestedNameSpecifierName);
    if (!Within)
      S->LookupName(R, S->TUScope);
    else
      S->LookupQualifiedName(R, Within);

    if (R.empty())
      return 0;

    R.resolveKind();

    return dyn_cast<NamespaceDecl>(R.getFoundDecl());
  }

  NamedDecl* Lookup::Named(Sema* S, const char* Name, DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);

    LookupResult R(*S, DName, SourceLocation(), Sema::LookupOrdinaryName,
                   Sema::ForRedeclaration);
    if (!Within)
      S->LookupName(R, S->TUScope);
    else
      S->LookupQualifiedName(R, Within);

    if (R.empty())
      return 0;

    R.resolveKind();

    return R.getFoundDecl();
    
  }

} // end namespace cling
