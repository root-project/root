//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Lookup.h"

using namespace clang;

namespace cling {
namespace utils {
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


  QualType Transform::GetPartiallyDesugaredType(const ASTContext& Ctx, 
                                                QualType QT, 
                              const llvm::SmallSet<const Type*, 4>& TypesToSkip){
    // If there are no constains - use the standard desugaring.
    if (!TypesToSkip.size())
      return QT.getDesugaredType(Ctx);

    while(isa<TypedefType>(QT.getTypePtr())) {
      if (!TypesToSkip.count(QT.getTypePtr())) 
        QT = QT.getSingleStepDesugaredType(Ctx);
      else
        return QT;
    }

    // In case of template specializations iterate over the arguments and 
    // desugar them as well.
    if(const TemplateSpecializationType* TST 
       = dyn_cast<const TemplateSpecializationType>(QT.getTypePtr())) {
     
      bool mightHaveChanged = false;
      llvm::SmallVector<TemplateArgument, 4> desArgs;
      for(TemplateSpecializationType::iterator I = TST->begin(), E = TST->end();
          I != E; ++I) {
        QualType SubTy = I->getAsType();
       
        if (SubTy.isNull())
          continue;

        // Check if the type needs more desugaring and recurse.
        if (isa<TypedefType>(SubTy) || isa<TemplateSpecializationType>(SubTy)) {
          mightHaveChanged = true;
          desArgs.push_back(TemplateArgument(GetPartiallyDesugaredType(Ctx,
                                                                       SubTy,
                                                                  TypesToSkip)));
        } 
        else
          desArgs.push_back(TemplateArgument(SubTy));
      }
      
      // If desugaring happened allocate new type in the AST.
      if (mightHaveChanged) {
        QualType Result 
          = Ctx.getTemplateSpecializationType(TST->getTemplateName(), 
                                              desArgs.data(),
                                              desArgs.size(),
                                              TST->getCanonicalTypeInternal());
        return Result;
      }
    }
    return QT;   
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
} // end namespace utils
} // end namespace cling
