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

   static
   NestedNameSpecifier *GetPartiallyDesugaredNNS(const ASTContext& Ctx, 
                                                  NestedNameSpecifier *scope, 
                                                  const llvm::SmallSet<const Type*, 4>& TypesToSkip){
     // Desugar the scope qualifier if needed.

     const Type *scope_type = scope->getAsType();
     if (scope_type) {
        // this is not a namespace, so we might need to desugar
        NestedNameSpecifier *outer_scope = scope->getPrefix();
        if (outer_scope) {
           outer_scope = GetPartiallyDesugaredNNS(Ctx, outer_scope, TypesToSkip);
        }

        QualType desugared = Transform::GetPartiallyDesugaredType(Ctx,
                                                                  QualType(scope_type,0),
                                                                  TypesToSkip);
        // NOTE: Should check whether the type has changed or not.
        return NestedNameSpecifier::Create(Ctx,outer_scope,false /* template keyword wanted */,
                                           desugared.getTypePtr());
     }
     return scope;
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
    NestedNameSpecifier *prefix = 0;
    const ElaboratedType *etype = llvm::dyn_cast<ElaboratedType>(QT.getTypePtr());
    if (etype) {
       // We have to also desugar the prefix.
 
       // Unfortunately this only look at one level, so in
       //   A<type1>::B<type2>::C<type3>::D<type4>
       // A<type1>::B<type2> will be left alone.
       // while C<type3> is desugared by the following code
       // and D<type4> is desugared by the loop below.
       prefix = GetPartiallyDesugaredNNS(Ctx, etype->getQualifier(), TypesToSkip);
       QT = etype->getNamedType();
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
       
        if (SubTy.isNull()) {
           desArgs.push_back(*I);
           continue;
        }

        // Check if the type needs more desugaring and recurse.
        if (isa<TypedefType>(SubTy) || isa<TemplateSpecializationType>(SubTy)) {
          mightHaveChanged = true;
          desArgs.push_back(TemplateArgument(GetPartiallyDesugaredType(Ctx,
                                                                       SubTy,
                                                                  TypesToSkip)));
         } else 
            desArgs.push_back(*I);
      }
      
      // If desugaring happened allocate new type in the AST.
      if (mightHaveChanged) {
        QT = Ctx.getTemplateSpecializationType(TST->getTemplateName(), 
                                              desArgs.data(),
                                              desArgs.size(),
                                              TST->getCanonicalTypeInternal());
      }
    }
    if (prefix) {
       QT = Ctx.getElaboratedType(ETK_None,prefix,QT);
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
