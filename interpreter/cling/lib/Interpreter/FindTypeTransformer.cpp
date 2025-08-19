//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Brock Mammen <brockmammen@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "FindTypeTransformer.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Sema/Sema.h>

using namespace clang;

namespace {
  using namespace cling;
  /// \brief Finds the decl for cling::LookupHelper::findType<T>()
  static FunctionTemplateDecl* FindFindTypeDecl(Sema& S) {
    NamespaceDecl* ClingNS = utils::Lookup::Namespace(&S, "cling");
    if (!ClingNS) {
      return nullptr;
    }
    TagDecl* LookupHelperDecl = utils::Lookup::Tag(&S, "LookupHelper", ClingNS);
    if (!LookupHelperDecl) {
      return nullptr;
    }
    TagDecl* LookupHelperDefinition =
        cast<clang::TagDecl>(LookupHelperDecl)->getDefinition();
    if (!LookupHelperDefinition) {
      return nullptr;
    }
    LookupResult R(S, &S.Context.Idents.get("findType"), SourceLocation(),
                   Sema::LookupOrdinaryName, Sema::ForVisibleRedeclaration);
    if (!S.LookupQualifiedName(R, LookupHelperDefinition)) {
      return nullptr;
    }
    R.resolveKind();
    for (NamedDecl* D : R) {
      if (FunctionTemplateDecl* FTD = dyn_cast<FunctionTemplateDecl>(D)) {
        return FTD;
      }
    }
    return nullptr;
  }

  class CallArgReplacer : public RecursiveASTVisitor<CallArgReplacer> {
    Sema& m_Sema;
    uintptr_t m_Ptr;

  public:
    CallArgReplacer(Sema& S, uintptr_t Ptr) : m_Sema(S), m_Ptr(Ptr) {}

    bool VisitCallExpr(CallExpr* CallExpr) {
      CallExpr->setArg(0, utils::Synthesize::CStyleCastPtrExpr(
                              &m_Sema, CallExpr->getArg(0)->getType(), m_Ptr));
      return false;
    }
  };

} // unnamed namespace

namespace cling {
  FindTypeTransformer::FindTypeTransformer(Sema* S) : ASTTransformer(S) {}

  FindTypeTransformer::~FindTypeTransformer() {}

  ASTTransformer::Result FindTypeTransformer::Transform(Decl* D) {
    if (FunctionTemplateDecl* FindTypeDecl = FindFindTypeDecl(*m_Sema);
        FindTypeDecl) {
      for (FunctionDecl* Spec : FindTypeDecl->specializations()) {
        const auto* TArgs = Spec->getTemplateSpecializationArgs();
        assert(TArgs->size() == 1 && "Expected 1 template argument");
        uintptr_t TPtr = (uintptr_t)TArgs->get(0).getAsType().getAsOpaquePtr();
        CallArgReplacer Replacer(*m_Sema, TPtr);
        Replacer.TraverseStmt(Spec->getBody());
      }
    }
    return Result(D, true);
  }
} // end namespace cling
