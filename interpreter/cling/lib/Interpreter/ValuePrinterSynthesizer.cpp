//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "ValuePrinterSynthesizer.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/raw_os_ostream.h"

#include <iostream>

using namespace clang;

namespace cling {

  ValuePrinterSynthesizer::ValuePrinterSynthesizer(clang::Sema* S)
    : WrapperTransformer(S), m_Context(&S->getASTContext()),
      m_LookupResult(nullptr) { }


  // pin the vtable here.
  ValuePrinterSynthesizer::~ValuePrinterSynthesizer()
  { }

  ASTTransformer::Result ValuePrinterSynthesizer::Transform(clang::Decl* D) {
    if (getCompilationOpts().ValuePrinting == CompilationOptions::VPDisabled)
      return Result(D, true);

    FunctionDecl* FD = cast<FunctionDecl>(D);
    assert(utils::Analyze::IsWrapper(FD) && "Expected wrapper");
    if (tryAttachVP(FD))
      return Result(FD, true);
    return Result(0, false);
  }

  bool ValuePrinterSynthesizer::tryAttachVP(FunctionDecl* FD) {
    // We have to be able to mark the expression for printout. There are
    // three scenarios:
    // 0: Expression printing disabled - don't do anything just exit.
    // 1: Expression printing enabled - print no matter what.
    // 2: Expression printing auto - analyze - rely on the omitted ';' to
    //    not produce the suppress marker.
    int indexOfLastExpr = -1;
    Expr* To = utils::Analyze::GetOrCreateLastExpr(FD, &indexOfLastExpr,
                                                   /*omitDS*/false,
                                                   m_Sema);
    if (To) {
      // Update the CompoundStmt body, avoiding alloc/dealloc of all the el.
      CompoundStmt* CS = cast<CompoundStmt>(FD->getBody());
      assert(CS && "Missing body?");

      switch (getCompilationOpts().ValuePrinting) {
      case CompilationOptions::VPDisabled:
        assert(0 && "Don't wait that long. Exit early!");
        break;
      case CompilationOptions::VPEnabled:
        break;
      case CompilationOptions::VPAuto: {
        // FIXME: Propagate the flag to the nested transactions also, they
        // must have the same CO as their parents.
        getCompilationOpts().ValuePrinting = CompilationOptions::VPEnabled;
        if ((int)CS->size() > indexOfLastExpr+1
            && (*(CS->body_begin() + indexOfLastExpr + 1))
            && isa<NullStmt>(*(CS->body_begin() + indexOfLastExpr + 1))) {
          // If next is NullStmt disable VP is disabled - exit. Signal this in
          // the CO of the transaction.
          getCompilationOpts().ValuePrinting = CompilationOptions::VPDisabled;
        }
        if (getCompilationOpts().ValuePrinting
            == CompilationOptions::VPDisabled)
          return true;
      }
        break;
      }

      // We can't PushDeclContext, because we don't have scope.
      Sema::ContextRAII pushedDC(*m_Sema, FD);

      if (To) {
        // Strip the parenthesis if any
        if (ParenExpr* PE = dyn_cast<ParenExpr>(To))
          To = PE->getSubExpr();

        Expr* Result = 0;
        // if (!m_Sema->getLangOpts().CPlusPlus)
        //   Result = SynthesizeVP(To);

        if (Result)
          *(CS->body_begin()+indexOfLastExpr) = Result;
      }
      // Clear the artificial NullStmt-s
      if (!ClearNullStmts(FD)) {
        // FIXME: Why it is here? Shouldn't it be in DeclExtractor?
        // if no body remove the wrapper
        DeclContext* DC = FD->getDeclContext();
        Scope* S = m_Sema->getScopeForContext(DC);
        if (S)
          S->RemoveDecl(FD);
        DC->removeDecl(FD);
      }
    }
    else // if nothing to attach to set the CO's ValuePrinting to disabled.
      getCompilationOpts().ValuePrinting = CompilationOptions::VPDisabled;
    return true;
  }

  // We need to artificially create:
  // cling_PrintValue(void* (ASTContext)C, void* (Expr)E, const void* (&i)
  Expr* ValuePrinterSynthesizer::SynthesizeVP(Expr* E) {
    QualType QT = E->getType();
    // For now we skip void and function pointer types.
    if (!QT.isNull() && (QT->isVoidType() || QT->isFunctionType()))
      return 0;

    // Find cling_PrintValue
    if (!m_LookupResult)
      FindAndCacheRuntimeLookupResult(E->getBeginLoc());


    Expr* VoidEArg = utils::Synthesize::CStyleCastPtrExpr(m_Sema,
                                                          m_Context->VoidPtrTy,
                                                          (uintptr_t)E);
    Expr* VoidCArg = utils::Synthesize::CStyleCastPtrExpr(m_Sema,
                                                          m_Context->VoidPtrTy,
                                                          (uintptr_t)m_Context);

    SourceLocation NoSLoc = SourceLocation();
    Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
    if (!QT->isPointerType()) {
      while(ImplicitCastExpr* ICE = dyn_cast<ImplicitCastExpr>(E))
        E = ICE->getSubExpr();
      E = m_Sema->BuildUnaryOp(S, NoSLoc, UO_AddrOf, E).get();
    }

    llvm::SmallVector<Expr*, 4> CallArgs;
    CallArgs.push_back(VoidEArg);
    CallArgs.push_back(VoidCArg);
    CallArgs.push_back(E);

    CXXScopeSpec CSS;
    Expr* unresolvedLookup
      = m_Sema->BuildDeclarationNameExpr(CSS, *m_LookupResult,
                                         /*ADL*/ false).get();

    Expr* Result = m_Sema->ActOnCallExpr(S, unresolvedLookup, E->getBeginLoc(),
                                         CallArgs, E->getEndLoc()).get();
    assert(Result && "Cannot create value printer!");

    return Result;
  }


  unsigned ValuePrinterSynthesizer::ClearNullStmts(FunctionDecl* FD) {
    CompoundStmt* CS = cast<CompoundStmt>(FD->getBody());
    assert(CS && "Missing body?");

    llvm::SmallVector<Stmt*, 8> FBody;
    for (auto&& child: CS->children())
      if (!isa<NullStmt>(child))
        FBody.push_back(child);

    // If body would be empty, return early - the function will be removed.
    if (FBody.empty())
      return 0;

    if (CS->size() != FBody.size()) {
      auto BodyCS = CompoundStmt::Create(*m_Context, FBody, CS->getLBracLoc(),
                                         CS->getRBracLoc());
      FD->setBody(BodyCS);
    }

    return FBody.size();
  }

  void ValuePrinterSynthesizer::FindAndCacheRuntimeLookupResult(
                                                  SourceLocation sourceLoc) {
    assert(!m_LookupResult && "Called multiple times!?");

    DeclarationName PVName = &m_Context->Idents.get("cling_PrintValue");
    m_LookupResult = new LookupResult(*m_Sema, PVName, sourceLoc,
                                      Sema::LookupOrdinaryName,
                                      Sema::ForVisibleRedeclaration);

    Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
    m_Sema->LookupName(*m_LookupResult, S);
    assert(!m_LookupResult->empty() && "Cannot find cling_PrintValue(...)");
  }

} // namespace cling
