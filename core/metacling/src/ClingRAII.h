// @(#)root/core/meta:$Id$
// Author: Axel Naumann November 2016

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "clang/Sema/Sema.h"
#include "cling/Utils/ParserStateRAII.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/ScopeInfo.h"

namespace ROOT {
 namespace Internal {

    ////////////////////////////////////////////////////////////////////////////////
    /// RAII used to store Parser, Sema, Preprocessor state for recursive parsing.
    struct ParsingStateRAII {
       struct SemaExprCleanupsRAII {
          decltype(clang::Sema::Cleanup) fCleanup;
          decltype(clang::Sema::ExprCleanupObjects) fExprCleanupObjects;
          decltype(clang::Sema::MaybeODRUseExprs) fMaybeODRUseExprs;
          decltype(clang::Sema::FunctionScopes) fFunctionScopes;
          decltype(clang::Sema::UndefinedButUsed) fUndefinedButUsed;
          // The start indices into FunctionScopes and InventedParameterInfos must
          // follow the swap: when we are called during a template instantiation,
          // Sema::ContextRAII has set them to the depth of the outer stack, which
          // is past the end of our fresh FunctionScopes.
          decltype(clang::Sema::FunctionScopesStart) fFunctionScopesStart = 0;
          decltype(clang::Sema::InventedParameterInfosStart) fInventedParameterInfosStart; // Set by the constructor to the size of Sema::InventedParameterInfos
          clang::Sema& fSema;
          void Swapem() {
             std::swap(fCleanup, fSema.Cleanup);
             std::swap(fExprCleanupObjects, fSema.ExprCleanupObjects);
             std::swap(fMaybeODRUseExprs, fSema.MaybeODRUseExprs);
             std::swap(fFunctionScopes, fSema.FunctionScopes);
             std::swap(fFunctionScopesStart, fSema.FunctionScopesStart);
             std::swap(fInventedParameterInfosStart, fSema.InventedParameterInfosStart);
             std::swap(fUndefinedButUsed, fSema.UndefinedButUsed);
          }
          SemaExprCleanupsRAII(clang::Sema& S):
             fInventedParameterInfosStart(S.InventedParameterInfos.size()), fSema(S) {
             fFunctionScopes.push_back(new clang::sema::FunctionScopeInfo(S.Diags));
             Swapem();
          };
          ~SemaExprCleanupsRAII() {
             Swapem();
             assert(fFunctionScopes.size() == 1 && "Expected only my FunctionScopeInfo.");
             delete fFunctionScopes.back();
          }
       };

       struct SemaParsingInitForAutoVarsRAII {
          using PIFAV_t = decltype(clang::Sema::ParsingInitForAutoVars);
          PIFAV_t& fSemaPIFAV;
          PIFAV_t fSavedPIFAV;
          SemaParsingInitForAutoVarsRAII(PIFAV_t& PIFAV): fSemaPIFAV(PIFAV) {
             fSavedPIFAV.swap(PIFAV);
          }
          ~SemaParsingInitForAutoVarsRAII() {
             fSavedPIFAV.swap(fSemaPIFAV);
          }
       };

       clang::Preprocessor::CleanupAndRestoreCacheRAII fCleanupRAII;
       clang::Parser::ParserCurTokRestoreRAII fSavedCurToken;
       cling::ParserStateRAII fParserRAII;

       // Buffer the delayed infos when doing recursive parsing.
       clang::Sema::DelayedInfoRAII fSemaInfoRAII;

       SemaExprCleanupsRAII fSemaExprCleanupsRAII;

       // We can't PushDeclContext, because we go up and the routine that pops
       // the DeclContext assumes that we drill down always.
       // We have to be on the global context. At that point we are in a
       // wrapper function so the parent context must be the global.
       clang::Sema::ContextAndScopeRAII fPushedDCAndS;

       SemaParsingInitForAutoVarsRAII fSemaParsingInitForAutoVarsRAII;

       clang::Sema::SavePendingInstantiationsRAII fPendingInstantiations;


       ParsingStateRAII(clang::Parser& parser, clang::Sema& sema):
          fCleanupRAII(sema.getPreprocessor()),
          fSavedCurToken(parser),
          fParserRAII(parser, false /*skipToEOF*/),
          fSemaInfoRAII(sema), fSemaExprCleanupsRAII(sema),
          fPushedDCAndS(sema, sema.getASTContext().getTranslationUnitDecl(),
                        sema.TUScope),
          fSemaParsingInitForAutoVarsRAII(sema.ParsingInitForAutoVars),
          fPendingInstantiations(sema)
       {
          // After we have saved the token reset the current one to something which
          // is safe (semi colon usually means empty decl)
          clang::Token& Tok = const_cast<clang::Token&>(parser.getCurToken());
          Tok.setKind(clang::tok::semi);
       }
    };

}
}
