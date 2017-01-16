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


namespace ROOT {
 namespace Internal {

    ////////////////////////////////////////////////////////////////////////////////
    /// RAII used to store Parser, Sema, Preprocessor state for recursive parsing.
    struct ParsingStateRAII {
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

       // We can't PushDeclContext, because we go up and the routine that pops
       // the DeclContext assumes that we drill down always.
       // We have to be on the global context. At that point we are in a
       // wrapper function so the parent context must be the global.
       clang::Sema::ContextAndScopeRAII fPushedDCAndS;

       SemaParsingInitForAutoVarsRAII fSemaParsingInitForAutoVarsRAII;

       ParsingStateRAII(clang::Parser& parser, clang::Sema& sema):
          fCleanupRAII(sema.getPreprocessor()),
          fSavedCurToken(parser),
          fParserRAII(parser, false /*skipToEOF*/),
          fSemaInfoRAII(sema),
          fPushedDCAndS(sema, sema.getASTContext().getTranslationUnitDecl(),
                        sema.TUScope),
          fSemaParsingInitForAutoVarsRAII(sema.ParsingInitForAutoVars)
       {
          // After we have saved the token reset the current one to something which
          // is safe (semi colon usually means empty decl)
          clang::Token& Tok = const_cast<clang::Token&>(parser.getCurToken());
          Tok.setKind(clang::tok::semi);
       }
    };

}
}