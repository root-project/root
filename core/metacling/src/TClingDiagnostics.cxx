// @(#)root/core/metacling:$Id$
// Author: Javier Lopez-Gomez   16/07/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClingDiagnostics.h"

TClingDelegateDiagnosticPrinter::TClingDelegateDiagnosticPrinter
(clang::DiagnosticOptions *DiagOpts, clang::LangOptions &LangOpts, handler_t fn)
  : TextDiagnosticPrinter(fOS, DiagOpts), fOS(fOS_out), fHandler(fn)
{
   // Required to initialize the internal `clang::TextDiagnostic` instance.
   TextDiagnosticPrinter::BeginSourceFile(LangOpts, nullptr);
}

void
TClingDelegateDiagnosticPrinter::HandleDiagnostic(clang::DiagnosticsEngine::Level Level,
                                                  const clang::Diagnostic &Info)
{
   fOS_out.clear();
   TextDiagnosticPrinter::HandleDiagnostic(Level, Info);
   fHandler(Level, fOS.str());
}
