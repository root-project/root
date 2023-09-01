// @(#)root/core/metacling:$Id$
// Author: Javier Lopez-Gomez   16/07/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingDiagnostics
#define ROOT_TClingDiagnostics

#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace clang {
   class LangOptions;
}

/// \brief Uses `clang::TextDiagnosticPrinter` to format diagnostics, which
///  are then passed to a user-specified function.
///
class TClingDelegateDiagnosticPrinter : public clang::TextDiagnosticPrinter {
public:
   typedef void (*handler_t)(clang::DiagnosticsEngine::Level Level,
                             const std::string &Info);
private:
   std::string fOS_out;
   llvm::raw_string_ostream fOS;
   handler_t fHandler;

public:
   TClingDelegateDiagnosticPrinter(clang::DiagnosticOptions *DiagOpts,
                                   clang::LangOptions &LangOpts, handler_t fn);
   ~TClingDelegateDiagnosticPrinter() override = default;

   void HandleDiagnostic(clang::DiagnosticsEngine::Level Level,
                         const clang::Diagnostic &Info) override;
};

#endif // ROOT_TClingDiagnostics
