/// \brief This file contains a specialised ROOT message handler to test for diagnostic in unit tests.
///
/// \author Stephan Hageboeck <stephan.hageboeck@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TestSupport.hxx"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>

namespace ROOT {
namespace TestSupport {

/// Error handler for gtests that generates failures for every received diagnostic > kInfo when this file is linked to.
static struct ForbidDiagnostics {
   ForbidDiagnostics() :
      sOldErrorHandler{ ::GetErrorHandler() } {
         SetErrorHandler(ForbidDiagnostics::handler);
      }

   ~ForbidDiagnostics() {
      ::SetErrorHandler(sOldErrorHandler);
   }

   /// Diagnostic handler that's installed for all google tests.
   /// It will generate a test failure when a diagnostic message is issued.
   static void handler(int level, bool abort,
         const char *location,
         const char *msg) {
      if (level <= gErrorIgnoreLevel) return;
      if (level <= kInfo) {
         std::cerr << "Diagnostic in '" << location << "': " << msg << std::endl;
         return;
      }

      if (abort) {
         std::cerr << "ROOT::TestSupport::ForbidDiagnostics::handler(): Forced to abort because of diagnostic with severity "
            << level << " in '" << location << "' reading '" << msg << "'\n";
         ::abort();
      }

      // FIXME: Windows has problem finding PCMs.
      if (level == kError && strcmp(location, "TCling::LoadPCM") == 0 && strstr(msg, "file does not exist") != nullptr) {
        std::cerr << "Error in " << location << " " << msg << std::endl;
        return;
      }

      // FIXME: RNTuple warns that it's in beta stage.
      if (level == kWarning
          && strstr(msg, "The RNTuple file format will change. Do not store real data with this version of RNTuple!") != nullptr) {
        std::cerr << "Warning in " << location << " " << msg << std::endl;
        return;
      }
      if (level == kWarning
          && strstr(msg, "Pre-release format version: RC ") != nullptr) {
        std::cerr << "Warning in " << location << " " << msg << std::endl;
        return;
      }

      // FIXME: DOAS backend is exprimental.
      if (level == kWarning
          && strstr(msg, "The DAOS backend is experimental and still under development") != nullptr) {
        std::cerr << "Warning in " << location << " " << msg << std::endl;
        return;
      }

      // FIXME: RooNaNPacker warns about not being implemented for big endian
      if (level == kWarning
	  && strcmp(msg, "Fast recovery from undefined function values only implemented for little-endian machines. If necessary, request an extension of functionality on https://root.cern") == 0) {
        std::cerr << "Warning in " << location << " " << msg << std::endl;
        return;
      }

      FAIL() << "Received unexpected diagnostic of severity "
         << level
         << " at '" << location << "' reading '" << msg << "'.\n"
         << "Suppress those using ROOT/TestSupport.hxx";
   }

   ErrorHandlerFunc_t const sOldErrorHandler;
} noDiagCheckerInstance;


CheckDiagsRAII * CheckDiagsRAII::sActiveInstance = nullptr;

CheckDiagsRAII::~CheckDiagsRAII() {
   sActiveInstance = fOldInstance;
   ::SetErrorHandler(fOldErrorHandler);
   gInterpreter->ReportDiagnosticsToErrorHandler(/*enable=*/false);

   if (!fUnexpectedDiags.empty()) ADD_FAILURE() << "ROOT::TestSupport::CheckDiagsRAII: Unexpected diagnostic messages received.";

   const bool missingDiag = std::any_of(fExpectedDiags.begin(), fExpectedDiags.end(), [](const Diag_t & diag){ return !diag.optional && diag.receivedCount < 1; });
   if (missingDiag) ADD_FAILURE() << "ROOT::TestSupport::CheckDiagsRAII: Diagnostic message missing.";

   if (!fUnexpectedDiags.empty() || missingDiag) {
      std::cout << "-------------------------\nPre-registered messages:\n";
      printDiags(fExpectedDiags);
      std::cout << "Unexpected messages received:\n";
      printDiags(fUnexpectedDiags);
      std::cout << "-------------------------" << std::endl;
   }
}

/// Search list of expected diagnostics for given arguments, and increase the receivedCount.
/// If no matching predefined diagnostic is found, this will trigger an unexpected diagnostic error on exit.
void CheckDiagsRAII::checkDiag(int severity, const char * location, const char * msg) {
   // Check that this received diagnostics was expected
   const std::string message = msg;
   const auto expectedMatch = std::find_if(fExpectedDiags.begin(), fExpectedDiags.end(), [=](const Diag_t& expectedDiag){
         return expectedDiag.severity == severity
         && strstr(location, expectedDiag.location.c_str()) != nullptr
         && (expectedDiag.matchFullString ? expectedDiag.message == message : (message.find(expectedDiag.message) != std::string::npos));
         });

    if (expectedMatch != fExpectedDiags.end()) {
       expectedMatch->receivedCount += 1;
    } else if (severity <= kInfo) {
       fOldErrorHandler(severity, false, location, msg);
    } else {
       fUnexpectedDiags.push_back({severity, location, std::move(message)});
    }
}

void CheckDiagsRAII::printDiags(std::vector<Diag_t> const & diags) const {
   std::map<int, std::string> dict = {
      {kInfo, "kInfo"},
      {kWarning, "kWarning"},
      {kError, "kError"},
      {kSysError, "kSysError"}
   };

   for (auto const & diag : diags) {
      std::cout << std::setw(10) << dict[diag.severity] << "\t";
      if (diag.receivedCount >=  0) {
         std::cout << diag.receivedCount << "x received\t"
            << "(" << (diag.optional ? "optional" : "required") << ", " << (diag.matchFullString ? "fullMatch" : "subMatch") << ")\t";
      }
      std::cout << "'" << diag.location << "' msg='" << diag.message << "'\n";
   }
}

} } //ROOT::TestSupport
