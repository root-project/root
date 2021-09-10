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

#include "ROOTUnitTestSupport.h"

#include <algorithm>
#include <iostream>
#include <iomanip>

namespace ROOTUnitTestSupport {

CheckDiagsRAII * CheckDiagsRAII::sActiveInstance = nullptr;

CheckDiagsRAII::~CheckDiagsRAII() {
   sActiveInstance = fOldInstance;
   ::SetErrorHandler(fOldErrorHandler);
   gInterpreter->ReportDiagnosticsToErrorHandler(/*enable=*/false);

   if (!fUnexpectedDiags.empty()) ADD_FAILURE() << "ROOTUnitTestSupport::CheckDiagsRAII: Unexpected diagnostic messages received.";

   const bool missingDiag = std::any_of(fExpectedDiags.begin(), fExpectedDiags.end(), [](const Diag_t & diag){ return !diag.optional && diag.receivedCount < 1; });
   if (missingDiag) ADD_FAILURE() << "ROOTUnitTestSupport::CheckDiagsRAII: Diagnostic message missing.";

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

}
