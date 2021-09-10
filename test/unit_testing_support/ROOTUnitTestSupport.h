/// \file ROOTUnitTestSupport.h
///
/// \brief The file contains facilities allowing easier writing of in-tree unit tests
///
/// \author Pratyush Das <reikdas@gmail.com>
/// \author Vassil Vassilev <vvasilev@cern.ch>
/// \author Stephan Hageboeck <stephan.hageboeck@cern.ch>
///
/// \date April, 2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_UNITTESTSUPPORT_H
#define ROOT_UNITTESTSUPPORT_H

#include "TError.h"
#include "TInterpreter.h"

#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

namespace ROOTUnitTestSupport {

/// \brief Allows a user function to catch and filter/analyse ROOT and cling diagnostics, e.g.
/// ```c++
/// FilterDiagsRAII RAII([] (int level, Bool_t abort,
///                          const char *location, const char *msg) {
///       EXPECT_THAT(msg, Not(HasSubstr("-Wunused-result")));
///    });
/// ```
class FilterDiagsRAII {
   ErrorHandlerFunc_t fPrevHandler;
   public:
   FilterDiagsRAII(ErrorHandlerFunc_t fn) : fPrevHandler(::GetErrorHandler()) {
      ::SetErrorHandler(fn);
      gInterpreter->ReportDiagnosticsToErrorHandler();
   }
   ~FilterDiagsRAII() {
      gInterpreter->ReportDiagnosticsToErrorHandler(/*enable=*/false);
      ::SetErrorHandler(fPrevHandler);
   }
};

/// Install a ROOT diagnostic handler to analyse diagnostics.
/// It will record all diagnostics during its lifetime, and analyse them at destruction.
/// Required and/or optional diagnostics need to be predefined with expected location and message.
/// Unexpected or missing diagnostics will lead to gtest failures.
///
/// Example:
/// ```c++
/// CheckDiagsRAII diagRAII{CheckDiagsRAII::EDK_Error, "TFile::TFile", "<Expected message>"};
/// diagRAII.optionalDiag(kInfo, "TFile::TFile", "Message that is issued only sometimes");
/// <test code>
///
/// ```
class CheckDiagsRAII {
   public:
      /// Register this instance as diagnostic handler.
      /// With no further action, any diagnostic will lead to a test failure.
      CheckDiagsRAII():
         fOldInstance(sActiveInstance),
         fOldErrorHandler(::GetErrorHandler())
   {
      sActiveInstance = this;
      ::SetErrorHandler(CheckDiagsRAII::callback);
      gInterpreter->ReportDiagnosticsToErrorHandler(/*enable=*/true);
   }

      /// Construct from ROOT's `kWarning, kError, ...` and strings specifying location and message.
      CheckDiagsRAII(int severity, std::string inRoutine, std::string E)
         : CheckDiagsRAII()
      {
         requiredDiag(severity, inRoutine, E);
      }

      ~CheckDiagsRAII();

      /// Register a new diagnostic to check for.
      /// \param severity One of kInfo kWarning kError kSysError.
      /// \param location Function name where the diagnostic should be issued.
      /// \param message Diagnostic message.
      /// \param matchFullMessage If true, the message must be exactly identical.
      ///        If false, it's sufficient that `message` is a substring of the diagnostic message.
      void requiredDiag(int severity, std::string location, std::string message, bool matchFullMessage = true) {
         if (severity != kInfo && severity != kWarning && severity != kError && severity != kSysError)
            throw std::invalid_argument("ExpectedDiagRAII::requiredDiag(): severity is none of kInfo, kWarning, kError, kSysError");

         fExpectedDiags.push_back({severity, std::move(location), std::move(message), matchFullMessage, false, 0});
      }

      /// Register a diagnostic that can, but need not necessarily be issued.
      /// \param severity One of kInfo kWarning kError kSysError.
      /// \param location Function name where the diagnostic should be issued.
      /// \param message Diagnostic message.
      /// \param matchFullMessage If true, the message must be exactly identical.
      ///        If false, it's sufficient that `message` is a substring of the diagnostic message.
      void optionalDiag(int severity, std::string location, std::string message, bool matchFullMessage = true) {
         if (severity != kInfo && severity != kWarning && severity != kError && severity != kSysError)
            throw std::invalid_argument("ExpectedDiagRAII::optionalDiag(): severity is none of kInfo, kWarning, kError, kSysError");

         fExpectedDiags.push_back({severity, std::move(location), std::move(message), matchFullMessage, true, 0});
      }

   private:
      struct Diag_t {
         int severity;
         std::string location;
         std::string message;
         bool matchFullString = true;
         bool optional = false;
         int receivedCount = -1;
      };

      /// Message handler that hands over all diagnostics to the currently active instance.
      static void callback(int severity, bool abort, const char * location, const char * msg) {
         if (sActiveInstance) {
            sActiveInstance->checkDiag(severity, location, msg);
         } else {
            throw std::logic_error("ExpectedDiagRAII::callback called without an active message handler.");
         }

         if (abort) {
            std::cerr << "ROOTUnitTestSupport::CheckDiagsRAII: Forced to abort because of diagnostic with severity "
               << severity << " in '" << location << "' reading '" << msg << "'\n";
            ::abort();
         }
      }

      /// Check all received diags against list of expected ones.
      void checkDiag(int severity, const char * location, const char * msg);
      /// Print the diags in `diags` to the terminal.
      void printDiags(std::vector<Diag_t> const & diags) const;

      std::vector<Diag_t> fExpectedDiags;
      std::vector<Diag_t> fUnexpectedDiags;

      CheckDiagsRAII * const fOldInstance;       /// Last active handler in case handlers are nested.
      ErrorHandlerFunc_t const fOldErrorHandler; /// Last active error handler function.

      static CheckDiagsRAII * sActiveInstance;   /// Instance that will receive ROOT's callbacks.
};

}

#define ROOT_EXPECT_ERROR(expression, where, expected_diag )       \
{                                                                  \
   using namespace ROOTUnitTestSupport;                            \
   CheckDiagsRAII EE(kError, where,                                \
         expected_diag);                                           \
   expression;                                                     \
}

#define ROOT_EXPECT_WARNING(expression, where, expected_diag)      \
{                                                                  \
   using namespace ROOTUnitTestSupport;                            \
   CheckDiagsRAII EE(kWarning, where,                              \
         expected_diag);                                           \
   expression;                                                     \
}

#define ROOT_EXPECT_INFO(expression, where, expected_diag)           \
{                                                                    \
   using namespace ROOTUnitTestSupport;                              \
   CheckDiagsRAII EE(kInfo, where,                                   \
         expected_diag);                                             \
   expression;                                                       \
}

#define ROOT_EXPECT_NODIAG(expression)                               \
{                                                                    \
   using namespace ROOTUnitTestSupport;                              \
   CheckDiagsRAII EE{};                                              \
   expression;                                                       \
}

#define ROOT_EXPECT_SYSERROR(expression, where, expected_diag)       \
{                                                                    \
   using namespace ROOTUnitTestSupport;                              \
   CheckDiagsRAII EE(kSysError, where,                               \
         expected_diag);                                             \
   expression;                                                       \
}

#endif // ROOT_UNITTESTSUPPORT_H
