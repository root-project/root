/// \file base_handler.cxx
/// \ingroup WebGui
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#if !defined(_MSC_VER)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#include "base_handler.h"

#include "TString.h"
#include "TError.h"
#include "TBase64.h"
#include "TEnv.h"

#include <ROOT/RLogger.hxx>

#include <sstream>
#include <string>

#include "include/base/cef_bind.h"
#include "include/cef_version.h"
#include "include/cef_app.h"
#include "include/views/cef_browser_view.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_closure_task.h"
#include "include/wrapper/cef_helpers.h"

BaseHandler::BaseHandler(THttpServer *serv) : fServer(serv), is_closing_(false)
{
   fConsole = gEnv->GetValue("WebGui.Console", (int)0);
}

void BaseHandler::OnAfterCreated(CefRefPtr<CefBrowser> browser)
{
   CEF_REQUIRE_UI_THREAD();

   // Add to the list of existing browsers.
   browser_list_.push_back(browser);
}

bool BaseHandler::DoClose(CefRefPtr<CefBrowser> browser)
{
   CEF_REQUIRE_UI_THREAD();

   // Closing the main window requires special handling. See the DoClose()
   // documentation in the CEF header for a detailed description of this
   // process.
   if (browser_list_.size() == 1) {
      // Set a flag to indicate that the window close should be allowed.
      is_closing_ = true;
   }

   // Allow the close. For windowed browsers this will result in the OS close
   // event being sent.
   return false;
}

void BaseHandler::OnBeforeClose(CefRefPtr<CefBrowser> browser)
{
   CEF_REQUIRE_UI_THREAD();

   // Remove from the list of existing browsers.
   BrowserList::iterator bit = browser_list_.begin();
   for (; bit != browser_list_.end(); ++bit) {
      if ((*bit)->IsSame(browser)) {
         browser_list_.erase(bit);
         break;
      }
   }

   if (browser_list_.empty()) {

      // All browser windows have closed. Quit the application message loop.

      CefQuitMessageLoop();
   }
}

#if CEF_COMMIT_NUMBER > 2230

#include "include/cef_parser.h"

namespace {

   // Returns a data: URI with the specified contents.
   std::string GetDataURI(const std::string& data, const std::string& mime_type)
   {
      return "data:" + mime_type + ";base64," +
              CefURIEncode(CefBase64Encode(data.data(), data.size()), false)
               .ToString();
   }

}

#endif

void BaseHandler::OnLoadError(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, ErrorCode errorCode,
                              const CefString &errorText, const CefString &failedUrl)
{
   CEF_REQUIRE_UI_THREAD();

   // Don't display an error for downloaded files.
   if (errorCode == ERR_ABORTED)
      return;

   // Display a load error message.
   std::stringstream ss;
   ss << "<html><body bgcolor=\"white\">"
         "<h2>Failed to load URL "
      << std::string(failedUrl) << " with error " << std::string(errorText) << " (" << errorCode
      << ").</h2></body></html>";
#if CEF_COMMIT_NUMBER > 2230
   frame->LoadURL(GetDataURI(ss.str(), "text/html"));
#else
   frame->LoadString(ss.str(), failedUrl);
#endif
}

void BaseHandler::CloseAllBrowsers(bool force_close)
{
   if (!CefCurrentlyOn(TID_UI)) {
      // Execute on the UI thread.
      CefPostTask(TID_UI, base::Bind(&BaseHandler::CloseAllBrowsers, this, force_close));
      return;
   }

   if (browser_list_.empty())
      return;

   BrowserList::const_iterator it = browser_list_.begin();
   for (; it != browser_list_.end(); ++it)
      (*it)->GetHost()->CloseBrowser(force_close);
}

void BaseHandler::OnTitleChange(CefRefPtr<CefBrowser> browser, const CefString &title)
{
   CEF_REQUIRE_UI_THREAD();
}

bool BaseHandler::OnConsoleMessage(CefRefPtr<CefBrowser> browser,
                                  cef_log_severity_t level,
                                  const CefString &message, const CefString &source,
                                  int line)
{
   switch (level) {
   case LOGSEVERITY_WARNING:
      if (fConsole > -1)
         R__WARNING_HERE("CEF") << Form("CEF: %s:%d: %s", source.ToString().c_str(), line, message.ToString().c_str());
      break;
   case LOGSEVERITY_ERROR:
      if (fConsole > -2)
         R__ERROR_HERE("CEF") << Form("CEF: %s:%d: %s", source.ToString().c_str(), line, message.ToString().c_str());
      break;
   default:
      if (fConsole > 0)
         R__DEBUG_HERE("CEF") << Form("CEF: %s:%d: %s", source.ToString().c_str(), line, message.ToString().c_str());
      break;
   }

   return true;
}

