/// \file ROOT/RWebBrowserArgs.hxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2018-10-24
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RWebBrowserArgs
#define ROOT7_RWebBrowserArgs

#include <string>

class THttpServer;

namespace ROOT {
namespace Experimental {

/// Argument used in RWebWindow::Show() method
class RWebBrowserArgs {
public:

   enum EBrowserKind {
      kDefault,   // default settings provided with --web argument when started ROOT
      kChrome,    // Google Chrome browser
      kFirefox,   // Mozilla Firefox browser
      kNative,    // either Chrome or Firefox - both support major functionality
      kCEF,       // Chromium Embedded Framework - local display with CEF libs
      kQt5,       // QWebEngine libraries - Chrome code packed in qt5
      kLocal,     // either CEF or Qt5 - both runs on local display without real http server
      kStandard,  // standard system web browser, not recognized by ROOT, without batch mode
      kCustom     // custom web browser, execution string should be provided
   };

protected:

   EBrowserKind fKind{kDefault};  ///<! name of web browser used for display
   std::string fUrl;              ///<! URL to display
   bool fHeadless{false};         ///<! is browser runs in headless mode
   THttpServer *fServer{nullptr}; ///<! http server which handle all requests
   int fWidth{0};                 ///<! custom window width, when not specified - used RWebWindow geometry
   int fHeight{0};                ///<! custom window width, when not specified - used RWebWindow geometry
   std::string fUrlOpt;           ///<! extra URL options, which are append to window URL
   std::string fExec;             ///<! string to run browser, used with kCustom type
   void *fDriverData{nullptr};    ///<! special data delivered to driver, can be used for QWebEngine

   RWebBrowserArgs();

   RWebBrowserArgs(const std::string &browser);

   void SetBrowserKind(const std::string &kind);
   void SetBrowserKind(EBrowserKind kind) { fKind = kind; }
   EBrowserKind GetBrowserKind() const { return fKind; }

   /// returns true if local display like CEF or Qt5 QWebEngine should be used
   bool IsLocalDisplay() const
   {
      return (GetBrowserKind() == kLocal) || (GetBrowserKind() == kCEF) || (GetBrowserKind() == kQt5);
   }

   /// returns true if browser supports headless mode
   bool IsSupportHeadless() const
   {
      return (GetBrowserKind() == kNative) || (GetBrowserKind() == kFirefox) || (GetBrowserKind() == kChrome);
   }

   void SetUrl(const std::string &url) { fUrl = url; }
   std::string GetUrl() const { return fUrl; }

   void SetUrlOpt(const std::string &opt) { fUrlOpt = opt; }
   std::string GetUrlOpt() const { return fUrlOpt; }

   std::string GetFullUrl() const;

   void SetHeadless(bool on = true) { fHeadless = on; }
   bool IsHeadless() const { return fHeadless; }

   void SetHttpServer(THttpServer *serv) { fServer = serv; }
   THttpServer *GetHttpServer() const { return fServer; }

   void SetWidth(int w = 0) { fWidth = w; }
   void SetHeight(int h = 0) { fHeight = h; }

   int GetWidth() const { return fWidth; }
   int GetHeight() const { return fHeight; }

   void SetCustomExec(const std::string &exec)
   {
      SetBrowserKind(kCustom);
      fExec = exec;
   }

   std::string GetCustomExec() const { return GetBrowserKind() == kCustom ? fExec : ""; }

   void SetDriverData(void *data) { fDriverData = data; }
   void *GetDriverData() const { return fDriverData; }
};

}
}



#endif
