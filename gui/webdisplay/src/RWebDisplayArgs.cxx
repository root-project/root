/// \file RWebDisplayArgs.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2018-10-24
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RWebDisplayArgs.hxx>
#include <ROOT/RWebWindow.hxx>
#include <ROOT/RConfig.hxx>

#include "TROOT.h"

/** \class ROOT::Experimental::RWebDisplayArgs
 * \ingroup webdisplay
 *
 * Holds different arguments for starting browser with RWebDisplayHandle::Display() method
 */

///////////////////////////////////////////////////////////////////////////////////////////
/// Default constructor - browser kind configured from gROOT->GetWebDisplay()

ROOT::Experimental::RWebDisplayArgs::RWebDisplayArgs()
{
   SetBrowserKind("");
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Constructor - browser kind specified as std::string
/// See SetBrowserKind() method for description of allowed parameters

ROOT::Experimental::RWebDisplayArgs::RWebDisplayArgs(const std::string &browser)
{
   SetBrowserKind(browser);
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Constructor - browser kind specified as const char *
/// See SetBrowserKind() method for description of allowed parameters

ROOT::Experimental::RWebDisplayArgs::RWebDisplayArgs(const char *browser)
{
   SetBrowserKind(browser);
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Constructor - specify window width and height

ROOT::Experimental::RWebDisplayArgs::RWebDisplayArgs(int width, int height, int x, int y, const std::string &browser)
{
   SetSize(width, height);
   SetPos(x, y);
   SetBrowserKind(browser);
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Constructor - specify master window and channel (if reserved already)

ROOT::Experimental::RWebDisplayArgs::RWebDisplayArgs(std::shared_ptr<RWebWindow> master, int channel)
{
   SetMasterWindow(master, channel);
}


///////////////////////////////////////////////////////////////////////////////////////////
/// Destructor

ROOT::Experimental::RWebDisplayArgs::~RWebDisplayArgs()
{
  // must be defined here to correctly call RWebWindow destructor
}

bool ROOT::Experimental::RWebDisplayArgs::SetSizeAsStr(const std::string &str)
{
   auto separ = str.find("x");
   if ((separ == std::string::npos) || (separ == 0) || (separ == str.length()-1)) return false;

   int width = 0, height = 0;

   try {
      width = std::stoi(str.substr(0,separ));
      height = std::stoi(str.substr(separ+1));
   } catch(...) {
       return false;
   }

   if ((width<=0) || (height<=0))
      return false;

   SetSize(width, height);
   return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Set browser kind as string argument
/// Recognized values:
///  chrome  - use Google Chrome web browser, supports headless mode from v60, default
///  firefox - use Mozilla Firefox browser, supports headless mode from v57
///   native - (or empty string) either chrome or firefox, only these browsers support batch (headless) mode
///  browser - default system web-browser, no batch mode
///   safari - Safari browser on Mac
///      cef - Chromium Embeded Framework, local display, local communication
///      qt5 - Qt5 WebEngine, local display, local communication
///    local - either cef or qt5
///   <prog> - any program name which will be started instead of default browser, like /usr/bin/opera

ROOT::Experimental::RWebDisplayArgs &ROOT::Experimental::RWebDisplayArgs::SetBrowserKind(const std::string &_kind)
{
   std::string kind = _kind;

   auto pos = kind.find("?");
   if (pos == 0) {
      SetUrlOpt(kind.substr(1));
      kind.clear();
   }

   pos = kind.find("size:");
   if (pos != std::string::npos) {
      auto epos = kind.find_first_of(" ;", pos+5);
      if (epos == std::string::npos) epos = kind.length();
      SetSizeAsStr(kind.substr(pos+5, epos-pos-5));
      kind.erase(pos, epos-pos);
   }

   // remove all trailing spaces
   while ((kind.length() > 0) && (kind[kind.length()-1] == ' '))
      kind.resize(kind.length()-1);

   // remove any remaining spaces?
   // kind.erase(remove_if(kind.begin(), kind.end(), std::isspace), kind.end());

   if (kind.empty())
      kind = gROOT->GetWebDisplay().Data();

   if (kind == "local")
      SetBrowserKind(kLocal);
   else if (kind.empty() || (kind == "native"))
      SetBrowserKind(kNative);
   else if (kind == "firefox")
      SetBrowserKind(kFirefox);
   else if ((kind == "chrome") || (kind == "chromium"))
      SetBrowserKind(kChrome);
   else if ((kind == "cef") || (kind == "cef3"))
      SetBrowserKind(kCEF);
   else if ((kind == "qt") || (kind == "qt5"))
      SetBrowserKind(kQt5);
   else if ((kind == "embed") || (kind == "embedded"))
      SetBrowserKind(kEmbedded);
   else if (!SetSizeAsStr(kind))
      SetCustomExec(kind);

   return *this;
}

/////////////////////////////////////////////////////////////////////
/// Returns configured browser name

std::string ROOT::Experimental::RWebDisplayArgs::GetBrowserName() const
{
   switch (GetBrowserKind()) {
      case kChrome: return "chrome";
      case kFirefox: return "firefox";
      case kNative: return "native";
      case kCEF: return "cef";
      case kQt5: return "qt5";
      case kLocal: return "local";
      case kStandard: return "default";
      case kEmbedded: return "embed";
      case kCustom:
          auto pos = fExec.find(" ");
          return (pos == std::string::npos) ? fExec : fExec.substr(0,pos);
   }

   return "";
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Assign window and channel id where other window will be embed

void ROOT::Experimental::RWebDisplayArgs::SetMasterWindow(std::shared_ptr<RWebWindow> master, int channel)
{
   SetBrowserKind(kEmbedded);
   fMaster = master;
   fMasterChannel = channel;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Append string to url options
/// Add "&" as separator if any options already exists

void ROOT::Experimental::RWebDisplayArgs::AppendUrlOpt(const std::string &opt)
{
   if (opt.empty()) return;

   if (!fUrlOpt.empty())
      fUrlOpt.append("&");

   fUrlOpt.append(opt);
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Returns full url, which is combined from URL and extra URL options
/// Takes into account "#" symbol in url - options are inserted before that symbol

std::string ROOT::Experimental::RWebDisplayArgs::GetFullUrl() const
{
   std::string url = GetUrl(), urlopt = GetUrlOpt();
   if (url.empty() || urlopt.empty()) return url;

   auto rpos = url.find("#");
   if (rpos == std::string::npos) rpos = url.length();

   if (url.find("?") != std::string::npos)
      url.insert(rpos, "&");
   else
      url.insert(rpos, "?");
   url.insert(rpos+1, urlopt);

   return url;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Configure custom web browser
/// Either just name of browser which can be used like "opera"
/// or full execution string which must includes $url like "/usr/bin/opera $url"

void ROOT::Experimental::RWebDisplayArgs::SetCustomExec(const std::string &exec)
{
   SetBrowserKind(kCustom);
   fExec = exec;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// returns custom executable to start web browser

std::string ROOT::Experimental::RWebDisplayArgs::GetCustomExec() const
{
   if (GetBrowserKind() != kCustom)
      return "";

#ifdef R__MACOSX
   if ((fExec == "safari") || (fExec == "Safari"))
      return "open -a Safari";
#endif

   return fExec;
}
