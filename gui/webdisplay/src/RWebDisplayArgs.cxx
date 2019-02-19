/// \file RWebDisplayArgs.cxx
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

#include <ROOT/RWebDisplayArgs.hxx>

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

ROOT::Experimental::RWebDisplayArgs::RWebDisplayArgs(const std::string &browser)
{
   SetBrowserKind(browser);
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Constructor - browser kind specified as const char *

ROOT::Experimental::RWebDisplayArgs::RWebDisplayArgs(const char *browser)
{
   SetBrowserKind(browser);
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Set browser kind using string argument

void ROOT::Experimental::RWebDisplayArgs::SetBrowserKind(const std::string &_kind)
{
   std::string kind = _kind;

   auto pos = kind.find("?");
   if (pos == 0) {
      SetUrlOpt(kind.substr(1));
      kind.clear();
   }

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
   else if (kind == "cef")
      SetBrowserKind(kCEF);
   else if (kind == "qt5")
      SetBrowserKind(kQt5);
   else
      SetCustomExec(kind);
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
      case kCustom: return "custom";
   }

   return "";
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Returns full url, which is combined from URL and extra URL options

std::string ROOT::Experimental::RWebDisplayArgs::GetFullUrl() const
{
   std::string url = GetUrl(), urlopt = GetUrlOpt();
   if (url.empty() || urlopt.empty()) return url;

   if (url.find("?") != std::string::npos)
      url.append("&");
   else
      url.append("?");
   url.append(urlopt);

   return url;
}
