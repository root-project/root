/// \file RWebBrowserArgs.cxx
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

#include <ROOT/RWebBrowserArgs.hxx>

#include "TROOT.h"

/** \class ROOT::Experimental::RWebBrowserArgs
 * \ingroup webdisplay
 *
 * Holds different arguments for starting browser with RWebDisplayHandle::Display() method
 */

ROOT::Experimental::RWebBrowserArgs::RWebBrowserArgs()
{
   SetBrowserKind(gROOT->GetWebDisplay().Data());
}

ROOT::Experimental::RWebBrowserArgs::RWebBrowserArgs(const std::string &browser)
{
   SetBrowserKind(browser);
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Set browser kind using string argument

void ROOT::Experimental::RWebBrowserArgs::SetBrowserKind(const std::string &kind)
{
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

///////////////////////////////////////////////////////////////////////////////////////////
/// Returns full url, which is combined from URL and extra URL options

std::string ROOT::Experimental::RWebBrowserArgs::GetFullUrl() const
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
