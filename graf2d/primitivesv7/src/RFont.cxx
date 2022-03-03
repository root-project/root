/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RFont.hxx"

#include "ROOT/RDisplayItem.hxx"

#include "TSystem.h"
#include "TString.h"
#include "TBase64.h"

#include <fstream>


using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////////
/// Set font source as URL.
/// Can be external url or path in the ROOT THttpServer
/// By default woff2 format is supposed

void RFont::SetUrl(const std::string &url, const std::string &fmt)
{
   if (url.empty())
      SetSrc("");
   else
      SetSrc(TString::Format("url('%s') format('%s')", url.c_str(), fmt.c_str()).Data());
}

////////////////////////////////////////////////////////////////////////
/// Set font source as file content.
/// Font file will be immediately read and converted in base64 string
/// By default woff2 format is supposed

void RFont::SetFile(const std::string &fname, const std::string &fmt)
{
   SetSrc("");

   if (fname.empty())
      return;

   TString fullname = fname;
   gSystem->ExpandPathName(fullname);

   if (!gSystem->AccessPathName(fullname.Data(), kReadPermission)) {
      std::ifstream is(fullname.Data(), std::ios::in | std::ios::binary);
      std::string res;
      if (is) {
         is.seekg(0, std::ios::end);
         res.resize(is.tellg());
         is.seekg(0, std::ios::beg);
         is.read((char *)res.data(), res.length());
         if (!is)
            res.clear();
      }

      if (!res.empty()) {
         TString base64 = TBase64::Encode(res.c_str(), res.length());
         SetSrc(TString::Format("url('data:application/font-%s;charset=utf-8;base64,%s') format('%s')", fmt.c_str(), base64.Data(), fmt.c_str()).Data());
      }
   }
}

////////////////////////////////////////////////////////////////////////
/// Set src attribute of font-face directly
/// Only for expert use

void RFont::SetSrc(const std::string &src)
{
   fSrc = src;
}
