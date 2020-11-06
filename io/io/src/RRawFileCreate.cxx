// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RConfig.h>
#include <ROOT/RRawFile.hxx>
#ifdef _WIN32
#include <ROOT/RRawFileWin.hxx>
#else
#include <ROOT/RRawFileUnix.hxx>
#endif

#include "TPluginManager.h"
#include "TROOT.h"

std::unique_ptr<ROOT::Internal::RRawFile>
ROOT::Internal::RRawFile::Create(std::string_view url, ROptions options)
{
   std::string transport = GetTransport(url);
   if (transport == "file") {
#ifdef _WIN32
      return std::unique_ptr<RRawFile>(new RRawFileWin(url, options));
#else
      return std::unique_ptr<RRawFile>(new RRawFileUnix(url, options));
#endif
   }
   if (transport == "http" || transport == "https") {
      if (TPluginHandler *h = gROOT->GetPluginManager()->FindHandler("ROOT::Internal::RRawFile")) {
         if (h->LoadPlugin() == 0) {
            return std::unique_ptr<RRawFile>(reinterpret_cast<RRawFile *>(h->ExecPlugin(2, &url, &options)));
         }
         throw std::runtime_error("Cannot load plugin handler for RRawFileDavix");
      }
      throw std::runtime_error("Cannot find plugin handler for RRawFileDavix");
   }
   throw std::runtime_error("Unsupported transport protocol: " + transport);
}
