// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RRawFile.hxx"

namespace ROOT {
namespace Detail {

ROOT::Detail::RRawFile::RRawFile(
   const std::string &protocol,
   const std::string &location,
   ROOT::Detail::RRawFile::ROptions options)
   : fProtocol(protocol)
   , fLocation(location)
   , fOptions(options)
   , fFilePos(0)
{
}


ROOT::Detail::RRawFile::~RRawFile()
{
}

ROOT::Detail::RRawFile* ROOT::Detail::RRawFile::Create(std::string_view url, ROptions options)
{
   return nullptr;
}



} // namespace Detail
} // namespace ROOT
