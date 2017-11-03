/// \file TDrawingOptsReader.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TDrawingOptsReader.hxx" // in src/

#include <ROOT/TColor.hxx>
#include <ROOT/TLogger.hxx>

#include <TROOT.h>
#include <TSystem.h>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

namespace {
}

TDrawingOptsReader::Attrs_t TDrawingOptsReader::ReadDefaults()
{
   Attrs_t ret;
   TDrawingOptsReader reader(ret);
   reader.AddFromStyleFile(std::string(TROOT::GetEtcDir()) + "/system.rootstylerc");
   reader.AddFromStyleFile(std::string(gSystem->GetHomeDirectory()) + "/.rootstylerc");
   reader.AddFromStyleFile(".rootstylerc");
   return ret;
}

ROOT::Experimental::TColor
TDrawingOptsReader::ParseColor(std::string_view attr, const ROOT::Experimental::TColor &deflt)
{
   auto iAttr = fAttrs.find(std::string(attr));
   if (iAttr == fAttrs.end())
      return deflt;
   return ROOT::Experimental::TColor::kBlack;
}

long long TDrawingOptsReader::ParseInt(std::string_view /*attr*/, long long /*deflt*/,
   std::vector<std::string_view> /*opts*/ /*= {}*/)
{
   // TODO: implement!
   return 0;
}

double TDrawingOptsReader::ParseFP(std::string_view /*attr*/, double /*deflt*/)
{
   // TODO: implement!
   return 0.;
}

bool TDrawingOptsReader::AddFromStyleFile(std::string_view /*filename*/)
{
   // TODO - implement!
   return false;
}
