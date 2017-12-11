/// \file TDrawingOptsBase.cxx
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

#include "ROOT/TDrawingOptsBase.hxx"

#include "ROOT/TCanvas.hxx"
#include "ROOT/TColor.hxx"
#include "ROOT/TLogger.hxx"
#include "ROOT/TPad.hxx"

#include "TDrawingOptsReader.hxx" // in src/

#include <cstddef>
#include <unordered_map>

using namespace ROOT::Experimental;

TDrawingOptsBaseNoDefault::TDrawingOptsBaseNoDefault(TPadBase &pad, std::string_view configPrefix):
fCanvas(&pad.GetCanvas()), fName(configPrefix) {}

ROOT::Experimental::TPadBase &TDrawingOptsBaseNoDefault::GetDefaultCanvas(const TStyle &style)
{
   static std::unordered_map<std::string, TCanvas> sCanv;

   auto iCanv = sCanv.find(style.GetName());
   if (iCanv != sCanv.end())
      return iCanv->second;

   TCanvas &canv = sCanv[style.GetName()];
   canv.SetTitle(style.GetName());
   return canv;
}

bool TDrawingOptsBaseNoDefault::IsDefaultCanvas(const TPadBase &canvPad)
{
   if (const TCanvas* canv = dynamic_cast<const TCanvas*>(&canvPad)) {
      if (TStyle* style = TStyle::Get(canv->GetTitle()))
         return &GetDefaultCanvas(*style) == &canvPad;
   }
   return false;
}


TDrawingOptsBaseNoDefault::~TDrawingOptsBaseNoDefault()
{
   fColorIdx.Release(GetCanvas());
   fIntIdx.Release(GetCanvas());
   fFPIdx.Release(GetCanvas());
}

TDrawingOptsBaseNoDefault::TDrawingOptsBaseNoDefault(const TDrawingOptsBaseNoDefault &other)
   : fCanvas(other.fCanvas), fColorIdx(other.fColorIdx), fIntIdx(other.fIntIdx), fFPIdx(other.fFPIdx)
{
   fColorIdx.RegisterCopy(GetCanvas());
   fIntIdx.RegisterCopy(GetCanvas());
   fFPIdx.RegisterCopy(GetCanvas());
}
