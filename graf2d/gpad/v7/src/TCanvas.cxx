/// \file TCanvas.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TCanvas.hxx"

#include "ROOT/TDrawable.hxx"
#include "ROOT/TLogger.hxx"

#include <memory>
#include <stdio.h>
#include <string.h>

#include "TROOT.h"

namespace {
static std::vector<std::shared_ptr<ROOT::Experimental::TCanvas>> &GetHeldCanvases()
{
   static std::vector<std::shared_ptr<ROOT::Experimental::TCanvas>> sCanvases;
   return sCanvases;
}
}

const std::vector<std::shared_ptr<ROOT::Experimental::TCanvas>> &ROOT::Experimental::TCanvas::GetCanvases()
{
   return GetHeldCanvases();
}

// void ROOT::Experimental::TCanvas::Paint() {
//  for (auto&& drw: fPrimitives) {
//    drw->Paint(*this);
//  }
// }

void ROOT::Experimental::TCanvas::Update()
{
   // SnapshotList_t lst;
   // for (auto&& drw: fPrimitives) {
   //   TSnapshot *snap = drw->CreateSnapshot(*this);
   //   lst.push_back(std::unique_ptr<TSnapshot>(snap));
   // }
}

std::shared_ptr<ROOT::Experimental::TCanvas> ROOT::Experimental::TCanvas::Create(const std::string &title)
{
   auto pCanvas = std::make_shared<TCanvas>();
   pCanvas->SetTitle(title);
   GetHeldCanvases().emplace_back(pCanvas);
   return pCanvas;
}

void ROOT::Experimental::TCanvas::Show()
{
   if (fPainter) return;
   bool batch_mode = gROOT->IsBatch();
   fPainter = Internal::TVirtualCanvasPainter::Create(*this, batch_mode);
}

void ROOT::Experimental::TCanvas::SaveAs(const std::string &filename)
{
   if (!fPainter) fPainter = Internal::TVirtualCanvasPainter::Create(*this, true);
   if (filename.find(".svg") != std::string::npos)
      fPainter->DoWhenReady("SVG", filename);
   else if (filename.find(".png") != std::string::npos)
      fPainter->DoWhenReady("PNG", filename);
}

// TODO: removal from GetHeldCanvases().
