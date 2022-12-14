// @(#)root/eve7:$Id$
// Author: Sergey Linev, 27.02.2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RGeoPainter.hxx>

#include "TGeoVolume.h"
#include "TGeoManager.h"
#include "TVirtualPad.h"

using namespace ROOT::Experimental;

RGeoPainter::RGeoPainter(TGeoManager *manager) : TVirtualGeoPainter(manager)
{
   TVirtualGeoPainter::SetPainter(this);
   fGeoManager = manager;
}

RGeoPainter::~RGeoPainter()
{
}

void RGeoPainter::SetTopVisible(Bool_t on)
{
   fTopVisible = on ? 1 : 0;
}


void RGeoPainter::SetGeoManager(TGeoManager *mgr)
{
   if (fViewer && (fGeoManager != mgr))
      fViewer->SetGeometry(fGeoManager);

   fGeoManager = mgr;
}

void RGeoPainter::DrawVolume(TGeoVolume *vol, Option_t *opt)
{
   if (gPad) {
      auto g = vol->GetGeoManager();

      // append volume or geomanager itself to the pad, web canvas also support geometry drawing now
      if (g && (g->GetTopVolume() == vol))
         g->AppendPad(opt);
      else
         vol->AppendPad(opt);

      return;
   }


   if (!fViewer)
      fViewer = std::make_shared<RGeomViewer>(fGeoManager);

   // select volume to draw
   fViewer->SetGeometry(fGeoManager, vol->GetName());

   std::string drawopt = "";
   if (opt && strstr(opt,"s"))
      drawopt = "wire";

   // specify JSROOT draw options - here clipping on X,Y,Z axes
   fViewer->SetDrawOptions(drawopt);

   if (fTopVisible >= 0)
      fViewer->SetTopVisible(fTopVisible > 0);


   // set default limits for number of visible nodes and faces
   // when viewer created, initial values exported from TGeoManager
   // viewer->SetLimits();

   // start browser
   fViewer->Show();
}
