// Author: Sergey Linev, 27.02.2020

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RGeoPainter.hxx>

#include "TGeoVolume.h"
#include "TGeoManager.h"
#include "TVirtualPad.h"
#include "TGeoTrack.h"

using namespace ROOT;

RGeoPainter::RGeoPainter(TGeoManager *manager) : TVirtualGeoPainter(manager)
{
   TVirtualGeoPainter::SetPainter(this);
   fGeoManager = manager;
}

RGeoPainter::~RGeoPainter() {}

TVirtualGeoTrack *RGeoPainter::AddTrack(Int_t id, Int_t pdgcode, TObject *particle)
{
   return (TVirtualGeoTrack *)(new TGeoTrack(id, pdgcode, nullptr, particle));
}

void RGeoPainter::AddTrackPoint(Double_t *point, Double_t *box, Bool_t reset)
{
   static Int_t npoints = 0;
   static Double_t xmin[3] = {0, 0, 0};
   static Double_t xmax[3] = {0, 0, 0};
   Int_t i;
   if (reset) {
      memset(box, 0, 6 * sizeof(Double_t));
      memset(xmin, 0, 3 * sizeof(Double_t));
      memset(xmax, 0, 3 * sizeof(Double_t));
      npoints = 0;
      return;
   }
   if (npoints == 0) {
      for (i = 0; i < 3; i++)
         xmin[i] = xmax[i] = 0;
      npoints++;
   }
   npoints++;
   Double_t ninv = 1. / Double_t(npoints);
   for (i = 0; i < 3; i++) {
      box[i] += ninv * (point[i] - box[i]);
      if (point[i] < xmin[i])
         xmin[i] = point[i];
      if (point[i] > xmax[i])
         xmax[i] = point[i];
      box[i + 3] = 0.5 * (xmax[i] - xmin[i]);
   }
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
   if (opt && strstr(opt, "s"))
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
