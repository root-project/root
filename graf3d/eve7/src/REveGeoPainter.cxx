// @(#)root/eve7:$Id$
// Author: Sergey Linev, 27.02.2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveGeoPainter.hxx>

#include "TGeoVolume.h"

using namespace ROOT::Experimental;

REveGeoPainter::REveGeoPainter(TGeoManager *manager) : TVirtualGeoPainter(manager)
{
   fGeoManager = manager;
}

REveGeoPainter::~REveGeoPainter()
{
}

void REveGeoPainter::SetGeoManager(TGeoManager *mgr)
{
   if (fViewer && (fGeoManager!=mgr))
      fViewer->SetGeometry(fGeoManager);

   fGeoManager = mgr;
}

void REveGeoPainter::DrawVolume(TGeoVolume *vol, Option_t *)
{
   if (!fViewer)
      fViewer = std::make_shared<REveGeomViewer>(fGeoManager);

   // select volume to draw
   fViewer->SetGeometry(fGeoManager, vol->GetName());

   // specify JSROOT draw options - here clipping on X,Y,Z axes
   // viewer->SetDrawOptions("clipxyz");

   // set default limits for number of visible nodes and faces
   // when viewer created, initial values exported from TGeoManager
   // viewer->SetLimits();

   // start browser
   fViewer->Show();
}
