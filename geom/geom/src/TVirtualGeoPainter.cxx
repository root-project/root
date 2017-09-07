// @(#)root/geom:$Id$
// Author: Andrei Gheata   11/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TVirtualGeoPainter
\ingroup Geometry_classes

Abstract class for geometry painters
*/

#include "TROOT.h"
#include "TVirtualGeoPainter.h"
#include "TPluginManager.h"
#include "TGeoManager.h"

TVirtualGeoPainter  *TVirtualGeoPainter::fgGeoPainter = 0;

ClassImp(TVirtualGeoPainter);

////////////////////////////////////////////////////////////////////////////////
/// Geometry painter default constructor

TVirtualGeoPainter::TVirtualGeoPainter(TGeoManager *)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Geometry painter default destructor

TVirtualGeoPainter::~TVirtualGeoPainter()
{
   fgGeoPainter = 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Static function returning a pointer to the geometry painter.
/// The painter will paint objects from the specified geometry.
/// If the geometry painter does not exist a default painter is created.

TVirtualGeoPainter *TVirtualGeoPainter::GeoPainter()
{
   // if no painter set yet, create a default painter via the PluginManager
   if (!fgGeoPainter) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualGeoPainter"))) {
         if (h->LoadPlugin() == -1)
            return 0;
         fgGeoPainter = (TVirtualGeoPainter*)h->ExecPlugin(1,gGeoManager);
      }
   }
   return fgGeoPainter;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function to set an alternative histogram painter.

void TVirtualGeoPainter::SetPainter(const TVirtualGeoPainter *painter)
{
   fgGeoPainter = (TVirtualGeoPainter*)painter;
}
