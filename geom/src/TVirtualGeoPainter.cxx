// @(#)root/geom:$Name:  $:$Id: TVirtualGeoPainter.cxx,v 1.3 2002/07/10 19:24:16 brun Exp $
// Author: Andrei Gheata   11/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TVirtualGeoPainter.h"
#include "TPluginManager.h"

TVirtualGeoPainter  *TVirtualGeoPainter::fgGeoPainter = 0;

ClassImp(TVirtualGeoPainter)

//______________________________________________________________________________
TVirtualGeoPainter::TVirtualGeoPainter()
{
//*-*-*-*-*-*-*-*-*-*-*Geometry painter default constructor*-*-*-*-*-*-*-*-*
//*-*                  ====================================
}

//______________________________________________________________________________
TVirtualGeoPainter::~TVirtualGeoPainter()
{
//*-*-*-*-*-*-*-*-*-*-*Geometry painter default destructor*-*-*-*-*-*-*-*-*
//*-*                  ===================================

   fgGeoPainter = 0;
}


//______________________________________________________________________________
TVirtualGeoPainter *TVirtualGeoPainter::GeoPainter()
{
   // Static function returning a pointer to the geometry painter.
   // The painter will paint objects from the specified geometry. 
   // If the geometry painter does not exist a default painter is created.

   // if no painter set yet, create a default painter via the PluginManager
   if (!fgGeoPainter) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualGeoPainter"))) {
         if (h->LoadPlugin() == -1)
            return 0;
         fgGeoPainter = (TVirtualGeoPainter*)h->ExecPlugin(0);
      }
   }
   return fgGeoPainter;
}

//______________________________________________________________________________
void TVirtualGeoPainter::SetPainter(const TVirtualGeoPainter *painter)
{
   // Static function to set an alternative histogram painter.

   fgGeoPainter = (TVirtualGeoPainter*)painter;
}
