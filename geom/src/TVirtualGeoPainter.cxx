// @(#)root/geom:$Name:$:$Id:$
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
}


//______________________________________________________________________________
TVirtualGeoPainter *TVirtualGeoPainter::GeoPainter()
{
   // Static function returning a pointer to the geometry painter.
   // The painter will paint objects from the specified geometry. 
   // If the geometry painter does not exist a default painter is created.

   // if no painter set yet, set TGeoPainter by default
   if (!fgGeoPainter) {
      if (gROOT->LoadClass("TGeoPainter","GeomPainter")) return 0;
      gROOT->ProcessLineFast("new TGeoPainter();");
   }
   return fgGeoPainter;
}

//______________________________________________________________________________
void TVirtualGeoPainter::SetPainter(TVirtualGeoPainter *painter)
{
   // Static function to set an alternative histogram painter.

   fgGeoPainter = painter;
}
