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

TClass  *TVirtualGeoPainter::fgGeoPainter = 0;

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
      if (gROOT->LoadClass("TGeoPainter","GeoPainter")) return 0;
      TVirtualGeoPainter::SetPainter("TGeoPainter");
      if (!fgGeoPainter) return 0;
   }
   //create an instance of the painter
   TVirtualGeoPainter *p = (TVirtualGeoPainter*)fgGeoPainter->New();
   return p;
}

//______________________________________________________________________________
void TVirtualGeoPainter::SetPainter(const char *painter)
{
   // Static function to set an alternative histogram painter.

   fgGeoPainter = gROOT->GetClass(painter);
}
