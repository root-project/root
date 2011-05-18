// @(#)root/hist:$Id$
// Author: Olivier Couet 20/05/08

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TVirtualGraphPainter.h"
#include "TPluginManager.h"
#include "TClass.h"

TVirtualGraphPainter *TVirtualGraphPainter::fgPainter = 0;

ClassImp(TVirtualGraphPainter)

//______________________________________________________________________________
//
//  TVirtualGraphPainter is an abstract interface to a histogram painter.
//


//______________________________________________________________________________
TVirtualGraphPainter *TVirtualGraphPainter::GetPainter()
{
   // Static function returning a pointer to the current graph painter.
   // If the graph painter does not exist a default painter (singleton) is created.

   // if no painter set yet, create a default painter via the PluginManager
   if (!fgPainter) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualGraphPainter"))) {
         if (h->LoadPlugin() == -1) return 0;
         fgPainter = (TVirtualGraphPainter*)gROOT->GetClass("TGraphPainter")->New();
      }
   }

   // Create an instance of the graph painter
   return fgPainter;
}

//______________________________________________________________________________
void TVirtualGraphPainter::SetPainter(TVirtualGraphPainter *painter)
{
   // Static function to set an alternative histogram painter.

   fgPainter = painter;
}
