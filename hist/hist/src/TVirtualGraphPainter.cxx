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

TClass *TVirtualGraphPainter::fgPainter = 0;


ClassImp(TVirtualGraphPainter)

//______________________________________________________________________________
//
//  TVirtualGraphPainter is an abstract interface to a histogram painter.
//


//______________________________________________________________________________
TVirtualGraphPainter *TVirtualGraphPainter::GraphPainter(TGraph *obj)
{
   // Static function returning a pointer to the current graph painter.
   // The painter will paint the specified obj. If the histogram painter
   // does not exist a default painter is created.

   // if no painter set yet, create a default painter via the PluginManager
   if (!fgPainter) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualGraphPainter"))) {
         if (h->LoadPlugin() == -1)
            return 0;
         TVirtualGraphPainter::SetPainter(h->GetClass());
         if (!fgPainter) return 0;
      }
   }

   //create an instance of the graph painter
   TVirtualGraphPainter *p = (TVirtualGraphPainter*)fgPainter->New();
   if (p) p->SetGraph(obj);
   return p;
}

//______________________________________________________________________________
void TVirtualGraphPainter::SetPainter(const char *painter)
{
   // Static function to set an alternative histogram painter.

   fgPainter = TClass::GetClass(painter);
}
