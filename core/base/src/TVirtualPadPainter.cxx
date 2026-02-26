// @(#)root/base:$Id$
// Author: Timur Pocheptsov   6/5/2009

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualPadPainter.h"
#include "TPluginManager.h"


/** \class TVirtualPadPainter
\ingroup Base

To make it possible to use GL for 2D graphic in a TPad/TCanvas.
TVirtualPadPainter interface must be used instead of TVirtualX.
Internally, non-GL implementation _should_ delegate all calls
to gVirtualX, GL implementation will delegate part of calls
to gVirtualX, and has to implement some of the calls from the scratch.
*/

////////////////////////////////////////////////////////////////////////////////
///Virtual dtor.

TVirtualPadPainter::~TVirtualPadPainter()
{
}

////////////////////////////////////////////////////////////////////////////////
///Empty definition.

void TVirtualPadPainter::InitPainter()
{
}

////////////////////////////////////////////////////////////////////////////////
///Empty definition.

void TVirtualPadPainter::InvalidateCS()
{
}

////////////////////////////////////////////////////////////////////////////////
///Empty definition.

void TVirtualPadPainter::LockPainter()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create a pad painter of specified type.

TVirtualPadPainter *TVirtualPadPainter::PadPainter(Option_t *type)
{
   TVirtualPadPainter *painter = nullptr;
   TPluginHandler *h = gPluginMgr->FindHandler("TVirtualPadPainter", type);

   if (h && h->LoadPlugin() != -1)
      painter = (TVirtualPadPainter *) h->ExecPlugin(0);

   return painter;
}


////////////////////////////////////////////////////////////////////////////////
/// Draw N segments on the pad
/// Exclude segments where both points match

void TVirtualPadPainter::DrawSegments(Int_t n, const Double_t *x, const Double_t *y)
{
   for(Int_t i = 0; i <= 2*n; i += 2)
      if ((x[i] != x[i+1]) || (y[i] != y[i + 1]))
         DrawLine(x[i], y[i], x[i+1], y[i+1]);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw N segments in NDC coordinates on the pad
/// Exclude segments where both points match

void TVirtualPadPainter::DrawSegmentsNDC(Int_t n, const Double_t *u, const Double_t *v)
{
   for(Int_t i = 0; i <= 2*n; i += 2)
      if ((u[i] != u[i+1]) || (v[i] != v[i + 1]))
         DrawLineNDC(u[i], v[i], u[i+1], v[i+1]);
}
