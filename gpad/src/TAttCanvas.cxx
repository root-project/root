// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Strlen.h"
#include "TAttCanvas.h"

ClassImp(TAttCanvas)

//______________________________________________________________________________
//
//     Manages canvas attributes. Referenced by TStyle.
//

//______________________________________________________________________________
TAttCanvas::TAttCanvas()
{
   ResetAttCanvas();
}

//______________________________________________________________________________
TAttCanvas::~TAttCanvas()
{

}

//______________________________________________________________________________
void TAttCanvas::Copy(TAttCanvas &attcanvas)
{
   attcanvas.fXBetween     = fXBetween;
   attcanvas.fYBetween     = fYBetween;
   attcanvas.fTitleFromTop = fTitleFromTop;
   attcanvas.fXdate        = fXdate;
   attcanvas.fYdate        = fYdate;
   attcanvas.fAdate        = fAdate;
}

//______________________________________________________________________________
void TAttCanvas::Print(Option_t *)
{

}

//______________________________________________________________________________
void TAttCanvas::ResetAttCanvas(Option_t *)
{
   fXBetween     = 2;
   fYBetween     = 2;
   fTitleFromTop = 1.2;
   fXdate        = 0.2;
   fYdate        = 0.3;
   fAdate        = 1;
}
