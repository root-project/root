// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   03/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttLineCanvas
#define ROOT_TAttLineCanvas


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttLineCanvas                                                       //
//                                                                      //
// A specialized dialog canvas to set line attributes.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDialogCanvas
#include "TDialogCanvas.h"
#endif

class TAttLineCanvas : public TDialogCanvas {

public:
   TAttLineCanvas();
   TAttLineCanvas(const char *name, const char *title, UInt_t ww=400, UInt_t wh=600);
   virtual        ~TAttLineCanvas();
   virtual void   UpdateLineAttributes(Int_t col, Int_t sty, Int_t width);

   ClassDef(TAttLineCanvas,0)  //A specialized dialog canvas to set line attributes.
};

#endif

