// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   04/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttTextCanvas
#define ROOT_TAttTextCanvas


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttTextCanvas                                                       //
//                                                                      //
// A specialized dialog canvas to set text attributes.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDialogCanvas
#include "TDialogCanvas.h"
#endif

class TAttTextCanvas : public TDialogCanvas {

public:
   TAttTextCanvas();
   TAttTextCanvas(const char *name, const char *title, UInt_t ww=400, UInt_t wh=600);
   virtual        ~TAttTextCanvas();
   virtual void   UpdateTextAttributes(Int_t align,Float_t angle,Int_t col,Int_t font,Float_t tsize);

   ClassDef(TAttTextCanvas,0)  //A specialized dialog canvas to set text attributes.
};

#endif

