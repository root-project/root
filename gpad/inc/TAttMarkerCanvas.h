// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   04/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttMarkerCanvas
#define ROOT_TAttMarkerCanvas


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttMarkerCanvas                                                     //
//                                                                      //
// A specialized dialog canvas to set marker attributes.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDialogCanvas
#include "TDialogCanvas.h"
#endif

class TAttMarkerCanvas : public TDialogCanvas {

public:
   TAttMarkerCanvas();
   TAttMarkerCanvas(const char *name, const char *title, UInt_t ww=400, UInt_t wh=600);
   virtual        ~TAttMarkerCanvas();
   virtual void   UpdateMarkerAttributes(Int_t col,Int_t sty,Float_t msiz);

   ClassDef(TAttMarkerCanvas,0)  //A specialized dialog canvas to set marker attributes.
};

#endif

