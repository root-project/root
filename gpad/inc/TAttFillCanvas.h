// @(#)root/gpad:$Name:  $:$Id: TAttFillCanvas.h,v 1.1.1.1 2000/05/16 17:00:41 rdm Exp $
// Author: Rene Brun   04/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttFillCanvas
#define ROOT_TAttFillCanvas


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttFillCanvas                                                       //
//                                                                      //
// A specialized dialog canvas to set fill attributes.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDialogCanvas
#include "TDialogCanvas.h"
#endif

class TAttFillCanvas : public TDialogCanvas {

public:
   TAttFillCanvas();
   TAttFillCanvas(const char *name, const char *title, Int_t ww=400, Int_t wh=600);
   virtual        ~TAttFillCanvas();
   virtual void   UpdateFillAttributes(Int_t col, Int_t sty);

   ClassDef(TAttFillCanvas,0)  //A specialized dialog canvas to set fill attributes.
};

#endif

