// @(#)root/graf:$Name:  $:$Id: TLink.h,v 1.2 2000/06/13 11:05:33 brun Exp $
// Author: Rene Brun   05/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLink
#define ROOT_TLink


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLink                                                                //
//                                                                      //
// Hypertext link to an object.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TText
#include "TText.h"
#endif

class TLink : public TText {

protected:
   enum { kObjIsParent = BIT(1) };

   void   *fLink;           //pointer to object

public:
   TLink();
   TLink(Double_t x, Double_t y, void *pointer);
   virtual ~TLink();
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);

   ClassDef(TLink,0)  //Link: hypertext link to an object
};

#endif
