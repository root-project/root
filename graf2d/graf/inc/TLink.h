// @(#)root/graf:$Id$
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

#include "TText.h"

class TLink : public TText {

protected:

   void   *fLink;           ///< pointer to object

public:
   enum EStatusBits { kIsStarStar = BIT(2) };
   TLink();
   TLink(Double_t x, Double_t y, void *pointer);
   virtual ~TLink();
   void     ExecuteEvent(Int_t event, Int_t px, Int_t py) override;

   ClassDefOverride(TLink,0)  //Link: hypertext link to an object
};

#endif
