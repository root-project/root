// @(#)root/graf:$Id$
// Author: Rene Brun   08/08/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCrown
#define ROOT_TCrown


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCrown                                                               //
//                                                                      //
// A crown is defined by an inner and outer radius                      //
// It can also be a truncated crown with [phimin,phimax]                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TEllipse
#include "TEllipse.h"
#endif


class TCrown : public TEllipse {

public:
   TCrown();
   TCrown(Double_t x1, Double_t y1,Double_t radin, Double_t radout,
          Double_t phimin=0,Double_t phimax=360);
   TCrown(const TCrown &crown);
   virtual ~TCrown();
   void   Copy(TObject &crown) const;
   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
   virtual void   DrawCrown(Double_t x1, Double_t y1, Double_t radin, Double_t radout,
                            Double_t  phimin=0, Double_t  phimax=360, Option_t *option="");
   virtual void   ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void   Paint(Option_t *option="");
   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TCrown,1)  //A crown or segment of crown
};

#endif
