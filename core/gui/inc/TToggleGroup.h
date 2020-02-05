// @(#)root/meta:$Id$
// Author: Piotr Golonka   31/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TToggleGroup
#define ROOT_TToggleGroup


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TToggleGroup                                                         //
//                                                                      //
// This class defines check-box facility for TToggle objects            //
// It is used in context menu "selectors" for picking up a value.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"
#include "TToggle.h"
#include "TOrdCollection.h"


class TToggleGroup : public TNamed {

private:
   TOrdCollection *fToggles;        // list of TToggle objects

public:
   TToggleGroup();
   TToggleGroup(const TToggleGroup&);
   TToggleGroup &operator=(const TToggleGroup&);
   virtual ~TToggleGroup();
   virtual Int_t       GetTogglesCount() {return fToggles->GetSize();};
   virtual TToggle    *At(Int_t idx) {return (TToggle*)fToggles->At(idx);};

   virtual void        Remove(TToggle *t) {fToggles->Remove(t);};
   virtual void        Remove(Int_t pos) {fToggles->RemoveAt(pos);};

   virtual void        DeleteAll();
   virtual TToggle    *First() {return (TToggle*)fToggles->First();};
   virtual TToggle    *Last()  {return (TToggle*)fToggles->Last();};

   virtual Int_t       IndexOf(TToggle *t) {return fToggles->IndexOf(t);};

   virtual Int_t       Add(TToggle *t, Bool_t select=1);
   virtual Int_t       InsertAt(TToggle *t, Int_t pos,Bool_t select=1);
   virtual void        Select(Int_t idx);
   virtual void        Select(TToggle *t);

   ClassDef(TToggleGroup,0)  // Group of contex-menu toggle objects
};

#endif

