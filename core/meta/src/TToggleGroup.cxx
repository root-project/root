// @(#)root/meta:$Id$
// Author: Piotr Golonka   31/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TToggleGroup                                                         //
//                                                                      //
// This class defines check-box facility for TToggle objects            //
// It is used in context menu "selectors" for picking up a value.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TMethod.h"
#include "TToggleGroup.h"

ClassImp(TToggleGroup)

//______________________________________________________________________________
TToggleGroup::TToggleGroup()
{
   // Constructor.

   fToggles  = new TOrdCollection();
}

//______________________________________________________________________________
TToggleGroup::TToggleGroup(const TToggleGroup& rhs) : TNamed(rhs),fToggles(0)
{
   // Copy constructor

   fToggles = (TOrdCollection*)rhs.fToggles->Clone();
}

//______________________________________________________________________________
TToggleGroup &TToggleGroup::operator=(const TToggleGroup &rhs)
{
   // Assignment operator.

   if (this != &rhs) {
      delete fToggles;
      fToggles = (TOrdCollection*)rhs.fToggles->Clone();
   }
   return *this;
}


//______________________________________________________________________________
TToggleGroup::~TToggleGroup()
{
   // Deletes togglegroup but does not disposes toggled objects!

   delete fToggles;
}

//______________________________________________________________________________
Int_t TToggleGroup::Add(TToggle *t, Bool_t select)
{
   // Add a new toggle.

   if (t) {
      fToggles->AddLast(t);
      if (select)
         Select(t);
      return IndexOf(t);
   } else
      return (-1);
}

//______________________________________________________________________________
Int_t TToggleGroup::InsertAt(TToggle *t, Int_t pos,Bool_t select)
{
   // Add a new toggle at a specific position.

   if (t) {
      fToggles->AddAt(t,pos);
      if (select)
         Select(t);
      return IndexOf(t);
   } else
      return (-1);
}

//______________________________________________________________________________
void TToggleGroup::Select(Int_t idx)
{
   // Select a toggle.

   TToggle *sel = At(idx);
   if (sel)
      Select(sel);
}

//______________________________________________________________________________
void TToggleGroup::Select(TToggle *t)
{
   // Selector a toggle.

   TIter next(fToggles);
   TToggle *i = 0;

   // Untoggle toggled , and toggle this one if it's present on a list!

   while ((i = (TToggle*)next()))
      if ( i->GetState() || (i==t) )
         i->Toggle();
}

//______________________________________________________________________________
void TToggleGroup::DeleteAll()
{
   // Disposes of all objects and clears array

   fToggles->Delete();
}
