// @(#)root/meta:$Id$
// Author: Piotr Golonka   31/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TToggleGroup

This class defines check-box facility for TToggle objects
It is used in context menu "selectors" for picking up a value.
*/


#include "TMethod.h"
#include "TToggleGroup.h"

ClassImp(TToggleGroup);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TToggleGroup::TToggleGroup()
{
   fToggles  = new TOrdCollection();
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TToggleGroup::TToggleGroup(const TToggleGroup& rhs) : TNamed(rhs),fToggles(0)
{
   fToggles = (TOrdCollection*)rhs.fToggles->Clone();
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

TToggleGroup &TToggleGroup::operator=(const TToggleGroup &rhs)
{
   if (this != &rhs) {
      delete fToggles;
      fToggles = (TOrdCollection*)rhs.fToggles->Clone();
   }
   return *this;
}


////////////////////////////////////////////////////////////////////////////////
/// Deletes togglegroup but does not disposes toggled objects!

TToggleGroup::~TToggleGroup()
{
   delete fToggles;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a new toggle.

Int_t TToggleGroup::Add(TToggle *t, Bool_t select)
{
   if (t) {
      fToggles->AddLast(t);
      if (select)
         Select(t);
      return IndexOf(t);
   } else
      return (-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a new toggle at a specific position.

Int_t TToggleGroup::InsertAt(TToggle *t, Int_t pos,Bool_t select)
{
   if (t) {
      fToggles->AddAt(t,pos);
      if (select)
         Select(t);
      return IndexOf(t);
   } else
      return (-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Select a toggle.

void TToggleGroup::Select(Int_t idx)
{
   TToggle *sel = At(idx);
   if (sel)
      Select(sel);
}

////////////////////////////////////////////////////////////////////////////////
/// Selector a toggle.

void TToggleGroup::Select(TToggle *t)
{
   TIter next(fToggles);
   TToggle *i = 0;

   // Untoggle toggled , and toggle this one if it's present on a list!

   while ((i = (TToggle*)next()))
      if ( i->GetState() || (i==t) )
         i->Toggle();
}

////////////////////////////////////////////////////////////////////////////////
/// Disposes of all objects and clears array

void TToggleGroup::DeleteAll()
{
   fToggles->Delete();
}
