// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveCompound.h"

//==============================================================================
//==============================================================================
// TEveCompound
//==============================================================================

//______________________________________________________________________________
//
// Description of TEveCompound
//

ClassImp(TEveCompound);

//______________________________________________________________________________
TEveCompound::TEveCompound(const char* n, const char* t, Bool_t doColor) :
   TEveElementList (n, t, doColor),
   TEveProjectable (),
   fCompoundOpen   (0)
{
   // Constructor.
}

//______________________________________________________________________________
void TEveCompound::SetMainColor(Color_t color)
{
   // SetMainColor for the compound.
   // The color is also propagated to children (compouind elements)
   // whoose current color is the same as the old color.

   Color_t old_color = GetMainColor();

   TEveElement::SetMainColor(color);

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if ((*i)->GetCompound() == this && (*i)->GetMainColor() == old_color)
         (*i)->SetMainColor(color);
   }
}

//******************************************************************************

//______________________________________________________________________________
void TEveCompound::AddElement(TEveElement* el)
{
   // Call base-class implementation.
   // If compund is open and compound of the new element is not set,
   // the el's compound is set to this.

   TEveElementList::AddElement(el);
   if (IsCompoundOpen() && el->GetCompound() == 0)
      el->SetCompound(this);
}

//______________________________________________________________________________
void TEveCompound::RemoveElementLocal(TEveElement* el)
{
   // Decompoundofy el, call base-class version.

   if (el->GetCompound() == this)
      el->SetCompound(0);

   TEveElementList::RemoveElementLocal(el);
}

//______________________________________________________________________________
void TEveCompound::RemoveElementsLocal()
{
   // Decompoundofy children, call base-class version.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if ((*i)->GetCompound() == this)
         (*i)->SetCompound(0);
   }

   TEveElementList::RemoveElementsLocal();
}

//******************************************************************************

//______________________________________________________________________________
void TEveCompound::FillImpliedSelectedSet(Set_t& impSelSet)
{
   // Recurse on all children that are in this compund and
   // call the base-class version.
   //
   // Note that projected replicas of the compound will be added to
   // the set in base-class function that handles projectables.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if ((*i)->GetCompound() == this)
      {
         impSelSet.insert(*i);
         (*i)->FillImpliedSelectedSet(impSelSet);
      }

   }
   TEveElementList::FillImpliedSelectedSet(impSelSet);
}

//******************************************************************************

//______________________________________________________________________________
TClass* TEveCompound::ProjectedClass() const
{
   // Virtual from TEveProjectable, returns TEveCompoundProjected class.

   return TEveCompoundProjected::Class();
}


//==============================================================================
//==============================================================================
// TEveCompoundProjected
//==============================================================================

//______________________________________________________________________________
//
// Description of TEveCompoundProjected
//

ClassImp(TEveCompoundProjected);

//______________________________________________________________________________
TEveCompoundProjected::TEveCompoundProjected() :
   TEveCompound  (),
   TEveProjected ()
{
   // Constructor.
}

//______________________________________________________________________________
void TEveCompoundProjected::SetMainColor(Color_t color)
{
   // Revert back to the behaviour of TEveElement as color
   // is propagated:
   // a) from projectable -> projected
   // b) from compound -> compound elements
   // and we do not need to do this twice for projected-compound-elements.

   TEveElement::SetMainColor(color);
}
