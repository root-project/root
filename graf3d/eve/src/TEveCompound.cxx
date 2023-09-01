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

/** \class TEveCompound
\ingroup TEve
Description of TEveCompound
*/

ClassImp(TEveCompound);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveCompound::TEveCompound(const char* n, const char* t, Bool_t doColor, Bool_t doTransparency) :
   TEveElementList (n, t, doColor, doTransparency),
   fCompoundOpen   (0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// SetMainColor for the compound.
/// The color is also propagated to children with compound set to this
/// whose current color is the same as the old color.
///
/// The following CompoundSelectionColorBits have further influence:
///  - kCSCBApplyMainColorToAllChildren      - apply color to all children;
///  - kCSCBApplyMainColorToMatchingChildren - apply color to children who have
///                                            matching old color.

void TEveCompound::SetMainColor(Color_t color)
{
   Color_t old_color = GetMainColor();

   TEveElement::SetMainColor(color);

   Bool_t color_all      = TestCSCBits(kCSCBApplyMainColorToAllChildren);
   Bool_t color_matching = TestCSCBits(kCSCBApplyMainColorToMatchingChildren);

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (color_all || (color_matching && (*i)->GetMainColor() == old_color) ||
          ((*i)->GetCompound() == this && (*i)->GetMainColor() == old_color))
      {
         (*i)->SetMainColor(color);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// SetMainTransparency for the compound.
/// The transparency is also propagated to children with compound set to this
/// whose current transparency is the same as the old transparency.
///
/// The following CompoundSelectionColorBits have further influence:
///  - kCSCBApplyMainTransparencyToAllChildren      - apply transparency to all children;
///  - kCSCBApplyMainTransparencyToMatchingChildren - apply transparency to children who have
///                                                   matching transparency.

void TEveCompound::SetMainTransparency(Char_t t)
{
   Char_t old_t = GetMainTransparency();

   TEveElement::SetMainTransparency(t);

   Bool_t chg_all      = TestCSCBits(kCSCBApplyMainTransparencyToAllChildren);
   Bool_t chg_matching = TestCSCBits(kCSCBApplyMainTransparencyToMatchingChildren);

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (chg_all || (chg_matching && (*i)->GetMainTransparency() == old_t) ||
          ((*i)->GetCompound() == this && (*i)->GetMainTransparency() == old_t))
      {
         (*i)->SetMainTransparency(t);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Call base-class implementation.
/// If compound is open and compound of the new element is not set,
/// the el's compound is set to this.
/// You might also want to call RecheckImpliedSelections().

void TEveCompound::AddElement(TEveElement* el)
{
   TEveElementList::AddElement(el);
   if (IsCompoundOpen() && el->GetCompound() == 0)
      el->SetCompound(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Decompoundofy el, call base-class version.

void TEveCompound::RemoveElementLocal(TEveElement* el)
{
   if (el->GetCompound() == this)
      el->SetCompound(0);

   TEveElementList::RemoveElementLocal(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Decompoundofy children, call base-class version.

void TEveCompound::RemoveElementsLocal()
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if ((*i)->GetCompound() == this)
         (*i)->SetCompound(0);
   }

   TEveElementList::RemoveElementsLocal();
}

////////////////////////////////////////////////////////////////////////////////
/// Recurse on all children that are in this compound and
/// call the base-class version.
/// If SelectionColorBit kSCBImplySelectAllChildren is set, then all
/// children are added to the set.
///
/// Note that projected replicas of the compound will be added to
/// the set in base-class function that handles projectable.

void TEveCompound::FillImpliedSelectedSet(Set_t& impSelSet)
{
   Bool_t select_all = TestCSCBits(kCSCBImplySelectAllChildren);

   for (List_i i = fChildren.begin(); i != fChildren.end(); ++i)
   {
      if (select_all || (*i)->GetCompound() == this)
      {
         if (impSelSet.insert(*i).second)
            (*i)->FillImpliedSelectedSet(impSelSet);
      }
   }

   TEveElementList::FillImpliedSelectedSet(impSelSet);
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveProjectable, returns TEveCompoundProjected class.

TClass* TEveCompound::ProjectedClass(const TEveProjection*) const
{
   return TEveCompoundProjected::Class();
}


/** \class TEveCompoundProjected
\ingroup TEve
Description of TEveCompoundProjected
*/

ClassImp(TEveCompoundProjected);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveCompoundProjected::TEveCompoundProjected() :
   TEveCompound  (),
   TEveProjected ()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Revert back to the behaviour of TEveElement as color
/// is propagated:
///  1. from projectable -> projected
///  2. from compound -> compound elements
/// and we do not need to do this twice for projected-compound-elements.

void TEveCompoundProjected::SetMainColor(Color_t color)
{
   TEveElement::SetMainColor(color);
}
