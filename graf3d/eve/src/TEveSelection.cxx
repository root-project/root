// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveSelection.h"
#include "TEveProjectionBases.h"
#include "TEveCompound.h"
#include "TEveManager.h"

#include "TClass.h"

/** \class TEveSelection
\ingroup TEve
Make sure there is a SINGLE running TEveSelection for each
selection type (select/highlight).
*/

ClassImp(TEveSelection);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveSelection::TEveSelection(const char* n, const char* t) :
   TEveElementList(n, t),
   fPickToSelect  (kPS_Projectable),
   fActive        (kTRUE),
   fIsMaster      (kTRUE)
{
   fSelElement       = &TEveElement::SelectElement;
   fIncImpSelElement = &TEveElement::IncImpliedSelected;
   fDecImpSelElement = &TEveElement::DecImpliedSelected;
}

////////////////////////////////////////////////////////////////////////////////
/// Set to 'highlight' mode.

void TEveSelection::SetHighlightMode()
{
   // Most importantly, this sets the pointers-to-function-members in
   // TEveElement that are used to mark elements as (un)selected and
   // implied-(un)selected.

   fPickToSelect = kPS_Projectable;
   fIsMaster     = kFALSE;

   fSelElement       = &TEveElement::HighlightElement;
   fIncImpSelElement = &TEveElement::IncImpliedHighlighted;
   fDecImpSelElement = &TEveElement::DecImpliedHighlighted;
}

////////////////////////////////////////////////////////////////////////////////
/// Select element indicated by the entry and fill its
/// implied-selected set.

void TEveSelection::DoElementSelect(TEveSelection::SelMap_i entry)
{
   TEveElement *el  = entry->first;
   Set_t       &set = entry->second;

   (el->*fSelElement)(kTRUE);
   el->FillImpliedSelectedSet(set);
   for (Set_i i = set.begin(); i != set.end(); ++i)
      ((*i)->*fIncImpSelElement)();
}

////////////////////////////////////////////////////////////////////////////////
/// Deselect element indicated by the entry and clear its
/// implied-selected set.

void TEveSelection::DoElementUnselect(TEveSelection::SelMap_i entry)
{
   TEveElement *el  = entry->first;
   Set_t       &set = entry->second;

   for (Set_i i = set.begin(); i != set.end(); ++i)
      ((*i)->*fDecImpSelElement)();
   set.clear();
   (el->*fSelElement)(kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Pre-addition check. Deny addition if el is already selected.
/// Virtual from TEveElement.

Bool_t TEveSelection::AcceptElement(TEveElement* el)
{
   return el != this && fImpliedSelected.find(el) == fImpliedSelected.end() &&
          el->IsA()->InheritsFrom(TEveSelection::Class()) == kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Add an element into selection, virtual from TEveElement.

void TEveSelection::AddElement(TEveElement* el)
{
   TEveElementList::AddElement(el);

   SelMap_i i = fImpliedSelected.insert(std::make_pair(el, Set_t())).first;
   if (fActive)
   {
      DoElementSelect(i);
   }
   SelectionAdded(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Add an element into selection, virtual from TEveElement.
/// Overriden here just so that a signal can be emitted.

void TEveSelection::RemoveElement(TEveElement* el)
{
   TEveElementList::RemoveElement(el);
   SelectionRemoved(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveElement.

void TEveSelection::RemoveElementLocal(TEveElement* el)
{
   SelMap_i i = fImpliedSelected.find(el);

   if (i != fImpliedSelected.end())
   {
      if (fActive)
      {
         DoElementUnselect(i);
      }
      fImpliedSelected.erase(i);
   }
   else
   {
      Warning("TEveSelection::RemoveElementLocal", "element not found in map.");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add an element into selection, virtual from TEveElement.
/// Overriden here just so that a signal can be emitted.

void TEveSelection::RemoveElements()
{
   TEveElementList::RemoveElements();
   SelectionCleared();
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveElement.

void TEveSelection::RemoveElementsLocal()
{
   if (fActive)
   {
      for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
         DoElementUnselect(i);
   }
   fImpliedSelected.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove element from all implied-selected sets.
///
/// This is called as part of the element destruction from
/// TEveManager::PreDeleteElement() and should not be called
/// directly.

void TEveSelection::RemoveImpliedSelected(TEveElement* el)
{
   for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
   {
      Set_i j = i->second.find(el);
      if (j != i->second.end())
         i->second.erase(j);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Recalculate implied-selected state for given selection entry.
/// Add new elements to implied-selected set and increase their
/// implied-selected count.

void TEveSelection::RecheckImpliedSet(SelMap_i smi)
{
   Set_t set;
   smi->first->FillImpliedSelectedSet(set);
   for (Set_i i = set.begin(); i != set.end(); ++i)
   {
      if (smi->second.find(*i) == smi->second.end())
      {
         smi->second.insert(*i);
         ((*i)->*fIncImpSelElement)();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// If given element is selected or implied-selected with this
/// selection and recheck implied-set for given selection entry.

void TEveSelection::RecheckImpliedSetForElement(TEveElement* el)
{
   // Top-level selected.
   {
      SelMap_i i = fImpliedSelected.find(el);
      if (i != fImpliedSelected.end())
         RecheckImpliedSet(i);
   }

   // Implied selected, need to loop over all.
   {
      for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++ i)
      {
         if (i->second.find(el) != i->second.end())
            RecheckImpliedSet(i);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Emit SelectionAdded signal.

void TEveSelection::SelectionAdded(TEveElement* el)
{
   Emit("SelectionAdded(TEveElement*)", (Longptr_t)el);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit SelectionRemoved signal.

void TEveSelection::SelectionRemoved(TEveElement* el)
{
   Emit("SelectionRemoved(TEveElement*)", (Longptr_t)el);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit SelectionCleared signal.

void TEveSelection::SelectionCleared()
{
   Emit("SelectionCleared()");
}

////////////////////////////////////////////////////////////////////////////////
/// Called when secondary selection changed internally.

void TEveSelection::SelectionRepeated(TEveElement* el)
{
   Emit("SelectionRepeated(TEveElement*)", (Longptr_t)el);
}

////////////////////////////////////////////////////////////////////////////////
/// Activate this selection.

void TEveSelection::ActivateSelection()
{
   for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
      DoElementSelect(i);
   fActive = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Deactivate this selection.

void TEveSelection::DeactivateSelection()
{
   fActive = kFALSE;
   for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
      DoElementUnselect(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Given element el that was picked or clicked by the user, find
/// the parent/ancestor element that should actually become the main
/// selected element according to current selection mode.

TEveElement* TEveSelection::MapPickedToSelected(TEveElement* el)
{
   if (el == 0)
      return 0;

   if (el->ForwardSelection())
   {
      return el->ForwardSelection();
   }

   switch (fPickToSelect)
   {
      case kPS_Ignore:
      {
         return 0;
      }
      case kPS_Element:
      {
         return el;
      }
      case kPS_Projectable:
      {
         TEveProjected* pted = dynamic_cast<TEveProjected*>(el);
         if (pted)
            return dynamic_cast<TEveElement*>(pted->GetProjectable());
         return el;
      }
      case kPS_Compound:
      {
         TEveElement* cmpnd = el->GetCompound();
         if (cmpnd)
            return cmpnd;
         return el;
      }
      case kPS_PableCompound:
      {
         TEveProjected* pted = dynamic_cast<TEveProjected*>(el);
         if (pted)
            el = dynamic_cast<TEveElement*>(pted->GetProjectable());
         TEveElement* cmpnd = el->GetCompound();
         if (cmpnd)
            return cmpnd;
         return el;
      }
      case kPS_Master:
      {
         TEveElement* mstr = el->GetMaster();
         if (mstr)
            return mstr;
         return el;
      }
   }
   return el;
}

////////////////////////////////////////////////////////////////////////////////
/// Called when user picks/clicks on an element. If multi is true,
/// the user is requiring a multiple selection (usually this is
/// associated with control-key being pressed at the time of pick
/// event).

void TEveSelection::UserPickedElement(TEveElement* el, Bool_t multi)
{
   TEveElement *edit_el = el ? el->ForwardEdit() : 0;

   el = MapPickedToSelected(el);

   if (el || HasChildren())
   {
      if (!multi)
         RemoveElements();
      if (el)
      {
         if (HasChild(el))
             RemoveElement(el);
         else
            AddElement(el);
      }
      if (fIsMaster)
         gEve->ElementSelect(edit_el ? edit_el : el);
      gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Called when secondary selection becomes empty.

void TEveSelection::UserRePickedElement(TEveElement* el)
{
   el = MapPickedToSelected(el);
   if (el && HasChild(el))
   {
      SelectionRepeated(el);
      gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Called when secondary selection becomes empty.

void TEveSelection::UserUnPickedElement(TEveElement* el)
{
   el = MapPickedToSelected(el);
   if (el)
   {
      RemoveElement(el);
      gEve->Redraw3D();
   }
}
