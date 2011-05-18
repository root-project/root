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

//______________________________________________________________________________
//
// Make sure there is a SINGLE running TEveSelection for each
// selection type (select/highlight).

ClassImp(TEveSelection);

//______________________________________________________________________________
TEveSelection::TEveSelection(const char* n, const char* t) :
   TEveElementList(n, t),
   fPickToSelect  (kPS_Projectable),
   fActive        (kTRUE),
   fIsMaster      (kTRUE)
{
   // Constructor.

   fSelElement       = &TEveElement::SelectElement;
   fIncImpSelElement = &TEveElement::IncImpliedSelected;
   fDecImpSelElement = &TEveElement::DecImpliedSelected;
}

//______________________________________________________________________________
void TEveSelection::SetHighlightMode()
{
   // Set to 'highlight' mode.

   // Most importantly, this sets the pointers-to-function-members in
   // TEveElement that are used to mark elements as (un)selected and
   // implied-(un)selected.

   fPickToSelect = kPS_Projectable;
   fIsMaster     = kFALSE;

   fSelElement       = &TEveElement::HighlightElement;
   fIncImpSelElement = &TEveElement::IncImpliedHighlighted;
   fDecImpSelElement = &TEveElement::DecImpliedHighlighted;
}


/******************************************************************************/
// Protected helpers
/******************************************************************************/

//______________________________________________________________________________
void TEveSelection::DoElementSelect(TEveSelection::SelMap_i entry)
{
   // Select element indicated by the entry and fill its
   // implied-selected set.

   TEveElement *el  = entry->first;
   Set_t       &set = entry->second;

   (el->*fSelElement)(kTRUE);
   el->FillImpliedSelectedSet(set);
   for (Set_i i = set.begin(); i != set.end(); ++i)
      ((*i)->*fIncImpSelElement)();
}

//______________________________________________________________________________
void TEveSelection::DoElementUnselect(TEveSelection::SelMap_i entry)
{
   // Deselect element indicated by the entry and clear its
   // implied-selected set.

   TEveElement *el  = entry->first;
   Set_t       &set = entry->second;

   for (Set_i i = set.begin(); i != set.end(); ++i)
      ((*i)->*fDecImpSelElement)();
   set.clear();
   (el->*fSelElement)(kFALSE);
}


/******************************************************************************/
// Overrides of child-element-management virtuals from TEveElement
/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveSelection::AcceptElement(TEveElement* el)
{
   // Pre-addition check. Deny addition if el is already selected.
   // Virtual from TEveElement.

   return el != this && fImpliedSelected.find(el) == fImpliedSelected.end() &&
          el->IsA()->InheritsFrom(TEveSelection::Class()) == kFALSE;
}

//______________________________________________________________________________
void TEveSelection::AddElement(TEveElement* el)
{
   // Add an element into selection, virtual from TEveElement.

   TEveElementList::AddElement(el);

   SelMap_i i = fImpliedSelected.insert(std::make_pair(el, Set_t())).first;
   if (fActive)
   {
      DoElementSelect(i);
   }
   SelectionAdded(el);
}

//______________________________________________________________________________
void TEveSelection::RemoveElement(TEveElement* el)
{
   // Add an element into selection, virtual from TEveElement.
   // Overriden here just so that a signal can be emitted.

   TEveElementList::RemoveElement(el);
   SelectionRemoved(el);
}

//______________________________________________________________________________
void TEveSelection::RemoveElementLocal(TEveElement* el)
{
   // Virtual from TEveElement.

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

//______________________________________________________________________________
void TEveSelection::RemoveElements()
{
   // Add an element into selection, virtual from TEveElement.
   // Overriden here just so that a signal can be emitted.

   TEveElementList::RemoveElements();
   SelectionCleared();
}

//______________________________________________________________________________
void TEveSelection::RemoveElementsLocal()
{
   // Virtual from TEveElement.

   if (fActive)
   {
      for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
         DoElementUnselect(i);
   }
   fImpliedSelected.clear();
}

//______________________________________________________________________________
void TEveSelection::RemoveImpliedSelected(TEveElement* el)
{
   // Remove element from all implied-selected sets.
   //
   // This is called as part of the element destruction from
   // TEveManager::PreDeleteElement() and should not be called
   // directly.

   for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
   {
      Set_i j = i->second.find(el);
      if (j != i->second.end())
         i->second.erase(j);
   }
}

//______________________________________________________________________________
void TEveSelection::RecheckImpliedSet(SelMap_i smi)
{
   // Recalculate implied-selected state for given selection entry.
   // Add new elements to implied-selected set and increase their
   // implied-selected count.

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

//______________________________________________________________________________
void TEveSelection::RecheckImpliedSetForElement(TEveElement* el)
{
   // If given element is selected or implied-selected with this
   // selection and recheck implied-set for given selection entry.

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


//******************************************************************************
// Signals
//******************************************************************************

//______________________________________________________________________________
void TEveSelection::SelectionAdded(TEveElement* el)
{
   // Emit SelectionAdded signal.

   Emit("SelectionAdded(TEveElement*)", (Long_t)el);
}

//______________________________________________________________________________
void TEveSelection::SelectionRemoved(TEveElement* el)
{
   // Emit SelectionRemoved signal.

   Emit("SelectionRemoved(TEveElement*)", (Long_t)el);
}

//______________________________________________________________________________
void TEveSelection::SelectionCleared()
{
   // Emit SelectionCleared signal.

   Emit("SelectionCleared()");
}

//______________________________________________________________________________
void TEveSelection::SelectionRepeated(TEveElement* el)
{
   // Called when secondary selection changed internally.

   Emit("SelectionRepeated(TEveElement*)", (Long_t)el);
}

/******************************************************************************/
// Activation / deactivation of selection
/******************************************************************************/

//______________________________________________________________________________
void TEveSelection::ActivateSelection()
{
   // Activate this selection.

   for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
      DoElementSelect(i);
   fActive = kTRUE;
}

//______________________________________________________________________________
void TEveSelection::DeactivateSelection()
{
   // Deactivate this selection.

   fActive = kFALSE;
   for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
      DoElementUnselect(i);
}


/******************************************************************************/
// User input processing
/******************************************************************************/

//______________________________________________________________________________
TEveElement* TEveSelection::MapPickedToSelected(TEveElement* el)
{
   // Given element el that was picked or clicked by the user, find
   // the parent/ancestor element that should actually become the main
   // selected element according to current selection mode.

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

//______________________________________________________________________________
void TEveSelection::UserPickedElement(TEveElement* el, Bool_t multi)
{
   // Called when user picks/clicks on an element. If multi is true,
   // the user is requiring a multiple selection (usually this is
   // associated with control-key being pressed at the time of pick
   // event).

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

//______________________________________________________________________________
void TEveSelection::UserRePickedElement(TEveElement* el)
{
   // Called when secondary selection becomes empty.

   el = MapPickedToSelected(el);
   if (el && HasChild(el))
   {
      SelectionRepeated(el);
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void TEveSelection::UserUnPickedElement(TEveElement* el)
{
   // Called when secondary selection becomes empty.

   el = MapPickedToSelected(el);
   if (el)
   {
      RemoveElement(el);
      gEve->Redraw3D();
   }
}
