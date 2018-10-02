// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveSelection.hxx>
#include <ROOT/REveProjectionBases.hxx>
#include <ROOT/REveCompound.hxx>
#include <ROOT/REveManager.hxx>

#include "TClass.h"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class REveSelection
\ingroup REve
Make sure there is a SINGLE running REveSelection for each
selection type (select/highlight).
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveSelection::REveSelection(const char* n, const char* t) :
   REveElementList(n, t),
   fPickToSelect  (kPS_Projectable),
   fActive        (kTRUE),
   fIsMaster      (kTRUE)
{
   fSelElement       = &REveElement::SelectElement;
   fIncImpSelElement = &REveElement::IncImpliedSelected;
   fDecImpSelElement = &REveElement::DecImpliedSelected;
}

////////////////////////////////////////////////////////////////////////////////
/// Set to 'highlight' mode.

void REveSelection::SetHighlightMode()
{
   // Most importantly, this sets the pointers-to-function-members in
   // REveElement that are used to mark elements as (un)selected and
   // implied-(un)selected.

   fPickToSelect = kPS_Projectable;
   fIsMaster     = kFALSE;

   fSelElement       = &REveElement::HighlightElement;
   fIncImpSelElement = &REveElement::IncImpliedHighlighted;
   fDecImpSelElement = &REveElement::DecImpliedHighlighted;
}

////////////////////////////////////////////////////////////////////////////////
/// Select element indicated by the entry and fill its
/// implied-selected set.

void REveSelection::DoElementSelect(REveSelection::SelMap_i entry)
{
   REveElement *el  = entry->first;
   Set_t       &set = entry->second;

   (el->*fSelElement)(kTRUE);
   el->FillImpliedSelectedSet(set);
   for (Set_i i = set.begin(); i != set.end(); ++i)
      ((*i)->*fIncImpSelElement)();
}

////////////////////////////////////////////////////////////////////////////////
/// Deselect element indicated by the entry and clear its
/// implied-selected set.

void REveSelection::DoElementUnselect(REveSelection::SelMap_i entry)
{
   REveElement *el  = entry->first;
   Set_t       &set = entry->second;

   for (Set_i i = set.begin(); i != set.end(); ++i)
      ((*i)->*fDecImpSelElement)();
   set.clear();
   (el->*fSelElement)(kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Pre-addition check. Deny addition if el is already selected.
/// Virtual from REveElement.

Bool_t REveSelection::AcceptElement(REveElement* el)
{
   return el != this && fImpliedSelected.find(el) == fImpliedSelected.end() &&
          el->IsA()->InheritsFrom(REveSelection::Class()) == kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Add an element into selection, virtual from REveElement.

void REveSelection::AddElement(REveElement* el)
{
   REveElementList::AddElement(el);

   SelMap_i i = fImpliedSelected.insert(std::make_pair(el, Set_t())).first;
   if (fActive)
   {
      DoElementSelect(i);
   }
   SelectionAdded(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Add an element into selection, virtual from REveElement.
/// Overriden here just so that a signal can be emitted.

void REveSelection::RemoveElement(REveElement* el)
{
   REveElementList::RemoveElement(el);
   SelectionRemoved(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from REveElement.

void REveSelection::RemoveElementLocal(REveElement* el)
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
      Warning("REveSelection::RemoveElementLocal", "element not found in map.");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add an element into selection, virtual from REveElement.
/// Overriden here just so that a signal can be emitted.

void REveSelection::RemoveElements()
{
   REveElementList::RemoveElements();
   SelectionCleared();
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from REveElement.

void REveSelection::RemoveElementsLocal()
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
/// REveManager::PreDeleteElement() and should not be called
/// directly.

void REveSelection::RemoveImpliedSelected(REveElement* el)
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

void REveSelection::RecheckImpliedSet(SelMap_i smi)
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

void REveSelection::RecheckImpliedSetForElement(REveElement* el)
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

void REveSelection::SelectionAdded(REveElement* /*el*/)
{
   // XXXX
   // Emit("SelectionAdded(REveElement*)", (Long_t)el);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit SelectionRemoved signal.

void REveSelection::SelectionRemoved(REveElement* /*el*/)
{
   // XXXX
   // Emit("SelectionRemoved(REveElement*)", (Long_t)el);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit SelectionCleared signal.

void REveSelection::SelectionCleared()
{
   // XXXX
   // Emit("SelectionCleared()");
}

////////////////////////////////////////////////////////////////////////////////
/// Called when secondary selection changed internally.

void REveSelection::SelectionRepeated(REveElement* /*el*/)
{
   // XXXX
   // Emit("SelectionRepeated(REveElement*)", (Long_t)el);
}

////////////////////////////////////////////////////////////////////////////////
/// Activate this selection.

void REveSelection::ActivateSelection()
{
   for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
      DoElementSelect(i);
   fActive = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Deactivate this selection.

void REveSelection::DeactivateSelection()
{
   fActive = kFALSE;
   for (SelMap_i i = fImpliedSelected.begin(); i != fImpliedSelected.end(); ++i)
      DoElementUnselect(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Given element el that was picked or clicked by the user, find
/// the parent/ancestor element that should actually become the main
/// selected element according to current selection mode.

REveElement* REveSelection::MapPickedToSelected(REveElement* el)
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
         REveProjected* pted = dynamic_cast<REveProjected*>(el);
         if (pted)
            return dynamic_cast<REveElement*>(pted->GetProjectable());
         return el;
      }
      case kPS_Compound:
      {
         REveElement* cmpnd = el->GetCompound();
         if (cmpnd)
            return cmpnd;
         return el;
      }
      case kPS_PableCompound:
      {
         REveProjected* pted = dynamic_cast<REveProjected*>(el);
         if (pted)
            el = dynamic_cast<REveElement*>(pted->GetProjectable());
         REveElement* cmpnd = el->GetCompound();
         if (cmpnd)
            return cmpnd;
         return el;
      }
      case kPS_Master:
      {
         REveElement* mstr = el->GetMaster();
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

void REveSelection::UserPickedElement(REveElement* el, Bool_t multi)
{
   REveElement *edit_el = el ? el->ForwardEdit() : 0;

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
         REX::gEve->ElementSelect(edit_el ? edit_el : el);
      REX::gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Called when secondary selection becomes empty.

void REveSelection::UserRePickedElement(REveElement* el)
{
   el = MapPickedToSelected(el);
   if (el && HasChild(el))
   {
      SelectionRepeated(el);
      REX::gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Called when secondary selection becomes empty.

void REveSelection::UserUnPickedElement(REveElement* el)
{
   el = MapPickedToSelected(el);
   if (el)
   {
      RemoveElement(el);
      REX::gEve->Redraw3D();
   }
}
