// @(#)root/eve7:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
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

#include "json.hpp"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class REveSelection
\ingroup REve
Make sure there is a SINGLE running REveSelection for each
selection type (select/highlight).
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveSelection::REveSelection(const std::string& n, const std::string& t, Color_t col) :
   REveElement(n, t),
   fPickToSelect  (kPS_Projectable),
   fActive        (kTRUE),
   fIsMaster      (kTRUE)
{
   SetupDefaultColorAndTransparency(col, true, false);

   // Managing complete selection state on element level.
   //
   // Method pointers for propagation of selected / implied selected state
   // to elements. This has to be done differently now -- and kept within
   // REveSelection.
   //
   // Also, see REveManager::PreDeleteElement. We might need some sort of
   // implied-selected-count after all (global, for all selections,
   // highlights) ... and traverse all selections if the element gets zapped.
   // Yup, we have it ...
   // XXXX but ... we can also go up to master and check there directly !!!!!
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

REveSelection::~REveSelection()
{
   DeactivateSelection();
   RemoveNieces();
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
}

////////////////////////////////////////////////////////////////////////////////
/// Select element indicated by the entry and fill its
/// implied-selected set.

void REveSelection::DoElementSelect(SelMap_i entry)
{
   Set_t &imp_set = entry->second.f_implied;

   entry->first->FillImpliedSelectedSet(imp_set);

   for (auto &imp_el : imp_set)  imp_el->IncImpliedSelected();
}

////////////////////////////////////////////////////////////////////////////////
/// Deselect element indicated by the entry and clear its
/// implied-selected set.

void REveSelection::DoElementUnselect(SelMap_i entry)
{
   Set_t &imp_set = entry->second.f_implied;

   for (auto &imp_el : imp_set)  imp_el->DecImpliedSelected();

   imp_set.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Check if elemenet el is selected (not implied selected).

bool REveSelection::HasNiece(REveElement *el) const
{
   return fMap.find(el) != fMap.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Check if any elements are selected.

bool REveSelection::HasNieces() const
{
   return ! fMap.empty();
}

////////////////////////////////////////////////////////////////////////////////
/// Pre-addition check. Deny addition if el is already selected.
/// Virtual from REveAunt.

bool REveSelection::AcceptNiece(REveElement* el)
{
   return el != this && fMap.find(el) == fMap.end() &&
          el->IsA()->InheritsFrom(TClass::GetClass<REveSelection>()) == kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Add an element into selection, virtual from REveAunt

void REveSelection::AddNieceInternal(REveElement* el)
{
   SelMap_i i = fMap.insert(std::make_pair(el, Record(el))).first;
   if (fActive)
   {
      DoElementSelect(i);
      SelectionAdded(el);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from REveAunt.

void REveSelection::RemoveNieceInternal(REveElement* el)
{
   SelMap_i i = fMap.find(el);

   if (i != fMap.end())
   {
      el->RemoveAunt(this);
      if (fActive)
      {
         DoElementUnselect(i);
         SelectionRemoved(el);
      }
      fMap.erase(i);
   }
   else
   {
      Warning("REveSelection::RemoveNieceLocal", "element not found in map.");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add an element into selection, virtual from REveAunt.
/// Overriden here just so that a signal can be emitted.

void REveSelection::RemoveNieces()
{
   for (SelMap_i i = fMap.begin(); i != fMap.end(); ++i)
   {
      i->first->RemoveAunt(this);
      DoElementUnselect(i);
   }
   fMap.clear();
   SelectionCleared();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove element from all implied-selected sets.
///
/// This is called as part of the element destruction from
/// REveManager::PreDeleteElement() and should not be called
/// directly.

void REveSelection::RemoveImpliedSelected(REveElement* el)
{
   for (SelMap_i i = fMap.begin(); i != fMap.end(); ++i)
   {
      Set_i j = i->second.f_implied.find(el);
      if (j != i->second.f_implied.end())
         i->second.f_implied.erase(j);
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
   for (auto i = set.begin(); i != set.end(); ++i)
   {
      if (smi->second.f_implied.find(*i) == smi->second.f_implied.end())
      {
         smi->second.f_implied.insert(*i);
         (*i)->IncImpliedSelected();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// If given element is selected or implied-selected within this
/// selection then recheck implied-set for given selection entry.

void REveSelection::RecheckImpliedSetForElement(REveElement* el)
{
   // Top-level selected.
   {
      SelMap_i i = fMap.find(el);
      if (i != fMap.end())
         RecheckImpliedSet(i);
   }

   // Implied selected (we can not tell if by this selection or some other),
   // then we need to loop over all.
   if (el->GetImpliedSelected() > 0)
   {
      for (SelMap_i i = fMap.begin(); i != fMap.end(); ++i)
      {
         if (i->second.f_implied.find(el) != i->second.f_implied.end())
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
/// Emit SelectionRepeated signal.

void REveSelection::SelectionRepeated(REveElement* /*el*/)
{
   // XXXX
   // Emit("SelectionRepeated(REveElement*)", (Long_t)el);
}

////////////////////////////////////////////////////////////////////////////////
/// Activate this selection.

void REveSelection::ActivateSelection()
{
   if (fActive) return;

   fActive = kTRUE;
   for (SelMap_i i = fMap.begin(); i != fMap.end(); ++i)
   {
      DoElementSelect(i);
      SelectionAdded(i->first);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Deactivate this selection.

void REveSelection::DeactivateSelection()
{
   if ( ! fActive) return;

   for (SelMap_i i = fMap.begin(); i != fMap.end(); ++i)
   {
      DoElementUnselect(i);
   }
   SelectionCleared();
   fActive = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Given element el that was picked or clicked by the user, find
/// the parent/ancestor element that should actually become the main
/// selected element according to current selection mode.

REveElement* REveSelection::MapPickedToSelected(REveElement* el)
{
   if (el == nullptr)
      return nullptr;

   if (el->ForwardSelection())
   {
      return el->ForwardSelection();
   }

   switch (fPickToSelect)
   {
      case kPS_Ignore:
      {
         return nullptr;
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
   el = MapPickedToSelected(el);

   if (el || HasChildren())
   {
      if ( ! multi)
         RemoveNieces();
      if (el)
      {
         if (HasNiece(el))
             RemoveNiece(el);
         else
            AddNiece(el);
      }
      if (fIsMaster)
         REX::gEve->ElementSelect(el);
      REX::gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Called when element selection is repeated.

void REveSelection::UserRePickedElement(REveElement* el)
{
   el = MapPickedToSelected(el);
   if (el && HasNiece(el))
   {
      SelectionRepeated(el);
      REX::gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Called when an element is unselected.

void REveSelection::UserUnPickedElement(REveElement* el)
{
   el = MapPickedToSelected(el);
   if (el && HasNiece(el))
   {
      RemoveNiece(el);
      REX::gEve->Redraw3D();
   }
}

//==============================================================================

void REveSelection::NewElementPicked(ElementId_t id, bool multi, bool secondary, const std::set<int>& secondary_idcs)
{
   static const REveException eh("REveSelection::NewElementPicked ");

   REveElement *pel = REX::gEve->FindElementById(id);

   if ( ! pel) throw eh + "picked element id=" + id + " not found.";

   REveElement *el  = MapPickedToSelected(pel);

   printf("REveSelection::NewElementPicked %p -> %p, multi: %d, secondary: %d", pel, el, multi, secondary);
   if (secondary)
   {
      printf(" { ");
      for (auto si : secondary_idcs) printf("%d ", si);
      printf("}");
   }
   printf("\n");

   Record *rec = find_record(el);

   if (multi)
   {
      if (el)
      {
         if (rec)
         {
            if (secondary || rec->is_secondary()) // ??? should actually be && ???
            {
               // XXXX union or difference:
               // - if all secondary_idcs are already in the record, toggle
               //   - if final result is empty set, remove element from selection
               // - otherwise union
            }
            else
            {
               // XXXX remove the existing record
            }
         }
         else
         {
            // XXXX insert the new record
         }
      }
      else
      {
         // Multiple selection with 0 element ... do nothing, I think.
      }
   }
   else // single selection (not multi)
   {
      if (el)
      {
         if (rec)
         {
            if (secondary || rec->is_secondary()) // ??? should actually be && ???
            {
               // if sets are identical, issue SelectionRepeated()
               // else modify record for the new one, issue Repeated
            }
            else
            {
               // clear selection
               // ??? should keep the newly clicked?
            }
         }
         else
         {
            if (HasNieces()) RemoveNieces();
            AddNiece(el);
         }
      }
      else // Single selection with zero element --> clear selection.
      {
         if (HasNieces()) RemoveNieces();
      }
   }

   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove pointers to el from implied selected sets.

int REveSelection::RemoveImpliedSelectedReferencesTo(REveElement *el)
{
   int count = 0;

   for (SelMap_i i = fMap.begin(); i != fMap.end(); ++i)
   {
      auto j = i->second.f_implied.find(el);

      if (j != i->second.f_implied.end())
      {
         i->second.f_implied.erase(j);
         ++count;
      }
   }

   return count;
}


////////////////////////////////////////////////////////////////////////////////
/// Write core json. If rnr_offset negative, render data will not be written

Int_t REveSelection::WriteCoreJson(nlohmann::json &j, Int_t /* rnr_offset */)
{
   REveElement::WriteCoreJson(j, -1);

   nlohmann::json sel_list = nlohmann::json::array();

   for (SelMap_i i = fMap.begin(); i != fMap.end(); ++i)
   {
      nlohmann::json rec = {}, imp = nlohmann::json::array(), sec = nlohmann::json::array();

      rec["primary"] = i->first->GetElementId();

      // XXX if not empty ???
      for (auto &imp_el : i->second.f_implied) imp.push_back(imp_el->GetElementId());
      rec["implied"]  = imp;

      // XXX if not empty / f_is_sec is false ???
      for (auto &sec_id : i->second.f_sec_idcs) sec.push_back(sec_id);
      rec["sec_idcs"] = sec;

      sel_list.push_back(rec);
   }

   j["sel_list"] = sel_list;

   j["UT_PostStream"] = "UT_Selection_Refresh_State"; // XXXX to be canonized

   // std::cout << j.dump(2) << std::endl;

   return 0;
}
