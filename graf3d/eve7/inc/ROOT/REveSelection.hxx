// @(#)root/eve7:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveSelection
#define ROOT7_REveSelection

#include <ROOT/REveElement.hxx>

#include <map>

namespace ROOT {
namespace Experimental {

class REveSelection : public REveElement,
                      public REveAunt
{
public:
   enum EPickToSelect   // How to convert picking events to top selected element:
   { kPS_Ignore,        // ignore picking
     kPS_Element,       // select element (default for selection)
     kPS_Projectable,   // select projectable
     kPS_Compound,      // select compound
     kPS_PableCompound, // select projectable and compound
     kPS_Master         // select master element (top-level compound)
   };

   struct Record
   {
      REveElement    *f_primary; // it's also implied through the map -- XXXX do i need it ????
      Set_t           f_implied;
      std::set<int>   f_sec_idcs;
      bool            f_is_sec;   // is secondary-selected -- XXXX do i need it ????

      Record(REveElement *el) :
         f_primary (el),
         f_is_sec  (false)
      {
         el->FillImpliedSelectedSet(f_implied);
      }

      Record(REveElement *el, const std::set<int>& secondary_idcs) :
         f_primary  (el),
         f_sec_idcs (secondary_idcs),
         f_is_sec   (true)
      {
         el->FillImpliedSelectedSet(f_implied);
      }

      bool is_secondary() const { return f_is_sec; }
   };

   typedef std::map<REveElement*, Record> SelMap_t;
   typedef SelMap_t::iterator             SelMap_i;

private:
   REveSelection(const REveSelection &);            // Not implemented
   REveSelection &operator=(const REveSelection &); // Not implemented

protected:
   Int_t            fPickToSelect;
   Bool_t           fActive;
   Bool_t           fIsMaster;

   SelMap_t         fMap;

   Record* find_record(REveElement *el)
   {
      auto i = fMap.find(el);
      return i != fMap.end() ? & i->second : nullptr;
   }

   void DoElementSelect  (SelMap_i entry);
   void DoElementUnselect(SelMap_i entry);

   void RecheckImpliedSet(SelMap_i smi);

public:
   REveSelection(const std::string& n = "REveSelection", const std::string& t = "", Color_t col = kViolet);
   virtual ~REveSelection();

   void   SetHighlightMode();

   Int_t  GetPickToSelect()   const { return fPickToSelect; }
   void   SetPickToSelect(Int_t ps) { fPickToSelect = ps; }

   Bool_t GetIsMaster()   const { return fIsMaster; }
   void   SetIsMaster(Bool_t m) { fIsMaster = m; }


   // Abstract methods of REveAunt
   bool HasNiece(REveElement *el) const; // override;
   bool HasNieces() const; // override;
   bool AcceptNiece(REveElement *el); // override;
   void AddNieceInternal(REveElement *el); // override;
   void RemoveNieceInternal(REveElement *el); // override;
   void RemoveNieces(); // override;

   void RemoveImpliedSelected(REveElement *el);

   void RecheckImpliedSetForElement(REveElement *el);

   void SelectionAdded(REveElement *el);    // *SIGNAL*
   void SelectionRemoved(REveElement *el);  // *SIGNAL*
   void SelectionCleared();                 // *SIGNAL*
   void SelectionRepeated(REveElement *el); // *SIGNAL*

   // ----------------------------------------------------------------
   // Interface to make selection active/non-active.

   virtual void ActivateSelection();
   virtual void DeactivateSelection();

   // ----------------------------------------------------------------
   // User input processing.

   REveElement *MapPickedToSelected(REveElement *el);

   virtual void UserPickedElement(REveElement *el, Bool_t multi = kFALSE);
   virtual void UserRePickedElement(REveElement *el);
   virtual void UserUnPickedElement(REveElement *el);

   void NewElementPicked(ElementId_t id, bool multi, bool secondary, const std::set<int>& secondary_idcs={});

   int  RemoveImpliedSelectedReferencesTo(REveElement *el);

   // ----------------------------------------------------------------

   Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset); // override;

   // ----------------------------------------------------------------

   ClassDef(REveSelection, 0); // Container for selected and highlighted elements.
};

} // namespace Experimental
} // namespace ROOT

#endif
