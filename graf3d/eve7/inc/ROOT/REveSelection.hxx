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

////////////////////////////////////////////////////////////////////////////////
/// REveSelection
/// Container for selected and highlighted elements.
////////////////////////////////////////////////////////////////////////////////

class REveSelection : public REveElement,
                      public REveAunt
{
   friend class Deviator;

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
      REveElement    *f_primary{nullptr}; ///<! it's also implied through the map -- XXXX do i need it ????
      Set_t           f_implied;
      std::set<int>   f_sec_idcs;
      bool            f_is_sec{false};   ///<! is secondary-selected -- XXXX do i need it ????

      Record(REveElement *el) :
         f_primary (el),
         f_is_sec  (false)
      {
         // Apparently done in DoElementSelect
         // el->FillImpliedSelectedSet(f_implied);
      }

      Record(REveElement *el, const std::set<int>& secondary_idcs) :
         f_primary  (el),
         f_sec_idcs (secondary_idcs),
         f_is_sec   (true)
      {
         // Apparently done in DoElementSelect
         // el->FillImpliedSelectedSet(f_implied);
      }

      bool is_secondary() const { return f_is_sec; }
   };

   class Deviator {
    public:
      virtual ~Deviator(){};
      Deviator() {}
      virtual bool DeviateSelection(REveSelection* s, REveElement* el, bool multi, bool secondary, const std::set<int>& secondary_idcs) = 0;
   protected:
      void ExecuteNewElementPicked(REveSelection* s, REveElement* el, bool multi, bool secondary, const std::set<int>& secondary_idcs)
      {
         s->NewElementPickedInternal(el, multi, secondary, secondary_idcs);
      }
   };

   typedef std::map<REveElement*, Record>  SelMap_t;
   typedef SelMap_t::iterator              SelMap_i;

private:
   REveSelection(const REveSelection &) = delete;
   REveSelection &operator=(const REveSelection &) = delete;

protected:
   Color_t          fVisibleEdgeColor; ///<!
   Color_t          fHiddenEdgeColor;  ///<!

   std::vector<int> fPickToSelect;     ///<!
   bool             fActive{true};     ///<!
   bool             fIsMaster{false};  ///<!
   bool             fIsHighlight{false}; ///<!

   SelMap_t         fMap;              ///<!
   
   std::shared_ptr<Deviator>        fDeviator;///<!

   Record* find_record(REveElement *el)
   {
      auto i = fMap.find(el);
      return i != fMap.end() ? & i->second : nullptr;
   }

   void DoElementSelect  (SelMap_i &entry);
   void DoElementUnselect(SelMap_i &entry);

   void RecheckImpliedSet(SelMap_i &entry);

   void AddNieceForSelection(REveElement*, bool secondary, const std::set<int>&);

   void NewElementPickedInternal(REveElement* el, bool multi, bool secondary, const std::set<int>& secondary_idcs);

public:
   REveSelection(const std::string &n = "REveSelection", const std::string &t = "",
                 Color_t col_visible = kViolet, Color_t col_hidden = kPink);
   virtual ~REveSelection();

   void   SetVisibleEdgeColorRGB(UChar_t r, UChar_t g, UChar_t b);
   void   SetHiddenEdgeColorRGB(UChar_t r, UChar_t g, UChar_t b);

   const std::vector<int>& RefPickToSelect()  const { return fPickToSelect; }
   void   ClearPickToSelect()     { fPickToSelect.clear(); }
   void   AddPickToSelect(int ps) { fPickToSelect.push_back(ps); }

   bool GetIsMaster() const { return fIsMaster; }
   void SetIsMaster(bool m) { fIsMaster = m; }
   bool GetIsHighlight() const { return fIsHighlight; }
   void SetIsHighlight(bool m) { fIsHighlight = m; }

   std::shared_ptr<Deviator> GetDeviator() const { return fDeviator; }
   void   SetDeviator(std::shared_ptr<Deviator> d) { fDeviator = d; }

   bool   IsEmpty()  const { return   fMap.empty(); }
   bool   NotEmpty() const { return ! fMap.empty(); }

   // Abstract methods of REveAunt
   bool HasNiece(REveElement *el) const override;
   bool HasNieces() const override;
   bool AcceptNiece(REveElement *el) override;
   void AddNieceInternal(REveElement *el) override;
   void RemoveNieceInternal(REveElement *el) override;
   void RemoveNieces() override;

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
   void NewElementPickedStr(ElementId_t id, bool multi, bool secondary, const char* secondary_idcs="");
   void ClearSelection();

   int  RemoveImpliedSelectedReferencesTo(REveElement *el);

   // ----------------------------------------------------------------

   Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset) override;

};

} // namespace Experimental
} // namespace ROOT

#endif
