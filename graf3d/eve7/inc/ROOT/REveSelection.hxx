// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
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

class REveSelection : public REveElementList {
public:
   enum EPickToSelect   // How to convert picking events to top selected element:
   { kPS_Ignore,        // ignore picking
     kPS_Element,       // select element (default for selection)
     kPS_Projectable,   // select projectable
     kPS_Compound,      // select compound
     kPS_PableCompound, // select projectable and compound
     kPS_Master         // select master element (top-level compound)
   };

private:
   REveSelection(const REveSelection &);            // Not implemented
   REveSelection &operator=(const REveSelection &); // Not implemented

protected:
   typedef std::map<REveElement *, Set_t> SelMap_t;
   typedef std::map<REveElement *, Set_t>::iterator SelMap_i;

   Int_t fPickToSelect;
   Bool_t fActive;
   Bool_t fIsMaster;

   SelMap_t fImpliedSelected;

   Select_foo fSelElement;
   ImplySelect_foo fIncImpSelElement;
   ImplySelect_foo fDecImpSelElement;

   void DoElementSelect(SelMap_i entry);
   void DoElementUnselect(SelMap_i entry);

   void RecheckImpliedSet(SelMap_i smi);

public:
   REveSelection(const char *n = "REveSelection", const char *t = "");
   virtual ~REveSelection() {}

   void SetHighlightMode();

   Int_t GetPickToSelect() const { return fPickToSelect; }
   void SetPickToSelect(Int_t ps) { fPickToSelect = ps; }

   Bool_t GetIsMaster() const { return fIsMaster; }
   void SetIsMaster(Bool_t m) { fIsMaster = m; }

   virtual Bool_t AcceptElement(REveElement *el);

   virtual void AddElement(REveElement *el);
   virtual void RemoveElement(REveElement *el);
   virtual void RemoveElementLocal(REveElement *el);
   virtual void RemoveElements();
   virtual void RemoveElementsLocal();

   virtual void RemoveImpliedSelected(REveElement *el);

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

   // ----------------------------------------------------------------

   ClassDef(REveSelection, 0); // Container for selected and highlighted elements.
};

} // namespace Experimental
} // namespace ROOT

#endif
