// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveSelection
#define ROOT_TEveSelection

#include "TEveElement.h"

#include "TQObject.h"

#include <map>

class TEveSelection : public TEveElementList,
                      public TQObject
{
public:
   enum EPickToSelect  // How to convert picking events to top selected element:
   {
      kPS_Ignore,        // ignore picking
      kPS_Element,       // select element (default for selection)
      kPS_Projectable,   // select projectable
      kPS_Compound,      // select compound
      kPS_PableCompound, // select projectable and compound
      kPS_Master         // select master element (top-level compound)
   };

private:
   TEveSelection(const TEveSelection&);            // Not implemented
   TEveSelection& operator=(const TEveSelection&); // Not implemented

protected:
   typedef std::map<TEveElement*, Set_t>           SelMap_t;
   typedef std::map<TEveElement*, Set_t>::iterator SelMap_i;

   Int_t            fPickToSelect;
   Bool_t           fActive;
   Bool_t           fIsMaster;

   SelMap_t         fImpliedSelected;

   Select_foo       fSelElement;
   ImplySelect_foo  fIncImpSelElement;
   ImplySelect_foo  fDecImpSelElement;

   void DoElementSelect  (SelMap_i entry);
   void DoElementUnselect(SelMap_i entry);

   void RecheckImpliedSet(SelMap_i smi);

public:
   TEveSelection(const char* n="TEveSelection", const char* t="");
   virtual ~TEveSelection() {}

   void SetHighlightMode();

   Int_t  GetPickToSelect()   const { return fPickToSelect; }
   void   SetPickToSelect(Int_t ps) { fPickToSelect = ps; }

   Bool_t GetIsMaster()       const { return fIsMaster; }
   void   SetIsMaster(Bool_t m)     { fIsMaster = m; }

   virtual Bool_t AcceptElement(TEveElement* el);

   virtual void AddElement(TEveElement* el);
   virtual void RemoveElement(TEveElement* el);
   virtual void RemoveElementLocal(TEveElement* el);
   virtual void RemoveElements();
   virtual void RemoveElementsLocal();

   virtual void RemoveImpliedSelected(TEveElement* el);

   void RecheckImpliedSetForElement(TEveElement* el);

   void SelectionAdded(TEveElement* el);    // *SIGNAL*
   void SelectionRemoved(TEveElement* el);  // *SIGNAL*
   void SelectionCleared();                 // *SIGNAL*
   void SelectionRepeated(TEveElement* el); // *SIGNAL*

   // ----------------------------------------------------------------
   // Interface to make selection active/non-active.

   virtual void ActivateSelection();
   virtual void DeactivateSelection();

   // ----------------------------------------------------------------
   // User input processing.

   TEveElement* MapPickedToSelected(TEveElement* el);

   virtual void UserPickedElement(TEveElement* el, Bool_t multi=kFALSE);
   virtual void UserRePickedElement(TEveElement* el);
   virtual void UserUnPickedElement(TEveElement* el);

   // ----------------------------------------------------------------

   ClassDef(TEveSelection, 0); // Container for selected and highlighted elements.
};

#endif
