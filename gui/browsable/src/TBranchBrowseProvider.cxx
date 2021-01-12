/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/TObjectElement.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>

#include "TBranchElement.h"

using namespace ROOT::Experimental::Browsable;

class TBrElement : public TObjectElement {

public:
   TBrElement(std::unique_ptr<RHolder> &br) : TObjectElement(br) {}

   virtual ~TBrElement() = default;

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      TBranchElement *br = const_cast<TBranchElement*> (fObject->Get<TBranchElement>()); // try to cast into TBranchElement
      if (!br) return nullptr;

      if (br->GetListOfBranches()->GetEntriesFast() > 0)
         return TObjectElement::GetChildsIter();

      return nullptr;
   }
};

// ==============================================================================================

class TBranchBrowseProvider : public RProvider {

public:
   TBranchBrowseProvider()
   {
      RegisterBrowse(TBranchElement::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TBrElement>(object);
      });
   }

} newTBranchBrowseProvider;
