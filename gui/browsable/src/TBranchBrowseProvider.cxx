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

#include "TTree.h"
#include "TNtuple.h"
#include "TBranch.h"
#include "TBranchElement.h"
#include "TBranchBrowsable.h"

using namespace ROOT::Experimental::Browsable;


////////////////////////////////////////////////////////////
/// Representing TBranch in browsables

class TBrElement : public TObjectElement {

public:
   TBrElement(std::unique_ptr<RHolder> &br) : TObjectElement(br) {}

   Long64_t GetSize() const override
   {
      auto br = fObject->Get<TBranch>();
      return br ? br->GetTotalSize() : -1;
   }
};

////////////////////////////////////////////////////////////
/// Representing TTree in browsables

class TTreeElement : public TObjectElement {

public:
   TTreeElement(std::unique_ptr<RHolder> &br) : TObjectElement(br) {}

   Long64_t GetSize() const override
   {
      auto tr = fObject->Get<TTree>();
      printf("Return TTree size %ld\n", (long) (tr ? tr->GetTotBytes() : -1));
      return tr ? tr->GetTotBytes() : -1;
   }
};


////////////////////////////////////////////////////////////
/// Representing TVirtualBranchBrowsable in browsables

class TBrBrowsableElement : public TObjectElement {

public:
   TBrBrowsableElement(std::unique_ptr<RHolder> &br) : TObjectElement(br) {}

   virtual ~TBrBrowsableElement() = default;

   int GetNumChilds() override
   {
      auto br = fObject->Get<TVirtualBranchBrowsable>();
      return br && br->GetLeaves() ? br->GetLeaves()->GetSize() : 0;
   }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      auto br = fObject->Get<TVirtualBranchBrowsable>();
      if (br && br->GetLeaves())
         return GetCollectionIter(br->GetLeaves());
      return nullptr;
   }
};


// ==============================================================================================

class TBranchBrowseProvider : public RProvider {

public:
   TBranchBrowseProvider()
   {
      RegisterBrowse(TBranch::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TBrElement>(object);
      });
      RegisterBrowse(TBranchElement::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TBrElement>(object);
      });
      RegisterBrowse(TTree::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TTreeElement>(object);
      });
      RegisterBrowse(TNtuple::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TTreeElement>(object);
      });
   }

} newTBranchBrowseProvider;
