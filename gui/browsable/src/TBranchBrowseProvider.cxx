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

#include "TLeafProvider.hxx"

#include "TTree.h"
#include "TNtuple.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TBranchElement.h"
#include "TBranchBrowsable.h"

using namespace ROOT::Browsable;
using namespace std::string_literals;


class TTreeBrowsingElement : public TObjectElement {
public:
   TTreeBrowsingElement(std::unique_ptr<RHolder> &obj) : TObjectElement(obj) {}

   /** Check if want to perform action */
   bool IsCapable(EActionKind action) const override
   {
      return (action == kActTree) || (action == kActDraw6) || (action == kActDraw7);
   }

   /** Get default action */
   EActionKind GetDefaultAction() const override
   {
      return GetDrawExpr().empty() ? kActTree : kActDraw6;
   }

   virtual std::string GetDrawExpr() const { return ""s; }

   std::string GetContent(const std::string &kind) override
   {
      if (kind == "tree"s)
         return GetDrawExpr();

      return TObjectElement::GetContent(kind);
   }

};


////////////////////////////////////////////////////////////
/// Representing TLeaf in browsables

class TBrLeafElement : public TTreeBrowsingElement {

public:
   TBrLeafElement(std::unique_ptr<RHolder> &leaf) : TTreeBrowsingElement(leaf) {}

   Long64_t GetSize() const override
   {
      auto leaf = fObject->Get<TLeaf>();
      if (!leaf)
         return -1;

      if (!leaf->GetBranch())
         return -1;

      return leaf->GetBranch()->GetTotalSize();
   }

   std::string GetDrawExpr() const override
   {
      auto leaf = fObject->Get<TLeaf>();

      TLeafProvider provider;
      TString expr, name;

      if (provider.GetDrawExpr(leaf, expr, name))
         return expr.Data();

      return ""s;
   }

};

////////////////////////////////////////////////////////////
/// Representing TBranch in browsables

class TBrElement : public TTreeBrowsingElement {

public:
   TBrElement(std::unique_ptr<RHolder> &br) : TTreeBrowsingElement(br) {}

   Long64_t GetSize() const override
   {
      auto br = fObject->Get<TBranch>();
      return br ? br->GetTotalSize() : -1;
   }

   std::string GetDrawExpr() const override
   {
      auto br = fObject->Get<TBranch>();
      auto elem = fObject->Get<TBranchElement>();

      TLeafProvider provider;
      TString expr, name;

      if (provider.GetDrawExpr(elem, expr, name))
         return expr.Data();

      if (provider.GetDrawExpr(br, expr, name))
         return expr.Data();

      return ""s;
   }

};

////////////////////////////////////////////////////////////
/// Representing TVirtualBranchBrowsable in browsables

class TBrBrowsableElement : public TTreeBrowsingElement {

public:
   TBrBrowsableElement(std::unique_ptr<RHolder> &br) : TTreeBrowsingElement(br) {}

   ~TBrBrowsableElement() override = default;

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

   std::string GetDrawExpr() const override
   {
      auto browsable = fObject->Get<TVirtualBranchBrowsable>();

      TLeafProvider provider;
      TString expr, name;

      if (provider.GetDrawExpr(browsable, expr, name))
         return expr.Data();

      return ""s;
   }

};

// ==============================================================================================

////////////////////////////////////////////////////////////
/// Representing TTree in browsables

class TTreeElement : public TObjectElement {

public:
   TTreeElement(std::unique_ptr<RHolder> &br) : TObjectElement(br) {}

   Long64_t GetSize() const override
   {
      auto tr = dynamic_cast<const TTree *>(CheckObject());
      return tr ? tr->GetTotBytes() : -1;
   }

   /** Get default action */
   EActionKind GetDefaultAction() const override
   {
      return kActTree;
   }

   /** Check if want to perform action */
   bool IsCapable(EActionKind action) const override
   {
      return action == kActTree;
   }

};

// ==============================================================================================

class TBranchBrowseProvider : public RProvider {

public:
   TBranchBrowseProvider()
   {
      RegisterBrowse(TLeaf::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TBrLeafElement>(object);
      });
      RegisterBrowse(TBranch::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TBrElement>(object);
      });
      RegisterBrowse(TBranchElement::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TBrElement>(object);
      });
      RegisterBrowse(TVirtualBranchBrowsable::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TBrBrowsableElement>(object);
      });
      RegisterBrowse(TTree::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TTreeElement>(object);
      });
      RegisterBrowse(TNtuple::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TTreeElement>(object);
      });
   }

} newTBranchBrowseProvider;
