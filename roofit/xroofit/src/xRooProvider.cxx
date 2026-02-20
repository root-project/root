/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RElement.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/TObjectItem.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/Browsable/RShared.hxx>

#include <RooFit/xRooFit/xRooNode.h>

#include "TVirtualPad.h"

#include "RooWorkspace.h"

using namespace ROOT::Browsable;
using namespace std::string_literals;
using namespace ROOT::Experimental::XRooFit;

class xRooBrowsingElement : public RElement {
   std::shared_ptr<xRooNode> fNode;
public:
   xRooBrowsingElement(std::shared_ptr<xRooNode> node)
   {
      fNode = node;
   }

   bool IsCapable(EActionKind action) const override
   {
      return (action == kActDraw6) || (action == kActBrowse);
   }

   /** Get default action */
   EActionKind GetDefaultAction() const override
   {
      if (fNode->IsFolder())
         return kActBrowse;
      return kActDraw6;
   }

   std::string GetName() const override
   {
      return fNode->GetName();
   }

   std::string GetTitle() const override
   {
      return fNode->GetTitle();
   }


   bool IsFolder() const override
   {
      return fNode->IsFolder();
   }

   int GetNumChilds() override
   {
      return fNode->IsFolder() ? (int) fNode->size() : 0;
   }

   std::unique_ptr<RHolder> GetObject() override
   {
      return std::make_unique<RShared<xRooNode>>(fNode);
   }

   std::unique_ptr<RLevelIter> GetChildsIter() override;
};

class xRooLevelIter : public RLevelIter {

   std::shared_ptr<xRooNode> fNode;

   int fCounter{-1};

public:
   explicit xRooLevelIter(std::shared_ptr<xRooNode> node) { fNode = node; }

   ~xRooLevelIter() override = default;

   auto NumElements() const { return fNode->size(); }

   bool Next() override { return ++fCounter < (int) fNode->size(); }

   std::string GetItemName() const override { return (*fNode)[fCounter]->GetName(); }

   bool CanItemHaveChilds() const override
   {
      return (*fNode)[fCounter]->IsFolder();
   }

   /** Create item for the browser */
   std::unique_ptr<RItem> CreateItem() override
   {
      auto xnode = (*fNode)[fCounter];

      auto item = std::make_unique<TObjectItem>(xnode->GetName(), xnode->size());

      item->SetTitle(xnode->GetTitle());

      item->SetClassName(xnode->get()->ClassName());
      item->SetIcon(RProvider::GetClassIcon(xnode->get()->IsA(), xnode->IsFolder()));

      return item;
   }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override
   {
      auto xnode = (*fNode)[fCounter];
      return std::make_shared<xRooBrowsingElement>(xnode);
   }

   bool Find(const std::string &name, int indx = -1) override
   {
      if ((indx >= 0) && (indx < (int) fNode->size()) && (name == (*fNode)[indx]->GetName())) {
         fCounter = indx;
         return true;
      }

      return RLevelIter::Find(name, -1);
   }
};


std::unique_ptr<RLevelIter> xRooBrowsingElement::GetChildsIter()
{
   // here central part of hole story
   // one need to provide list of sub-items via iterator

   fNode->browse();

   if (fNode->size() == 0)
      return nullptr;

   return std::make_unique<xRooLevelIter>(fNode);
}

// ==============================================================================================

class xRooProvider : public RProvider {

public:
   xRooProvider()
   {
      RegisterBrowse(RooWorkspace::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         auto wk = object->get_shared<RooWorkspace>();

         auto wkNode = std::make_shared<xRooNode>(wk);

         return std::make_shared<xRooBrowsingElement>(wkNode);
      });

      RegisterDraw6(xRooNode::Class(), [this](TVirtualPad *pad, std::unique_ptr<RHolder> &obj, const std::string &opt) -> bool {
         auto xnode = const_cast<xRooNode *>(obj->Get<xRooNode>());
         if (!xnode)
            return false;

         pad->cd();
         xnode->Draw(opt.c_str());
         return true;
      });

      // example how custom icons can be provided
      RegisterClass("RooRealVar", "sap-icon://picture");
   }

} newxRooProvider;
