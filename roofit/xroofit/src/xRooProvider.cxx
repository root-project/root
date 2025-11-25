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

#include <RooFit/xRooFit/xRooNode.h>


#include "RooWorkspace.h"

using namespace ROOT::Browsable;
using namespace std::string_literals;


class xRooBrowsingElement : public RElement {
   std::shared_ptr<ROOT::Experimental::XRooFit::xRooNode> fNode;
public:
   xRooBrowsingElement(std::shared_ptr<ROOT::Experimental::XRooFit::xRooNode> node);
   bool IsCapable(EActionKind action) const override;
   /** Get default action */
   EActionKind GetDefaultAction() const override;

   std::string GetName() const override { return fNode->GetName(); }

   std::unique_ptr<RLevelIter> GetChildsIter() override;
};

class xRooLevelIter : public RLevelIter {

   std::shared_ptr<ROOT::Experimental::XRooFit::xRooNode> fNode;

   int fCounter{-1};

public:
   explicit xRooLevelIter(std::shared_ptr<ROOT::Experimental::XRooFit::xRooNode> node) { fNode = node; }

   ~xRooLevelIter() override = default;

   auto NumElements() const { return fNode->size(); }

   bool Next() override { return ++fCounter < (int) fNode->size(); }

   // use default implementation for now
   // bool Find(const std::string &name) override { return FindDirEntry(name); }

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


xRooBrowsingElement::xRooBrowsingElement(std::shared_ptr<ROOT::Experimental::XRooFit::xRooNode> node)
{
   fNode = node;
}

   /** Check if want to perform action */
bool xRooBrowsingElement::IsCapable(RElement::EActionKind action) const
{
   return action == kActDraw6;
}

   /** Get default action */
RElement::EActionKind xRooBrowsingElement::GetDefaultAction() const
{
   return kActDraw6;
}

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

         printf("Create entry for workspace %s %s\n", wk->GetName(), wk->ClassName());

         auto wkNode = std::make_shared<ROOT::Experimental::XRooFit::xRooNode>(wk);

         return std::make_shared<xRooBrowsingElement>(wkNode);
      });
   }

} newxRooProvider;
