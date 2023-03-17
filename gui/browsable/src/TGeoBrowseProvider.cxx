// Author: Sergey Linev, 17.03.2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/TObjectElement.hxx>
#include <ROOT/Browsable/RItem.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TROOT.h"


namespace ROOT {
namespace Experimental {
namespace Browsable {

////////////////////////////////////////////////////////////
/// Representing TGeoVolume in browsables

class TGeoVolumeElement : public TObjectElement {

public:
   TGeoVolumeElement(std::unique_ptr<RHolder> &br) : TObjectElement(br) {}

   TGeoVolume *GetVolume() const
   {
      if (!CheckObject()) return nullptr;
      return dynamic_cast<TGeoVolume *>(fObj);
   }

   bool IsFolder() const override
   {
      auto vol = GetVolume();
      return vol ? vol->GetNdaughters() > 0 : false;
   }

   /** Get default action */
   EActionKind GetDefaultAction() const override
   {
      return kActGeom;
   }

   /** Check if want to perform action */
   bool IsCapable(EActionKind action) const override
   {
      return action == kActGeom;
   }

   /** Provide iterator over TGeoVolume */
   std::unique_ptr<RLevelIter> GetChildsIter() override;
};


////////////////////////////////////////////////////////////
/// Representing TGeoNode in browsables

class TGeoNodeElement : public TObjectElement {

public:
   TGeoNodeElement(TGeoNode *node) : TObjectElement(node) {}

   TGeoNodeElement(std::unique_ptr<RHolder> &br) : TObjectElement(br) {}

   TGeoNode *GetNode() const
   {
      if (!CheckObject()) return nullptr;
      return dynamic_cast<TGeoNode *>(fObj);
   }

   /** Get default action */
   EActionKind GetDefaultAction() const override
   {
      return kActGeom;
   }

   /** Check if want to perform action */
   bool IsCapable(EActionKind action) const override
   {
      return action == kActGeom;
   }

   /** Provide iterator over TGeoVolume */
   std::unique_ptr<RLevelIter> GetChildsIter() override;
};


////////////////////////////////////////////////////////////
/// Iterating over nodes in the volume

class TGeoVolumeIter : public RLevelIter {
   TGeoIterator fIter;                ///<! iterator
   TGeoNode *fCurrent{nullptr};       ///<! current node

public:
   explicit TGeoVolumeIter(TGeoVolume *vol) : fIter(vol)
   {
      fIter.SetType(1); // iterate only current level
   }

   virtual ~TGeoVolumeIter() = default;

   bool Next() override
   {
      fCurrent = fIter.Next();
      return fCurrent != nullptr;
   }

   std::string GetItemName() const override { return fCurrent->GetName(); }

   bool CanItemHaveChilds() const override
   {
      return fCurrent ? fCurrent->IsFolder() : false;
   }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override
   {
      return std::make_shared<TGeoNodeElement>(fCurrent);
   }

   /** Returns full item information */
   std::unique_ptr<RItem> CreateItem() override
   {
      auto item = RLevelIter::CreateItem();
      item->SetIcon("sap-icon://product");
      return item;
   }
};


/** Provide iterator over TGeoManager */
std::unique_ptr<RLevelIter> TGeoVolumeElement::GetChildsIter()
{
   auto vol = GetVolume();
   return vol ? std::make_unique<TGeoVolumeIter>(vol) : nullptr;
}

/** Provide iterator over TGeoManager */
std::unique_ptr<RLevelIter> TGeoNodeElement::GetChildsIter()
{
   auto node = GetNode();
   return node ? std::make_unique<TGeoVolumeIter>(node->GetVolume()) : nullptr;
}


////////////////////////////////////////////////////////////
/// Representing TGeoManager in browsables

class TGeoManagerElement : public TObjectElement {

public:
   TGeoManagerElement(std::unique_ptr<RHolder> &br) : TObjectElement(br) {}

   const TObject *CheckObject() const override
   {
      // during TROOT destructor just forget about file reference
      if (!gROOT || gROOT->TestBit(TObject::kInvalidObject)) {
         ForgetObject();
         return nullptr;
      }

      return TObjectElement::CheckObject();
   }

   TGeoManager *GetMgr() const
   {
      if (!CheckObject()) return nullptr;
      return dynamic_cast<TGeoManager *>(fObj);
   }

   Long64_t GetSize() const override
   {
      auto mgr = GetMgr();
      return mgr ? 128 : -1;
   }

   bool IsFolder() const override { return true; }

   /** Get default action */
   EActionKind GetDefaultAction() const override
   {
      return kActGeom;
   }

   /** Check if want to perform action */
   bool IsCapable(EActionKind action) const override
   {
      return action == kActGeom;
   }

   /** Provide iterator over TGeoManager */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      auto mgr = GetMgr();

      return mgr ? std::make_unique<TGeoVolumeIter>(mgr->GetMasterVolume()) : nullptr;
   }

};


/////////////////////////////////////////////////////////////////////////////////
/// Provider for TGeo browsing

class TGeoBrowseProvider : public RProvider {

public:
   TGeoBrowseProvider()
   {
      RegisterBrowse(TGeoManager::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TGeoManagerElement>(object);
      });
      RegisterBrowse(TGeoVolume::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TGeoVolumeElement>(object);
      });
      RegisterBrowse(TGeoNode::Class(), [](std::unique_ptr<RHolder> &object) -> std::shared_ptr<RElement> {
         return std::make_shared<TGeoNodeElement>(object);
      });
   }

} newTGeoBrowseProvider;

} // namespace Browsable
} // namespace Experimental
} // namespace ROOT
