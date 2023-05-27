// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGeoNode
#define ROOT_TEveGeoNode

#include "TEveElement.h"
#include <list>

class TGeoVolume;
class TGeoNode;
class TGeoHMatrix;
class TGeoManager;
class TGeoShape;

class TEveGeoShapeExtract;

//----------------------------------------------------------------

class TEveGeoNode : public TEveElement,
                    public TObject
{
   friend class TEveGeoNodeEditor;

   TEveGeoNode(const TEveGeoNode&);            // Not implemented
   TEveGeoNode& operator=(const TEveGeoNode&); // Not implemented

protected:
   TGeoNode *fNode;
   TEveGeoShapeExtract* DumpShapeTree(TEveGeoNode* geon, TEveGeoShapeExtract* parent=nullptr, Bool_t leafs_only=kFALSE);

   static Int_t                  fgCSGExportNSeg;  //!
   static std::list<TGeoShape*>  fgTemporaryStore; //!

public:
   TEveGeoNode(TGeoNode* node);

   TObject* GetObject(const TEveException&) const override
   { const TObject* obj = this; return const_cast<TObject*>(obj); }

   const char* GetName()  const override;
   const char* GetTitle() const override;
   const char* GetElementName()  const override;
   const char* GetElementTitle() const override;

   TGeoNode* GetNode() const { return fNode; }

   void   ExpandIntoListTree(TGListTree* ltree, TGListTreeItem* parent) override;

   virtual void   ExpandIntoListTrees();
   virtual void   ExpandIntoListTreesRecursively();

   Bool_t CanEditElement() const override { return kFALSE; }

   void   AddStamp(UChar_t bits) override;

   Bool_t CanEditMainColor() const override;
   void   SetMainColor(Color_t color) override;

   Bool_t  CanEditMainTransparency() const override;
   Char_t  GetMainTransparency() const override;
   void    SetMainTransparency(Char_t t) override;

   void UpdateNode(TGeoNode* node);
   void UpdateVolume(TGeoVolume* volume);

   void Save(const char* file, const char* name="Extract", Bool_t leafs_only=kFALSE);
   void SaveExtract(const char* file, const char* name, Bool_t leafs_only);
   void WriteExtract(const char* name, Bool_t leafs_only);

   void Draw(Option_t* option="") override;

   static Int_t GetCSGExportNSeg();
   static void  SetCSGExportNSeg(Int_t nseg);

   ClassDefOverride(TEveGeoNode, 0); // Wrapper for TGeoNode that allows it to be shown in GUI and controlled as a TEveElement.
};

//----------------------------------------------------------------

class TEveGeoTopNode : public TEveGeoNode
{
   TEveGeoTopNode(const TEveGeoTopNode&);            // Not implemented
   TEveGeoTopNode& operator=(const TEveGeoTopNode&); // Not implemented

protected:
   TGeoManager* fManager;
   Int_t        fVisOption;
   Int_t        fVisLevel;
   Int_t        fMaxVisNodes;

public:
   TEveGeoTopNode(TGeoManager* manager, TGeoNode* node, Int_t visopt=1,
                  Int_t vislvl=3, Int_t maxvisnds=10000);
   ~TEveGeoTopNode() override {}

   void         UseNodeTrans();

   TGeoManager* GetGeoManager() const { return fManager; }

   Int_t GetVisOption()      const { return fVisOption; }
   void  SetVisOption(Int_t vo)    { fVisOption = vo;   }
   Int_t GetVisLevel()       const { return fVisLevel;  }
   void  SetVisLevel(Int_t vl)     { fVisLevel = vl;    }
   Int_t GetMaxVisNodes()    const { return fMaxVisNodes; }
   void  SetMaxVisNodes(Int_t mvn) { fMaxVisNodes = mvn;  }

   Bool_t CanEditElement() const override { return kTRUE; }
   Bool_t SingleRnrState() const override { return kTRUE; }

   void   AddStamp(UChar_t bits) override;

   void Draw(Option_t* option="") override;
   void Paint(Option_t* option="") override;

   // Signals from GeoManager.
   // These are not available any more ... colors in list-tree not refreshed
   // properly.
   void VolumeVisChanged(TGeoVolume* volume);
   void VolumeColChanged(TGeoVolume* volume);
   void NodeVisChanged(TGeoNode* node);

   ClassDefOverride(TEveGeoTopNode, 0); // Top-level TEveGeoNode with a pointer to TGeoManager and controls for steering of TGeoPainter.
};

#endif
