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
#include "TEveTrans.h"
#include "TEveProjectionBases.h"

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
   TEveGeoShapeExtract* DumpShapeTree(TEveGeoNode* geon, TEveGeoShapeExtract* parent = 0, Int_t level = 0);
public:
   TEveGeoNode(TGeoNode* node);

   virtual const Text_t* GetName()  const;
   virtual const Text_t* GetTitle() const;

   TGeoNode* GetNode() const { return fNode; }

   virtual Int_t ExpandIntoListTree(TGListTree* ltree, TGListTreeItem* parent);

   virtual Bool_t CanEditRnrElement() { return false; }
   virtual void SetRnrSelf(Bool_t rnr);
   virtual void SetRnrChildren(Bool_t rnr);
   virtual void SetRnrState(Bool_t rnr);

   virtual Bool_t CanEditMainColor()  { return true; }
   virtual void   SetMainColor(Color_t color);
   virtual void   SetMainColor(Pixel_t pixel);

   void UpdateNode(TGeoNode* node);
   void UpdateVolume(TGeoVolume* volume);

   void Save(const char* file, const char* name="Extract");

   virtual void Draw(Option_t* option="");

   ClassDef(TEveGeoNode, 1); // Wrapper for TGeoNode that allows it to be shown in GUI and controlled as a TEveElement.
};

//----------------------------------------------------------------

class TEveGeoTopNode : public TEveGeoNode
{
   TEveGeoTopNode(const TEveGeoTopNode&);            // Not implemented
   TEveGeoTopNode& operator=(const TEveGeoTopNode&); // Not implemented

protected:
   TGeoManager* fManager;
   TEveTrans       fGlobalTrans;
   Int_t        fVisOption;
   Int_t        fVisLevel;

public:
   TEveGeoTopNode(TGeoManager* manager, TGeoNode* node, Int_t visopt=1, Int_t vislvl=3);
   virtual ~TEveGeoTopNode();

   virtual Bool_t     CanEditMainHMTrans() { return  kTRUE; }
   virtual TEveTrans* PtrMainHMTrans()     { return &fGlobalTrans; }

   TEveTrans&   RefGlobalTrans() { return fGlobalTrans; }
   void         SetGlobalTrans(const TGeoHMatrix* m);
   void         UseNodeTrans();

   Int_t GetVisOption() const { return fVisOption; }
   void  SetVisOption(Int_t visopt);
   Int_t GetVisLevel()  const { return fVisLevel; }
   void  SetVisLevel(Int_t vislvl);

   virtual Bool_t CanEditRnrElement() { return true; }
   virtual void SetRnrSelf(Bool_t rnr);

   virtual void Draw(Option_t* option="");
   virtual void Paint(Option_t* option="");

   // Signals from GeoManager.
   // These are not available any more ... colors in list-tree not refreshed
   // properly.
   void VolumeVisChanged(TGeoVolume* volume);
   void VolumeColChanged(TGeoVolume* volume);
   void NodeVisChanged(TGeoNode* node);

   ClassDef(TEveGeoTopNode, 1); // Top-level TEveGeoNode with a pointer to TGeoManager and controls for steering of TGeoPainter.
};


//----------------------------------------------------------------
//----------------------------------------------------------------

class TEveGeoShape : public TEveElement,
                     public TNamed,
                     public TEveProjectable
{
   TEveGeoShape(const TEveGeoShape&);            // Not implemented
   TEveGeoShape& operator=(const TEveGeoShape&); // Not implemented

protected:
   TEveTrans         fHMTrans;
   Color_t           fColor;
   UChar_t           fTransparency;
   TGeoShape*        fShape;

   static TEveGeoShape* SubImportShapeExtract(TEveGeoShapeExtract* gse, TEveElement* parent);
   TEveGeoShapeExtract*     DumpShapeTree(TEveGeoShape* geon, TEveGeoShapeExtract* parent = 0);

public:
   TEveGeoShape(const Text_t* name="TEveGeoShape", const Text_t* title=0);
   virtual ~TEveGeoShape();

   virtual Bool_t CanEditMainColor() { return kTRUE; }

   virtual Bool_t  CanEditMainTransparency()      { return kTRUE; }
   virtual UChar_t GetMainTransparency() const    { return fTransparency; }
   virtual void    SetMainTransparency(UChar_t t) { fTransparency = t; }

   virtual Bool_t     CanEditMainHMTrans() { return  kTRUE; }
   virtual TEveTrans* PtrMainHMTrans()     { return &fHMTrans; }

   TEveTrans& RefHMTrans() { return fHMTrans; }
   void SetTransMatrix(Double_t* carr)        { fHMTrans.SetFrom(carr); }
   void SetTransMatrix(const TGeoMatrix& mat) { fHMTrans.SetFrom(mat);  }

   Color_t     GetColor()        { return fColor; }
   TGeoShape*  GetShape()        { return fShape; }

   virtual void Paint(Option_t* option="");

   void Save(const char* file, const char* name="Extract");
   static TEveGeoShape*        ImportShapeExtract(TEveGeoShapeExtract* gse, TEveElement* parent);

   // NLTGeoProjectable
   virtual TBuffer3D*           MakeBuffer3D();
   virtual TClass*              ProjectedClass() const;

   ClassDef(TEveGeoShape, 1); // Wrapper for TGeoShape with absolute positioning and color attributes allowing display of extracted TGeoShape's (without an active TGeoManager) and simplified geometries (needed for NLT projections).
};

#endif
