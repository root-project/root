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

   virtual const char* GetName()  const;
   virtual const char* GetTitle() const;
   virtual const char* GetElementName()  const;
   virtual const char* GetElementTitle() const;

   TGeoNode* GetNode() const { return fNode; }

   virtual void   ExpandIntoListTree(TGListTree* ltree, TGListTreeItem* parent);

   virtual Bool_t CanEditElement() const { return kFALSE; }

   virtual void   AddStamp(UChar_t bits);

   virtual Bool_t CanEditMainColor() const;
   virtual void   SetMainColor(Color_t color);

   virtual Bool_t  CanEditMainTransparency() const;
   virtual UChar_t GetMainTransparency() const;
   virtual void    SetMainTransparency(UChar_t t);

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
   Int_t        fVisOption;
   Int_t        fVisLevel;
   Int_t        fMaxVisNodes;

public:
   TEveGeoTopNode(TGeoManager* manager, TGeoNode* node, Int_t visopt=1,
                  Int_t vislvl=3, Int_t maxvisnds=10000);
   virtual ~TEveGeoTopNode() {}

   void         UseNodeTrans();

   Int_t GetVisOption()      const { return fVisOption; }
   void  SetVisOption(Int_t vo)    { fVisOption = vo;   }
   Int_t GetVisLevel()       const { return fVisLevel;  }
   void  SetVisLevel(Int_t vl)     { fVisLevel = vl;    }
   Int_t GetMaxVisNodes()    const { return fMaxVisNodes; }
   void  SetMaxVisNodes(Int_t mvn) { fMaxVisNodes = mvn;  }

   virtual Bool_t CanEditElement() const { return kTRUE; }
   virtual Bool_t SingleRnrState() const { return kTRUE; }

   virtual void   AddStamp(UChar_t bits);

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
   Color_t           fColor;
   Int_t             fNSegments;
   TGeoShape*        fShape;

   static TGeoManager* fgGeoMangeur;

   static TEveGeoShape* SubImportShapeExtract(TEveGeoShapeExtract* gse, TEveElement* parent);
   TEveGeoShapeExtract* DumpShapeTree(TEveGeoShape* geon, TEveGeoShapeExtract* parent = 0);

public:
   TEveGeoShape(const char* name="TEveGeoShape", const char* title=0);
   virtual ~TEveGeoShape();

   virtual Bool_t  CanEditMainColor()        const { return kTRUE; }
   virtual Bool_t  CanEditMainTransparency() const { return kTRUE; }

   Color_t     GetColor()      const { return fColor; }
   Int_t       GetNSegments()  const { return fNSegments; }
   void        SetNSegments(Int_t s) { fNSegments = s; }
   TGeoShape*  GetShape()            { return fShape; }
   void        SetShape(TGeoShape* s);

   virtual void Paint(Option_t* option="");

   void Save(const char* file, const char* name="Extract");
   static TEveGeoShape* ImportShapeExtract(TEveGeoShapeExtract* gse, TEveElement* parent=0);

   // GeoProjectable
   virtual TBuffer3D*   MakeBuffer3D();
   virtual TClass*      ProjectedClass() const;

   static TGeoManager*  GetGeoMangeur();

   ClassDef(TEveGeoShape, 1); // Wrapper for TGeoShape with absolute positioning and color attributes allowing display of extracted TGeoShape's (without an active TGeoManager) and simplified geometries (needed for NLT projections).
};

#endif
