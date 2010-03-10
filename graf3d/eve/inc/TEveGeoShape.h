// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGeoShape
#define ROOT_TEveGeoShape

#include "TEveElement.h"
#include "TEveProjectionBases.h"
#include "TAttBBox.h"

class TGeoShape;
class TEveGeoShapeExtract;
class TBuffer3D;

class TEveGeoShape : public TEveElement,
                     public TNamed,
                     public TEveProjectable
{
private:
   TEveGeoShape(const TEveGeoShape&);            // Not implemented
   TEveGeoShape& operator=(const TEveGeoShape&); // Not implemented

protected:
   Color_t           fColor;
   Int_t             fNSegments;
   TGeoShape*        fShape;

   static TGeoManager* fgGeoMangeur;

   static TEveGeoShape* SubImportShapeExtract(TEveGeoShapeExtract* gse, TEveElement* parent);
   TEveGeoShapeExtract* DumpShapeTree(TEveGeoShape* geon, TEveGeoShapeExtract* parent=0);

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
   void SaveExtract(const char* file, const char* name);
   void WriteExtract(const char* name);

   static TEveGeoShape* ImportShapeExtract(TEveGeoShapeExtract* gse, TEveElement* parent=0);

   // GeoProjectable
   virtual TBuffer3D*   MakeBuffer3D();
   virtual TClass*      ProjectedClass(const TEveProjection* p) const;

   static TGeoManager*  GetGeoMangeur();

   ClassDef(TEveGeoShape, 1); // Wrapper for TGeoShape with absolute positioning and color attributes allowing display of extracted TGeoShape's (without an active TGeoManager) and simplified geometries (needed for NLT projections).
};

//------------------------------------------------------------------------------

class TEveGeoShapeProjected : public TEveElementList,
                              public TEveProjected,
                              public TAttBBox
{
private:
   TEveGeoShapeProjected(const TEveGeoShapeProjected&);            // Not implemented
   TEveGeoShapeProjected& operator=(const TEveGeoShapeProjected&); // Not implemented

protected:
   TBuffer3D*  fBuff;

   virtual void SetDepthLocal(Float_t d);

public:
   TEveGeoShapeProjected();
   virtual ~TEveGeoShapeProjected() {}

   virtual Bool_t  CanEditMainTransparency() const { return kTRUE; }

   virtual void SetProjection(TEveProjectionManager* proj, TEveProjectable* model);
   virtual void UpdateProjection();

   virtual void ComputeBBox();
   virtual void Paint(Option_t* option = "");

   ClassDef(TEveGeoShapeProjected, 0);
};

#endif
