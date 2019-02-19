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

#include "TEveShape.h"

class TGeoShape;
class TGeoHMatrix;
class TGeoCompositeShape;
class TEveGeoShapeExtract;
class TBuffer3D;

class TEveGeoShape : public TEveShape
{
private:
   TEveGeoShape(const TEveGeoShape&);            // Not implemented
   TEveGeoShape& operator=(const TEveGeoShape&); // Not implemented

protected:
   Int_t               fNSegments;
   TGeoShape          *fShape;
   TGeoCompositeShape *fCompositeShape; //! Temporary holder (if passed shape is composite shape).

   static TGeoManager *fgGeoMangeur;

   static TEveGeoShape* SubImportShapeExtract(TEveGeoShapeExtract* gse, TEveElement* parent);
   TEveGeoShapeExtract* DumpShapeTree(TEveGeoShape* geon, TEveGeoShapeExtract* parent=0);

   TGeoShape* MakePolyShape();

public:
   TEveGeoShape(const char* name="TEveGeoShape", const char* title=0);
   virtual ~TEveGeoShape();

   virtual TObject* GetObject(const TEveException&) const
   { const TObject* obj = this; return const_cast<TObject*>(obj); }

   Int_t       GetNSegments()  const { return fNSegments; }
   TGeoShape*  GetShape()      const { return fShape;     }
   void        SetNSegments(Int_t s);
   void        SetShape(TGeoShape* s);

   virtual void ComputeBBox();
   virtual void Paint(Option_t* option="");

   void Save(const char* file, const char* name="Extract");
   void SaveExtract(const char* file, const char* name);
   void WriteExtract(const char* name);

   static TEveGeoShape* ImportShapeExtract(TEveGeoShapeExtract* gse, TEveElement* parent=0);

   // GeoProjectable
   virtual TBuffer3D*   MakeBuffer3D();
   virtual TClass*      ProjectedClass(const TEveProjection* p) const;

   static TGeoManager*  GetGeoMangeur();
   static TGeoHMatrix*  GetGeoHMatrixIdentity();

   ClassDef(TEveGeoShape, 0); // Wrapper for TGeoShape with absolute positioning and color attributes allowing display of extracted TGeoShape's (without an active TGeoManager) and simplified geometries (needed for NLT projections).
};

//------------------------------------------------------------------------------

class TEveGeoShapeProjected : public TEveShape,
                              public TEveProjected
{
private:
   TEveGeoShapeProjected(const TEveGeoShapeProjected&);            // Not implemented
   TEveGeoShapeProjected& operator=(const TEveGeoShapeProjected&); // Not implemented

protected:
   TBuffer3D   *fBuff;

   virtual void SetDepthLocal(Float_t d);

public:
   TEveGeoShapeProjected();
   virtual ~TEveGeoShapeProjected();

   virtual void SetProjection(TEveProjectionManager* proj, TEveProjectable* model);
   virtual void UpdateProjection();
   virtual TEveElement* GetProjectedAsElement() { return this; }

   virtual void ComputeBBox();

   ClassDef(TEveGeoShapeProjected, 0);
};

#endif
