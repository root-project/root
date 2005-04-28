// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.h,v 1.23 2005/04/01 13:53:18 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLSceneObject
#define ROOT_TGLSceneObject

#include <utility>

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_Gtypes
#include "Gtypes.h"
#endif
#ifndef ROOT_CsgOps
#include "CsgOps.h"
#endif

class TGLFrustum;
class TBuffer3D;
class TBuffer3DSphere;
class TBuffer3DTube;

/////////////////////////////////////////////////////////////
class TGLSelection {
private:
   Double_t fBBox[6];

public:
   TGLSelection();
   TGLSelection(const Double_t *bbox);
   TGLSelection(Double_t xmin, Double_t xmax, Double_t ymin,
                Double_t ymax, Double_t zmin, Double_t zmax);

   void DrawBox()const;

   void SetBBox(const Double_t *newBbox);
   void SetBBox(Double_t xmin, Double_t xmax, Double_t ymin,
               Double_t ymax, Double_t zmin, Double_t zmax);
   const Double_t *GetData()const{return fBBox;}


   void Shift(Double_t x, Double_t y, Double_t z);
   void Stretch(Double_t xs, Double_t ys, Double_t zs);

   const Double_t *GetRangeX()const{return fBBox;}
   const Double_t *GetRangeY()const{return fBBox + 2;}
   const Double_t *GetRangeZ()const{return fBBox + 4;}
};

/////////////////////////////////////////////////////////////
class TGLSceneObject : public TObject {
protected:
   std::vector<Double_t> fVertices;
   Float_t               fColor[17];
   TGLSelection          fSelectionBox;
   Bool_t                fIsSelected;

private:
   UInt_t                fGLName;
   TGLSceneObject        *fNextT;
   TObject               *fRealObject;

public:
   TGLSceneObject(const TBuffer3D &buffer, Int_t verticesReserve, 
                  const Float_t *color = 0, UInt_t glName = 0, TObject *realObj = 0);
   TGLSceneObject(const TBuffer3D &buffer,
                  const Float_t *color = 0, UInt_t glName = 0, TObject *realObj = 0);
	TGLSceneObject(UInt_t glName, const Float_t *color, Short_t trans, TObject *realObj);

   virtual Bool_t IsTransparent()const;

   virtual void GLDraw(const TGLFrustum *fr)const = 0;

   virtual void Shift(Double_t x, Double_t y, Double_t z);
   virtual void Stretch(Double_t xs, Double_t ys, Double_t zs);

   TGLSelection *GetBBox(){return &fSelectionBox;}
   const TGLSelection *GetBBox()const{return &fSelectionBox;}

   void SetNextT(TGLSceneObject *next){fNextT = next;}
   TGLSceneObject *GetNextT()const{return fNextT;}

   UInt_t GetGLName()const{return fGLName;}
   TObject *GetRealObject()const{return fRealObject;}

   const Float_t *GetColor()const{return fColor;}
   void SetColor(const Float_t *newColor, Bool_t fromCtor = kFALSE);

   void Select(Bool_t select = kTRUE){fIsSelected = select;}

   void SetBBox();
private:
   TGLSceneObject(const TGLSceneObject &);
   TGLSceneObject & operator = (const TGLSceneObject &);

   void SetBBox(const TBuffer3D & buffer);

   ClassDef(TGLSceneObject,0)
};

///////////////////////////////////////////////////////////////////////
class TGLFaceSet : public TGLSceneObject {
private:
   std::vector<Double_t> fNormals;
   std::vector<Int_t>    fPolyDesc;
   UInt_t                fNbPols;

public:
   TGLFaceSet(const TBuffer3D &buff, const Float_t *color,
              UInt_t glName, TObject *realObj);
   TGLFaceSet(const RootCsg::BaseMesh *m, const Float_t *c, Short_t trans, UInt_t n, TObject *r);

   void GLDraw(const TGLFrustum *fr)const;
   void GLDrawPolys()const;
   void Stretch(Double_t xs, Double_t ys, Double_t zs);

private:
   Int_t CheckPoints(const Int_t *source, Int_t *dest)const;
   static Bool_t Eq(const Double_t *p1, const Double_t *p2);
   void CalculateNormals();
};

////////////////////////////////////////////////////////////////////////
class TGLPolyMarker : public TGLSceneObject {
private:
   UInt_t fStyle;

public:
   TGLPolyMarker(const TBuffer3D &buff, const Float_t *color, UInt_t glName, TObject *realObject);
   void GLDraw(const TGLFrustum *fr)const;

private:
   void DrawStars()const;
};


class TGLPolyLine : public TGLSceneObject {
public:
   TGLPolyLine(const TBuffer3D &buff, const Float_t *color, UInt_t glName, TObject *realObject);
   void GLDraw(const TGLFrustum *fr)const;
};

// Utility class to draw a Sphere using OpenGL Sphere primitive
class TGLSphere : public TGLSceneObject {
private:
   Double_t fX;      // Sphere X center position
   Double_t fY;      // Sphere Y center position
   Double_t fZ;      // Sphere Z center position
   Double_t fRadius; // Sphere radius
   Int_t    fNdiv;   // Number of divisions

   static void BuildList();

public:
   TGLSphere(const TBuffer3DSphere &buffer, const Float_t *color, UInt_t glName, TObject *realObject);

   void GLDraw(const TGLFrustum *fr)const;

   void Shift(Double_t x, Double_t y, Double_t z);
   void Stretch(Double_t xs, Double_t ys, Double_t zs);

   static UInt_t fSphereList;
};

class TGLMesh;

class TGLCylinder : public TGLSceneObject {
private:
   std::vector<TGLMesh *> fParts;
   Bool_t   fInv;

public:
   TGLCylinder(const TBuffer3DTube &buff, const Float_t *color,
               UInt_t glName, TObject *realObject);
   ~TGLCylinder();

   void GLDraw(const TGLFrustum *fr)const;

   void Shift(Double_t x, Double_t y, Double_t z);
   void Stretch(Double_t xs, Double_t ys, Double_t zs);

private:
   void CreateParts(const TBuffer3DTube & buffer);
};

#endif
