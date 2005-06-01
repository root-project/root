// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.h,v 1.25 2005/05/25 14:25:16 brun Exp $
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
#ifndef ROOT_TGLLogicalShape
#include "TGLLogicalShape.h"
#endif

class TBuffer3D;
class TBuffer3DSphere;
class TBuffer3DTube;

/////////////////////////////////////////////////////////////
class TGLSceneObject : public TGLLogicalShape 
{
protected:
   std::vector<Double_t> fVertices;

private:
   TObject               *fRealObject;

public:
   TGLSceneObject(const TBuffer3D &buffer, Int_t verticesReserve, 
                  TObject *realObj = 0);
   TGLSceneObject(const TBuffer3D &buffer, TObject *realObj = 0);
	TGLSceneObject(TObject *realObj);
   
   void InvokeContextMenu(TContextMenu &menu, UInt_t x, UInt_t y) const;

   //virtual void Shift(Double_t x, Double_t y, Double_t z);
   //virtual void Stretch(Double_t xs, Double_t ys, Double_t zs);

private:
   TGLSceneObject(const TGLSceneObject &);
   TGLSceneObject & operator = (const TGLSceneObject &);

   ClassDef(TGLSceneObject,0)
};

///////////////////////////////////////////////////////////////////////
class TGLFaceSet : public TGLSceneObject {
private:
   std::vector<Double_t> fNormals;
   std::vector<Int_t>    fPolyDesc;
   UInt_t                fNbPols;

protected:
   void DirectDraw(UInt_t LOD) const;  
   
public:
   TGLFaceSet(const TBuffer3D &buff, TObject *realObj);
   void SetFromMesh(const RootCsg::BaseMesh *m);
   //void Stretch(Double_t xs, Double_t ys, Double_t zs);
   void DrawWireFrame(UInt_t) const;
   void DrawOutline(UInt_t) const;

private:
   void GLDrawPolys()const;
   Int_t CheckPoints(const Int_t *source, Int_t *dest)const;
   static Bool_t Eq(const Double_t *p1, const Double_t *p2);
   void CalculateNormals();
};

////////////////////////////////////////////////////////////////////////
class TGLPolyMarker : public TGLSceneObject {
private:
   UInt_t fStyle;

protected:
   void DirectDraw(UInt_t LOD) const;  
   
public:
   TGLPolyMarker(const TBuffer3D &buff, TObject *realObject);

private:
   void DrawStars()const;
};


class TGLPolyLine : public TGLSceneObject {
protected:
   void DirectDraw(UInt_t LOD) const;  
   
public:
   TGLPolyLine(const TBuffer3D &buff, TObject *realObject);

};

// Utility class to draw a Sphere using OpenGL Sphere primitive
class TGLSphere : public TGLSceneObject {
private:
   Double_t fRadius; // Sphere radius

protected:
   void DirectDraw(UInt_t LOD) const;  

public:
   TGLSphere(const TBuffer3DSphere &buffer, TObject *realObject);

   // void Shift(Double_t x, Double_t y, Double_t z);
   // void Stretch(Double_t xs, Double_t ys, Double_t zs);
};

class TGLMesh;

class TGLCylinder : public TGLSceneObject {
private:
   std::vector<TGLMesh *> fParts;

protected:
   void DirectDraw(UInt_t LOD) const;  

public:
   TGLCylinder(const TBuffer3DTube &buff, TObject *realObject);
   ~TGLCylinder();

   //void Shift(Double_t x, Double_t y, Double_t z);
   //void Stretch(Double_t xs, Double_t ys, Double_t zs);

private:
   void CreateParts(const TBuffer3DTube & buffer);
};

#endif
