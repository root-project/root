// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.h,v 1.31 2006/01/18 16:57:58 brun Exp $
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
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
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

private:
   TGLSceneObject(const TGLSceneObject &);
   TGLSceneObject & operator = (const TGLSceneObject &);

   ClassDef(TGLSceneObject,0) // abstract scene object logical
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
   void SetFromMesh(const RootCsg::TBaseMesh *m);

   virtual ELODAxes SupportedLODAxes() const { return kLODAxesNone; }
   void   DrawWireFrame(UInt_t) const;
   void   DrawOutline(UInt_t) const;

private:
   void GLDrawPolys()const;
   Int_t CheckPoints(const Int_t *source, Int_t *dest)const;
   static Bool_t Eq(const Double_t *p1, const Double_t *p2);
   void CalculateNormals();

   ClassDef(TGLFaceSet,0) // a faceset logical shape
};

////////////////////////////////////////////////////////////////////////
class TGLPolyMarker : public TGLSceneObject {
private:
   UInt_t fStyle;

protected:
   void DirectDraw(UInt_t LOD) const;  
   
public:
   TGLPolyMarker(const TBuffer3D &buff, TObject *realObject);

   virtual ELODAxes SupportedLODAxes() const { return kLODAxesNone; }

private:
   void DrawStars()const;

   ClassDef(TGLPolyMarker,0) // a polymarker logical shape
};


class TGLPolyLine : public TGLSceneObject {
protected:
   void DirectDraw(UInt_t LOD) const;  
   
public:
   TGLPolyLine(const TBuffer3D &buff, TObject *realObject);

   virtual ELODAxes SupportedLODAxes() const { return kLODAxesNone; }
   ClassDef(TGLPolyLine,0) // a polyline logical shape
};

// Utility class to draw a Sphere using OpenGL Sphere primitive
class TGLSphere : public TGLSceneObject {
private:
   Double_t fRadius; // Sphere radius

protected:
   void DirectDraw(UInt_t LOD) const;  

public:
   TGLSphere(const TBuffer3DSphere &buffer, TObject *realObject);

   virtual ELODAxes SupportedLODAxes() const { return kLODAxesAll; }
   ClassDef(TGLSphere,0) // a spherical logical shape
};

class TGLMesh;

class TGLCylinder : public TGLSceneObject {
private:
   Double_t fR1, fR2, fR3, fR4;
   Double_t fDz;
   Double_t fPhi1, fPhi2;

   TGLVector3 fLowPlaneNorm, fHighPlaneNorm;
   Bool_t fSegMesh;

protected:
   void DirectDraw(UInt_t LOD) const;  

public:
   TGLCylinder(const TBuffer3DTube &buff, TObject *realObject);
   ~TGLCylinder();

   // Cylinders support LOD (tesselation quality) adjustment along
   // X/Y axes (round the cylinder radius), but not along length (Z)
   virtual ELODAxes SupportedLODAxes() const { return ELODAxes(kLODAxesX | kLODAxesY); }

private:
   ClassDef(TGLCylinder,0) // a cylinderical logical shape
};

#endif
