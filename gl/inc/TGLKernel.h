// @(#)root/base:$Name:  $:$Id: TGLKernel.h,v 1.23 2006/01/26 11:59:41 brun Exp $
// Author: Valery Fine(fine@vxcern.cern.ch)   05/03/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLKernel
#define ROOT_TGLKernel


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLKernel                                                            //
//                                                                      //
// The TGLKernel implementation of TVirtualGL class.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif

#ifdef WIN32
#include "Windows4root.h"
#endif

#include <GL/glu.h>

class TGLKernel : public TVirtualGL {

private:
   void SetCurrentColor(Int_t color);
   void LightIndex(Int_t i)
     { SetCurrentColor(fColorIndx + 201 + ((i == 0) ? 0 : TMath::Abs(i%7-3))); }

   Float_t *Invert(Float_t *vector) {
      for (int i = 0; i < 3; i++) vector[i] = -vector[i]; return vector; }

protected:
   UInt_t       fColorIndx;     // Current color index;
   Bool_t       fRootLight;     // Whether the "ROOT" light will be used (otherwise OpenGL)
   Bool_t       fTrueColorMode; // Defines the whether the current hardware layer supports the true colors
   EG3D2GLmode  fFaceFlag;      // The current "face" definiton - clockwise/counterclockwise
   GLUquadric * fQuad;
   GLUtriangulatorObj *fTessObj;
public:
   TGLKernel(TVirtualGLImp *imp = 0);
   TGLKernel(const char *name);
   virtual ~TGLKernel();

   void NewGLModelView(Int_t ilist);
   void AddRotation(Double_t *rotmatrix,Double_t *extraangles);
   void ClearColor(Int_t colors);
   void ClearGLColor(Float_t red, Float_t green, Float_t blue, Float_t alpha);
   void ClearGLColor(Float_t colors[4]);
   void ClearGL(UInt_t bufbits );
   Int_t CreateGLLists(Int_t range);
   void DeleteGLLists(Int_t ilist, Int_t range);
   void EndGLList();
   void BeginGLCmd(EG3D2GLmode mode);
   void DisableGL(EG3D2GLmode mode);
   void EnableGL(EG3D2GLmode mode);
   void GetGL(EG3D2GLmode mode, UChar_t *params);
   void GetGL(EG3D2GLmode mode, Double_t *params);
   void GetGL(EG3D2GLmode mode, Float_t *params);
   void GetGL(EG3D2GLmode mode, Int_t *params);
   Int_t GetGLError();
   void EndGLCmd();
   void FlushGL();
   void FrontGLFace(EG3D2GLmode faceflag);
   void MultGLMatrix(Double_t *mat);
   void NewGLList(UInt_t ilist,EG3D2GLmode mode);
   void NewModelView(Double_t *angles,Double_t *delta );
   void NewProjectionView(Double_t viewboxmin[],Double_t viewboxmax[],Bool_t perspective=kTRUE);
   void PaintGLPoints(Int_t n, Float_t *p, Option_t *option);
   void PaintCone(Float_t *vertex,Int_t ndiv,Int_t nstacks);
   void PaintPolyLine(Int_t n, Float_t *p, Option_t *option);
   void PaintPolyLine(Int_t n, Double_t *p, Option_t *option);
   void PaintGLPointsObject(const TPoints3DABC *points, Option_t *option="");
   void PaintBrik(Float_t vertex[24]);
   void PaintXtru(Float_t *vertex, Int_t nxy, Int_t nz);
   void PolygonGLMode(EG3D2GLmode face , EG3D2GLmode mode);
   void PopGLMatrix();
   void PushGLMatrix();
   void RotateGL(Double_t *direction, Int_t mode);
   void RotateGL(Double_t angle, Double_t x,Double_t y,Double_t z);
   void RotateGL(Double_t Theta, Double_t Phi, Double_t Psi);
   void RunGLList(Int_t list);
   void TranslateGL(Double_t x,Double_t y,Double_t z);
   void TranslateGL(Double_t *xyz);
   void SetGLColor(Float_t *rgb);
   void SetGLColorIndex(Int_t color);
   void SetGLLineWidth(Float_t width);
   void SetGLPointSize(Float_t size);
   void SetGLVertex(Float_t *vertex);
   void SetStack(Double_t *matrix=0);
   void SetRootLight(Bool_t flag=kTRUE);
   void ShadeGLModel(EG3D2GLmode mode);
   void SetLineAttr(Color_t color, Int_t width);
   void UpdateMatrix(Double_t *translate=0, Double_t *rotate=0, Bool_t isreflection=kFALSE);
   void ClearGLDepth(Float_t val);
   void MatrixModeGL(EG3D2GLmode mode);
   void NewMVGL();
   void NewPRGL();
   void FrustumGL(Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t znear, Double_t zfar);
   void GLLight(EG3D2GLmode name, EG3D2GLmode prop, const Float_t * lig_prop);
   void LightModel(EG3D2GLmode name, const Float_t * lig_prop);
   void LightModel(EG3D2GLmode name, Int_t prop);
   void CullFaceGL(EG3D2GLmode);
   void ViewportGL(Int_t x, Int_t y, Int_t width, Int_t height);
   void MaterialGL(EG3D2GLmode face, const Float_t * mat_prop);
   void MaterialGL(EG3D2GLmode face, Float_t mat_prop);
   void BeginGL(EG3D2GLmode);
   void EndGL();
   void SetGLVertex(const Double_t * vertex);
   void SetGLNormal(const Double_t * normal);
   void PaintPolyMarker(const Double_t * vertex, Style_t marker_style, UInt_t size);
   void DrawSelectionBox(Double_t xmin, Double_t xmax,
                         Double_t ymin, Double_t ymax,
                         Double_t zmin, Double_t zmax);
   void EnterSelectionMode(UInt_t * buff, Int_t size, Event_t *, Int_t * viewport);
   Int_t ExitSelectionMode();
   void GLLoadName(UInt_t name);
   void DrawFaceSet(const Double_t * pnts, const Int_t * pols,
                    const Double_t * normals, const Float_t * mat,
                    UInt_t size);
   void SetTrueColorMode(Bool_t flag=kTRUE) { fTrueColorMode = flag; }
   Bool_t GetRootLight() {return fRootLight;}
   Bool_t GetTrueColorMode() {return fTrueColorMode;}
   void DrawSphere(const Float_t *rgba);
   void DrawViewer(TGLViewer *viewer);
   Bool_t SelectViewer(TGLViewer *viewer, const TGLRect * rect);
   Bool_t SelectManip(TGLManip *manip, const TGLCamera * camera, const TGLRect * rect, const TGLBoundingBox * sceneBox);
   void CaptureViewer(TGLViewer *viewer, Int_t format, const char * filePath = 0);

private:
   void DrawStars(const Double_t * vertex, Style_t marker_style, UInt_t size);

   //rootcint cannot be run on this file because of the CINT version of GL.h
   //ClassDef(TGLKernel,0) //Concrete GL interface on top of TVirtualGL
};

#endif
