// @(#)root/base:$Name:  $:$Id: TVirtualGL.h,v 1.5 2004/08/03 16:01:17 brun Exp $
// Author: Valery Fine(fine@vxcern.cern.ch)   05/03/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualGL
#define ROOT_TVirtualGL


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGL                                                           //
//                                                                      //
// The TVirtualGL class is an abstract base class defining the          //
// OpenGL interface protocol. All interactions with OpenGL should be    //
// done via the global pointer gVirtualGL. If the OpenGL library is     //
// available this pointer is pointing to an instance of the TGLKernel   //
// class which provides the actual interface to OpenGL. Using this      //
// scheme of ABC we can use OpenGL in other parts of the framework      //
// without having to link with the OpenGL library in case we don't      //
// use the classes using OpenGL.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_Gtypes
#include "Gtypes.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif
#ifndef ROOT_GLConstants
#include "GLConstants.h"
#endif

class TPoints3DABC;
struct GLUtesselator;


class TVirtualGLimp {

public:
   virtual Window_t CreateGLWindow(Window_t wind) = 0;
   virtual ULong_t CreateContext(Window_t wind) = 0;
   virtual void DeleteContext(ULong_t ctx) = 0;
   virtual void MakeCurrent(Window_t wind, ULong_t ctx) = 0;
   virtual void SwapLayerBuffers(Window_t wind) = 0;

   ClassDef(TVirtualGLimp,0);
}; 



class TVirtualGL : public TNamed {

protected:
   TVirtualGLimp *fImp;

public:
   TVirtualGL(TVirtualGLimp *imp = 0);
   TVirtualGL(const char *name);
   virtual ~TVirtualGL() { delete fImp; }

   // system specific GL methods
   virtual Window_t CreateGLWindow(Window_t wind) { return fImp ? fImp->CreateGLWindow(wind) : 0; }
   virtual ULong_t CreateContext(Window_t wind) { return fImp ? fImp->CreateContext(wind) : 0; }
   virtual void DeleteContext(ULong_t ctx) {  if (fImp) fImp->DeleteContext(ctx); }
   virtual void MakeCurrent(Window_t wind, ULong_t ctx) { if (fImp) fImp->MakeCurrent(wind, ctx); }
   virtual void SwapLayerBuffers(Window_t wind) { if (fImp) fImp->SwapLayerBuffers(wind); }

   // common/kernel GL methods
   virtual void AddRotation(Double_t *rotmatrix, Double_t *extraangles) = 0;
   virtual void BeginGLCmd(EG3D2GLmode mode) = 0;
   virtual void ClearGL(UInt_t bufbits ) = 0;
   virtual void ClearColor(Int_t color) = 0;
   virtual void ClearGLColor(Float_t red, Float_t green, Float_t blue, Float_t alpha) = 0;
   virtual void ClearGLDepth(Float_t val) = 0;
   virtual void MatrixModeGL(EG3D2GLmode matrix) = 0;
   virtual void NewMVGL() = 0;
   virtual void NewPRGL() = 0;
   virtual void FrustumGL(Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t znear, Double_t zfar) = 0;
   virtual void GLLight(EG3D2GLmode name, const Float_t * lig_prop) = 0;
   virtual void LightModel(EG3D2GLmode name, const Float_t * lig_prop) = 0;
   virtual void LightModel(EG3D2GLmode name, Int_t prop) = 0;
   virtual void CullFaceGL(EG3D2GLmode) = 0;
   virtual void ViewportGL(Int_t xmin, Int_t ymin, Int_t width, Int_t height) = 0;
   virtual void MaterialGL(EG3D2GLmode face, const Float_t * mat_prop) = 0;
   virtual void MaterialGL(EG3D2GLmode face, Float_t mat_prop) = 0;
   virtual void BeginGL() = 0;
   virtual void EndGL() = 0;
   virtual void SetGLVertex(const Double_t *vert) = 0;
   virtual void SetGLVertex(Float_t *vertex) = 0;
   virtual void SetGLNormal(const Double_t *norm) = 0;
   virtual GLUtesselator *GLUNewTess() = 0;
   virtual void GLUDeleteTess(GLUtesselator *) = 0;
   virtual void GLUTessCallback(GLUtesselator *) = 0;
   virtual void GLUNextContour(GLUtesselator *) = 0;
   virtual void GLUBeginPolygon(GLUtesselator *) = 0;
   virtual void GLUEndPolygon(GLUtesselator *) = 0;
   virtual void GLUTessVertex(GLUtesselator *, const Double_t *) = 0;
   virtual Int_t CreateGLLists(Int_t range) = 0;
   virtual void DeleteGLLists(Int_t ilist, Int_t range) = 0;
   virtual void DisableGL(EG3D2GLmode mode) = 0;
   virtual void EnableGL(EG3D2GLmode mode) = 0;
   virtual void EndGLList() = 0;
   virtual void EndGLCmd() = 0;
   virtual void FlushGL() = 0;
   virtual void FrontGLFace(EG3D2GLmode faceflag) = 0;
   virtual void GetGL(EG3D2GLmode mode, UChar_t *params) = 0;
   virtual void GetGL(EG3D2GLmode mode, Double_t *params) = 0;
   virtual void GetGL(EG3D2GLmode mode, Float_t *params) = 0;
   virtual void GetGL(EG3D2GLmode mode, Int_t *params) = 0;
   virtual Int_t GetGLError() = 0;
   virtual void MultGLMatrix(Double_t *mat) = 0;
   virtual void NewGLList(UInt_t ilist=1, EG3D2GLmode mode=kCOMPILE) = 0;
   virtual void NewGLModelView(Int_t ilist) = 0;
   virtual void PaintGLPoints(Int_t n, Float_t *p, Option_t *option) = 0;
   virtual void PolygonGLMode(EG3D2GLmode face, EG3D2GLmode mode) = 0;
   virtual void PushGLMatrix() = 0;
   virtual void PopGLMatrix() = 0;
   virtual void RotateGL(Double_t angle, Double_t x, Double_t y, Double_t z) = 0;
   virtual void RotateGL(Double_t Theta, Double_t Phi, Double_t Psi) = 0;
   virtual void SetGLColor(Float_t *rgb) = 0;
   virtual void SetGLColorIndex(Int_t color) = 0;
   virtual void SetGLLineWidth(Float_t width) = 0;
   virtual void SetGLPointSize(Float_t size) = 0;
   virtual void SetStack(Double_t *matrix=0) = 0;
   virtual void ShadeGLModel(EG3D2GLmode mode) = 0;
   virtual void TranslateGL(Double_t x, Double_t y, Double_t z) = 0;
   virtual void RunGLList(Int_t list) = 0;
   virtual void NewProjectionView(Double_t viewboxmin[], Double_t viewboxmax[], Bool_t perspective=kTRUE) = 0;
   virtual void NewModelView(Double_t *angles, Double_t *delta ) = 0;
   virtual void PaintCone(Float_t *vertex, Int_t ndiv, Int_t nstacks) = 0;
   virtual void PaintPolyLine(Int_t n, Float_t *p, Option_t *option) = 0;
   virtual void PaintGLPointsObject(const TPoints3DABC *points, Option_t *option="") = 0;
   virtual void PaintBrik(Float_t vertex[24])  = 0;
   virtual void PaintXtru(Float_t *vertex, Int_t nxy, Int_t nz) = 0;
   virtual void SetLineAttr(Color_t color, Int_t width=1) = 0;
   virtual void UpdateMatrix(Double_t *translate=0, Double_t *rotate=0, Bool_t isreflection=kFALSE)  = 0;
   virtual Bool_t GetRootLight() = 0;
   virtual void SetRootLight(Bool_t flag=kTRUE)  = 0;
   virtual Bool_t GetTrueColorMode() = 0;
   virtual void SetTrueColorMode(Bool_t flag=kTRUE) = 0;

   static TVirtualGL *&Instance();

   ClassDef(TVirtualGL,0);
};

#ifndef __CINT__
#define gVirtualGL (TVirtualGL::Instance())
R__EXTERN TVirtualGL *(*gPtr2VirtualGL)();
#endif


#endif
