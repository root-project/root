// @(#)root/base:$Name:  $:$Id: TVirtualGL.h,v 1.3 2000/09/14 07:03:43 brun Exp $
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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_Gtypes
#include "Gtypes.h"
#endif

class TVirtualGL;
class TGLViewerImp;
class TPadOpenGLView;
class TVirtualPad;
class TPoints3DABC;
class TPadView3D;

R__EXTERN TVirtualGL *gVirtualGL;

#include "GLConstants.h"


class TVirtualGL {

protected:
   UInt_t       fColorIndx;     // Current color index;
   Bool_t       fRootLight;     // Whether the "ROOT" light will be used (otherwise OpenGL)
   Bool_t       fTrueColorMode; // Defines the whether the current hardware layer supports the true colors
   EG3D2GLmode  fFaceFlag;      // The current "face" definiton - clockwise/counterclockwise

   Float_t *Invert(Float_t *vector)
      { for (int i = 0; i < 3; i++) vector[i] = -vector[i]; return vector; }

public:
   TVirtualGL();
   virtual ~TVirtualGL() { }

   virtual void AddRotation(Double_t *rotmatrix, Double_t *extraangles);
   virtual void BeginGLCmd(EG3D2GLmode mode);
   virtual void ClearColor(Int_t color);
   virtual void ClearGLColor(Float_t red, Float_t green, Float_t blue, Float_t alpha);
   virtual void ClearGL(UInt_t bufbits );
   virtual Int_t CreateGLLists(Int_t range);
   virtual TGLViewerImp *CreateGLViewerImp(TPadOpenGLView *c, const char *title, UInt_t width, UInt_t height);
   virtual TPadView3D *CreatePadGLView(TVirtualPad *c);
   virtual void DeleteGLLists(Int_t ilist, Int_t range);
   virtual void DisableGL(EG3D2GLmode mode);
   virtual void EnableGL(EG3D2GLmode mode);
   virtual void EndGLList() { }
   virtual void EndGLCmd() { }
   virtual void FlushGL() { }
   virtual void FrontGLFace(EG3D2GLmode faceflag);
   virtual void GetGL(EG3D2GLmode mode, UChar_t *params);
   virtual void GetGL(EG3D2GLmode mode, Double_t *params);
   virtual void GetGL(EG3D2GLmode mode, Float_t *params);
   virtual void GetGL(EG3D2GLmode mode, Int_t *params);
   virtual Int_t GetGLError() {return 0;}
           Bool_t GetRootLight() {return fRootLight;}
           Bool_t GetTrueColorMode() {return fTrueColorMode;}
   virtual void MultGLMatrix(Double_t *mat);
   virtual void NewGLList(UInt_t ilist=1, EG3D2GLmode mode=kCOMPILE);
   virtual void NewGLModelView(Int_t ilist);
   virtual void PaintGLPoints(Int_t n, Float_t *p, Option_t *option);
   virtual void PolygonGLMode(EG3D2GLmode face, EG3D2GLmode mode);
   virtual void PushGLMatrix() { }
   virtual void PopGLMatrix() { }
   virtual void RotateGL(Double_t angle, Double_t x, Double_t y, Double_t z);
   virtual void RotateGL(Double_t Theta, Double_t Phi, Double_t Psi);
   virtual void SetGLVertex(Float_t *vertex);
   virtual void SetGLColor(Float_t *rgb);
   virtual void SetGLColorIndex(Int_t color);
   virtual void SetGLLineWidth(Float_t width);
   virtual void SetGLPointSize(Float_t size);
   virtual void SetStack(Double_t *matrix=0);
   virtual void ShadeGLModel(EG3D2GLmode mode);

   virtual void TranslateGL(Double_t x, Double_t y, Double_t z);

   virtual void RunGLList(Int_t list);
   virtual void NewProjectionView(Double_t viewboxmin[], Double_t viewboxmax[], Bool_t perspective=kTRUE);
   virtual void NewModelView(Double_t *angles, Double_t *delta );
   virtual void PaintCone(Float_t *vertex, Int_t ndiv, Int_t nstacks);
   virtual void PaintPolyLine(Int_t n, Float_t *p, Option_t *option);
   virtual void PaintGLPointsObject(const TPoints3DABC *points, Option_t *option="");
   virtual void PaintBrik(Float_t vertex[24]);
   virtual void PaintXtru(Float_t *vertex, Int_t nxy, Int_t nz);

   virtual void SetRootLight(Bool_t flag=kTRUE) { fRootLight = flag; }
   virtual void SetLineAttr(Color_t color, Int_t width=1);
   virtual void UpdateMatrix(Double_t *translate=0, Double_t *rotate=0, Bool_t isreflection=kFALSE);

   void SetTrueColorMode(Bool_t flag=kTRUE) { fTrueColorMode = flag; }
};

//--- inlines ------------------------------------------------------------------
inline void TVirtualGL::AddRotation(Double_t *,Double_t *) { }
inline void TVirtualGL::BeginGLCmd(EG3D2GLmode) { }  // Begins the vertices of a primitive or a group of like primitives.
inline void TVirtualGL::ClearColor(Int_t) { }
inline void TVirtualGL::ClearGLColor(Float_t, Float_t, Float_t, Float_t) { }
inline void TVirtualGL::ClearGL(UInt_t) { }
inline Int_t TVirtualGL::CreateGLLists(Int_t) { return 0; }
inline TGLViewerImp *TVirtualGL::CreateGLViewerImp(TPadOpenGLView *, const char *, UInt_t, UInt_t) { return 0; }
inline TPadView3D *TVirtualGL::CreatePadGLView(TVirtualPad *) { return 0; }
inline void TVirtualGL::DeleteGLLists(Int_t, Int_t) { }
inline void TVirtualGL::DisableGL(EG3D2GLmode) { }
inline void TVirtualGL::EnableGL(EG3D2GLmode) { }
inline void TVirtualGL::FrontGLFace(EG3D2GLmode) { }
inline void TVirtualGL::GetGL(EG3D2GLmode, UChar_t *) { }
inline void TVirtualGL::GetGL(EG3D2GLmode, Double_t *) { }
inline void TVirtualGL::GetGL(EG3D2GLmode, Float_t *) { }
inline void TVirtualGL::GetGL(EG3D2GLmode, Int_t *) { }
inline void TVirtualGL::MultGLMatrix(Double_t *) { }
inline void TVirtualGL::NewGLList(UInt_t,EG3D2GLmode) { }
inline void TVirtualGL::NewGLModelView(Int_t) { }
inline void TVirtualGL::PaintGLPoints(Int_t, Float_t *, Option_t *) { }
inline void TVirtualGL::PolygonGLMode(EG3D2GLmode , EG3D2GLmode) { }
inline void TVirtualGL::RotateGL(Double_t, Double_t, Double_t, Double_t) { }
inline void TVirtualGL::RotateGL(Double_t, Double_t, Double_t) { }
inline void TVirtualGL::SetGLVertex(Float_t *) { }
inline void TVirtualGL::SetGLColor(Float_t *) { }
inline void TVirtualGL::SetGLColorIndex(Int_t) { }
inline void TVirtualGL::SetGLLineWidth(Float_t) { }
inline void TVirtualGL::SetGLPointSize(Float_t) { }
inline void TVirtualGL::SetStack(Double_t *) { }
inline void TVirtualGL::ShadeGLModel(EG3D2GLmode) { }
inline void TVirtualGL::TranslateGL(Double_t, Double_t, Double_t) { }
inline void TVirtualGL::RunGLList(Int_t) { }
inline void TVirtualGL::NewProjectionView(Double_t [], Double_t [], Bool_t) { }
inline void TVirtualGL::NewModelView(Double_t *, Double_t *) { }
inline void TVirtualGL::PaintCone(Float_t *, Int_t, Int_t) { }
inline void TVirtualGL::PaintPolyLine(Int_t, Float_t *, Option_t *) { }
inline void TVirtualGL::PaintGLPointsObject(const TPoints3DABC *, Option_t *) { }
inline void TVirtualGL::PaintBrik(Float_t [24]) { }
inline void TVirtualGL::PaintXtru(Float_t *, Int_t, Int_t) { }
inline void TVirtualGL::SetLineAttr(Color_t, Int_t) { }
inline void TVirtualGL::UpdateMatrix(Double_t *, Double_t *, Bool_t) { }

#endif
