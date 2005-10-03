// @(#)root/base:$Name:  $:$Id: TVirtualGL.h,v 1.21 2005/09/02 07:51:51 brun Exp $
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

class TVirtualViewer3D;
class TGLSceneObject;
class TPoints3DABC;
class TGLViewer;
class TGLRect;
class TGLWindow;

class TVirtualGLImp {

public:
   virtual ~TVirtualGLImp() { }
   virtual Window_t CreateGLWindow(Window_t wind) = 0;
   virtual ULong_t  CreateContext(Window_t wind) = 0;
   virtual void     DeleteContext(ULong_t ctx) = 0;
   virtual void     MakeCurrent(Window_t wind, ULong_t ctx) = 0;
   virtual void     SwapBuffers(Window_t wind) = 0;

   ClassDef(TVirtualGLImp,0);
};



class TVirtualGL : public TNamed {

protected:
   TVirtualGLImp *fImp;

public:
   TVirtualGL(TVirtualGLImp *imp = 0);
   TVirtualGL(const char *name);
   virtual ~TVirtualGL() { delete fImp; }

   // system specific GL methods
   virtual Window_t CreateGLWindow(Window_t wind) { return fImp ? fImp->CreateGLWindow(wind) : 0; }
   virtual ULong_t  CreateContext(Window_t wind) { return fImp ? fImp->CreateContext(wind) : 0; }
   virtual void     DeleteContext(ULong_t ctx) { if (fImp) fImp->DeleteContext(ctx); }
   virtual void     MakeCurrent(Window_t wind, ULong_t ctx) { if (fImp) fImp->MakeCurrent(wind, ctx); }
   virtual void     SwapBuffers(Window_t wind) { if (fImp) fImp->SwapBuffers(wind); }

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
   virtual void GLLight(EG3D2GLmode name, EG3D2GLmode prop_name, const Float_t * lig_prop) = 0;
   virtual void LightModel(EG3D2GLmode name, const Float_t * lig_prop) = 0;
   virtual void LightModel(EG3D2GLmode name, Int_t prop) = 0;
   virtual void CullFaceGL(EG3D2GLmode) = 0;
   virtual void ViewportGL(Int_t xmin, Int_t ymin, Int_t width, Int_t height) = 0;
   virtual void MaterialGL(EG3D2GLmode face, const Float_t * mat_prop) = 0;
   virtual void MaterialGL(EG3D2GLmode face, Float_t mat_prop) = 0;
   virtual void BeginGL(EG3D2GLmode) = 0;
   virtual void EndGL() = 0;
   virtual void SetGLVertex(const Double_t *vert) = 0;
   virtual void SetGLVertex(Float_t *vertex) = 0;
   virtual void SetGLNormal(const Double_t *norm) = 0;
   virtual void PaintPolyMarker(const Double_t * place, Style_t marker_style, UInt_t size) = 0;
   virtual void DrawSelectionBox(Double_t xmin, Double_t xmax,
                                 Double_t ymin, Double_t ymax,
                                 Double_t zmin, Double_t zmax) = 0;
   virtual void EnterSelectionMode(UInt_t * buff, Int_t size, Event_t *, Int_t * viewport) = 0;
   virtual Int_t ExitSelectionMode() = 0;
   virtual void GLLoadName(UInt_t name) = 0;
   virtual void DrawFaceSet(const Double_t * pnts, const Int_t * pols,
                            const Double_t * normals, const Float_t * mat,
                            UInt_t size) = 0;
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
   virtual void PaintPolyLine(Int_t n, Double_t *p, Option_t *option) = 0;
   virtual void PaintGLPointsObject(const TPoints3DABC *points, Option_t *option="") = 0;
   virtual void PaintBrik(Float_t vertex[24])  = 0;
   virtual void PaintXtru(Float_t *vertex, Int_t nxy, Int_t nz) = 0;
   virtual void SetLineAttr(Color_t color, Int_t width=1) = 0;
   virtual void UpdateMatrix(Double_t *translate=0, Double_t *rotate=0, Bool_t isreflection=kFALSE)  = 0;
   virtual Bool_t GetRootLight() = 0;
   virtual void SetRootLight(Bool_t flag=kTRUE)  = 0;
   virtual Bool_t GetTrueColorMode() = 0;
   virtual void SetTrueColorMode(Bool_t flag=kTRUE) = 0;
   virtual void DrawSphere(const Float_t *rgba) = 0;
   virtual void DrawViewer(TGLViewer *viewer) = 0;
   virtual Bool_t SelectViewer(TGLViewer *viewer, const TGLRect * rect) = 0;
   virtual void SelectViewerManip(TGLViewer *viewer, const TGLRect * rect) = 0;
   virtual void CaptureViewer(TGLViewer *viewer, Int_t format, const char * filePath = 0) = 0;

   static TVirtualGL *&Instance();

   ClassDef(TVirtualGL,0);
};

#ifndef __CINT__
#define gVirtualGL (TVirtualGL::Instance())
R__EXTERN TVirtualGL *(*gPtr2VirtualGL)();
#endif

//This class (and its descendants) in future will replace (?) 
//TVirtualGL/TGLKernel/TGWin32GL/TGX11GL

class TGLManager : public TNamed {
public:
   TGLManager();

   //index returned can be used as a result of gVirtualX->InitWindow
   //isOffScreen important only for GLX, not for WGL
   virtual Int_t    InitGLWindow(Window_t winID, Bool_t isOffScreen = kFALSE) = 0;
   //virtual void     CloseGLWindow(Int_t winInd) = 0;
   //double-buffered on-screen rendering
   virtual Int_t    CreateGLContext(Int_t winInd) = 0;
   //off-screen rendering into pixmap (DIB section)
   //this pixmap cannot be used directly with gVirtualX
   virtual Int_t    OpenGLPixmap(Int_t winInd, Int_t x, Int_t y, UInt_t w, UInt_t h) = 0;
   virtual void     ResizeGLPixmap(Int_t pixInd, Int_t x, Int_t y, UInt_t w, UInt_t h) = 0;

   //instead of gVirtualX->SelectWindow(fPixmapID) => gVirtualGL->SelectGLPixmap(fPixmapID)
   //after that gVirtualX can draw into this pixmap
   virtual void     SelectGLPixmap(Int_t pixInd) = 0;
   //GLX pixmap or DIB can be used directly by TVirtualX,
   //obtain correct index first
   virtual Int_t    GetVirtualXInd(Int_t pixInd) = 0;
   //copy pixmap into window directly, without gVirtualX (which copies into buffer first
   // and requires gVirtualX->UpdateWindow)
   virtual void     MarkForDirectCopy(Int_t pixInd, Bool_t) = 0;
   //can be used with gl context and gl pixmaps
   virtual Bool_t   MakeCurrent(Int_t devInd) = 0;
   //can be used with gl context and gl pixmap and context
   virtual void     Flush(Int_t devInd, Int_t x = 0, Int_t y = 0) = 0;
   virtual void     DeletePaintDevice(Int_t devInd) = 0;
   //viewport extracted only for pixmap
   virtual void     ExtractViewport(Int_t devInd, Int_t *vp) = 0;
   
   //functions to switch between threads in win32
   //used by viewer
   virtual void     DrawViewer(TVirtualViewer3D *vv) = 0;
   virtual TObject* Select(TVirtualViewer3D *vv, Int_t x, Int_t y) = 0;

   static TGLManager *&Instance();

private:
   //compiler can't generate implicit copy ctor 
   //for descendants
   TGLManager(const TGLManager &);
   TGLManager &operator = (const TGLManager &);

   ClassDef(TGLManager, 0)
};

#ifndef __CINT__
#define gGLManager (TGLManager::Instance())
R__EXTERN TGLManager *(*gPtr2GLManager)();
#endif

#endif
