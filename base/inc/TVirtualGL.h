// @(#)root/base:$Name:  $:$Id: TVirtualGL.h,v 1.34 2006/10/24 14:20:41 brun Exp $
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
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif
#ifndef ROOT_GLConstants
#include "GLConstants.h"
#endif

class TVirtualViewer3D;
class TPoints3DABC;
class TGLViewer;
class TGLCamera;
class TGLManip;
class TGLBoundingBox;
class TGLRect;

class TVirtualGLImp {

public:
   virtual ~TVirtualGLImp() { }
   virtual Window_t CreateGLWindow(Window_t wind) = 0;
   virtual ULong_t  CreateContext(Window_t wind) = 0;
   virtual void     DeleteContext(ULong_t ctx) = 0;
   virtual void     MakeCurrent(Window_t wind, ULong_t ctx) = 0;
   virtual void     SwapBuffers(Window_t wind) = 0;

   ClassDef(TVirtualGLImp,0);  // Interface for a GL concrete Implementation
};



class TVirtualGL : public TNamed {

protected:
   TVirtualGLImp *fImp;

   TVirtualGL(const TVirtualGL& vgl) 
     : TNamed(vgl), fImp(vgl.fImp) { }
   TVirtualGL& operator=(const TVirtualGL& vgl) 
     {if(this!=&vgl) {TNamed::operator=(vgl); fImp=vgl.fImp;} 
     return *this;}

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
   virtual Bool_t SelectManip(TGLManip *manip, const TGLCamera * camera, const TGLRect * rect, const TGLBoundingBox * sceneBox) = 0;
   virtual void CaptureViewer(TGLViewer *viewer, Int_t format, const char * filePath = 0) = 0;

   static TVirtualGL *&Instance();

   ClassDef(TVirtualGL,0);  // Interface for OpenGL 
};

#ifndef __CINT__
#define gVirtualGL (TVirtualGL::Instance())
R__EXTERN TVirtualGL *(*gPtr2VirtualGL)();
#endif

//TVirtualGLPainter is the base for histogramm painters.

class TVirtualGLPainter {
public:
   virtual ~TVirtualGLPainter(){}

   virtual void     Paint() = 0;
   virtual void     Pan(Int_t px, Int_t py) = 0;
   virtual Bool_t   PlotSelected(Int_t px, Int_t py) = 0;
   //Used by status bar in a canvas.
   virtual char    *GetPlotInfo(Int_t px, Int_t py) = 0;

   ClassDef(TVirtualGLPainter, 0); // Interface for OpenGL painter
};

//We need this class to implement TGWin32GLManager's SelectManip
class TVirtualGLManip {
public:
   virtual ~TVirtualGLManip(){}
   virtual Bool_t Select(const TGLCamera & camera, const TGLRect & rect, const TGLBoundingBox & sceneBox) = 0;
      
   ClassDef(TVirtualGLManip, 0); //Interface for GL manipulator
};

//This class (and its descendants) in future will replace (?)
//TVirtualGL/TGLKernel/TGWin32GL/TGX11GL

class TGLManager : public TNamed {
public:
   TGLManager();

   //index returned can be used as a result of gVirtualX->InitWindow
   virtual Int_t    InitGLWindow(Window_t winID) = 0;
   //winInd is the index, returned by InitGLWindow
   virtual Int_t    CreateGLContext(Int_t winInd) = 0;

   //[            Off-screen rendering part
   //create DIB section/pixmap to read GL buffer into it, 
   //ctxInd is the index, returned by CreateGLContext
   virtual Bool_t   AttachOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h) = 0;
   virtual Bool_t   ResizeOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h) = 0;
   //analog of gVirtualX->SelectWindow(fPixmapID) => gVirtualGL->SelectOffScreenDevice(fPixmapID)
   virtual void     SelectOffScreenDevice(Int_t ctxInd) = 0;
   //Index of DIB/pixmap, valid for gVirtualX
   virtual Int_t    GetVirtualXInd(Int_t ctxInd) = 0;
   //copy pixmap into window directly
   virtual void     MarkForDirectCopy(Int_t ctxInd, Bool_t) = 0;
   //Off-screen device holds sizes for glViewport
   virtual void     ExtractViewport(Int_t ctxInd, Int_t *vp) = 0;
   //Read GL buffer into off-screen device
   virtual void     ReadGLBuffer(Int_t ctxInd) = 0;
   //]            
   
   //Make the gl context current
   virtual Bool_t   MakeCurrent(Int_t ctxInd) = 0;
   //Swap buffers or copies DIB/pixmap (via BitBlt/XCopyArea)
   virtual void     Flush(Int_t ctxInd) = 0;
   //GL context and off-screen device deletion
   virtual void     DeleteGLContext(Int_t ctxInd) = 0;

   //functions to switch between threads in win32
   virtual void     DrawViewer(TVirtualViewer3D *vv) = 0;
   //
   virtual Bool_t   SelectViewer(TVirtualViewer3D *viewer, const TGLRect *selRect) = 0;
   virtual Bool_t   SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox) = 0;
   //
   virtual void     PaintSingleObject(TVirtualGLPainter *) = 0;
   virtual void     PanObject(TVirtualGLPainter *o, Int_t x, Int_t y) = 0;
   //EPS/PDF output
   virtual void     PrintViewer(TVirtualViewer3D *vv) = 0;

   virtual Bool_t   PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py) = 0;
   virtual char    *GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py) = 0;

   virtual Bool_t   HighColorFormat(Int_t ctxInd) = 0;

   static TGLManager *&Instance();

private:
   TGLManager(const TGLManager &);
   TGLManager &operator = (const TGLManager &);

   ClassDef(TGLManager, 0)// Interface for OpenGL manager
};

#ifndef __CINT__
#define gGLManager (TGLManager::Instance())
R__EXTERN TGLManager *(*gPtr2GLManager)();
#endif

#endif
