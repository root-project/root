// @(#)root/gl:$Name:  $:$Id: TWin32GLKernel.h,v 1.1.1.1 2000/05/16 17:00:47 rdm Exp $
// Author: Valery Fine(fine@vxcern.cern.ch)   05/03/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-* T G L K e r n e l ABC class *-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ============================
//*-*
//*-*   TGLKernel class defines the interface for OpenGL command and utilities
//*-*   Those are defined with GL/gl and GL/glu include directories
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#ifndef ROOT_TWin32GLKernel
#define ROOT_TWin32GLKernel

#ifndef ROOT_TGLKernel
#include "TGLKernel.h"
#endif

#ifndef ROOT_TWin32HookViaThread
#include "TWin32HookViaThread.h"
#endif
typedef enum EGLCallbackCmd {
                              kNewGLModelView, kEndGLList        ,kBeginGLCmd
                             ,kGetGLError    , kEndGLCmd         ,kPushGLMatrix   ,kPopGLMatrix
                             ,kTranslateGL   , kMultGLMatrix     ,kSetGLColorIndex
                             ,kSetGLLineWidth, kSetGLVertex      ,kDeleteGLLists  ,kCreateGLLists
                             ,kRunGLList     , kNewProjectionView,kPaintPolyLine  ,kPaintBrik
                             ,kClearGLColor  , kClearGL          ,kFlushGL        ,kNewGLList
                             ,kSetGLColor    , kPaintCone        ,kDisableGL      ,kEnableGL
                             ,kRotateGL      , kFrontGLFace      ,kPaintGLPoints  ,kSetGLPointSize
                             ,kClearColor    , kNewModelView     ,kPolygonGLMode  ,kGetGL
                             ,kShadeGLModel  , kSetRootLight     ,kSetLineAttr    ,kUpdateMatrix
                             ,kAddRotation   , kSetStack         ,kPaintGLPointsObject
                            } ;

class TWin32GLKernel : protected TWin32HookViaThread, public TGLKernel {

protected:

   virtual void ExecThreadCB(TWin32SendClass *command);


public:

   TWin32GLKernel();
   virtual ~TWin32GLKernel();
   virtual void NewGLModelView(Int_t ilist);
   virtual void EndGLList();
   virtual void AddRotation(Double_t *rotmatrix,Double_t *extraangles);
   virtual void BeginGLCmd(EG3D2GLmode mode);
   virtual void ClearColor(Int_t color);
   virtual void ClearGLColor(Float_t red, Float_t green, Float_t blue, Float_t alpha);
   virtual void ClearGL(UInt_t bufbits );
   virtual TGLViewerImp *CreateGLViewerImp(TPadOpenGLView *c, const char *title, UInt_t width, UInt_t height);
   virtual void DisableGL(EG3D2GLmode mode);
   virtual void EnableGL(EG3D2GLmode mode);
   virtual void FlushGL();
   virtual void FrontGLFace(EG3D2GLmode faceflag);
   virtual void GetGL(EG3D2GLmode mode, Bool_t *params);
   virtual void GetGL(EG3D2GLmode mode, Double_t *params);
   virtual void GetGL(EG3D2GLmode mode, Float_t *params);
   virtual void GetGL(EG3D2GLmode mode, Int_t *params);
   virtual Int_t GetGLError();
   virtual void EndGLCmd();
   virtual void PaintGLPoints(Int_t n, Float_t *p, Option_t *option);
   virtual void PaintGLPointsObject(const TPoints3DABC *points, Option_t *option="");
   virtual void PolygonGLMode(EG3D2GLmode face , EG3D2GLmode mode);
   virtual void PushGLMatrix();
   virtual void PopGLMatrix();
   virtual void RotateGL(Double_t angle, Double_t x,Double_t y,Double_t z);
   virtual void RotateGL(Double_t Theta, Double_t Phi, Double_t Psi);
   virtual void TranslateGL(Double_t x,Double_t y,Double_t z);
   virtual void MultGLMatrix(Double_t *mat);
   virtual void NewGLList(UInt_t ilist=1,EG3D2GLmode mode=kCOMPILE);
   virtual void SetGLColor(Float_t *rgb);
   virtual void SetGLColorIndex(Int_t color);
   virtual void SetGLLineWidth(Float_t width);
   virtual void SetGLPointSize(Float_t size);
   virtual void SetGLVertex(Float_t *vertex);
   virtual void SetStack(Double_t *matrix=0);
   virtual void SetRootLight(Bool_t flag=kTRUE);
   virtual void ShadeGLModel(EG3D2GLmode mode);

   virtual void DeleteGLLists(Int_t ilist, Int_t range);
   virtual Int_t CreateGLLists(Int_t range);
   virtual void RunGLList(Int_t list);
   virtual void NewProjectionView(Double_t viewboxmin[],Double_t viewboxmax[],Bool_t perspective=kTRUE);
   virtual void NewModelView(Double_t *angles,Double_t *delta );
   virtual void PaintCone(Float_t *vertex,Int_t ndiv,Int_t nstacks);
   virtual void PaintPolyLine(Int_t n, Float_t *p, Option_t *option);
   virtual void PaintBrik(Float_t vertex[24]);
   virtual void PaintXtru(Float_t *vertex, Int_t nxy, Int_t nz);
   virtual void SetLineAttr(Color_t color, Int_t width);
   virtual void UpdateMatrix(Double_t *translate=0, Double_t *rotate=0, Bool_t isreflection=kFALSE);

};

#endif
