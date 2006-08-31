// @(#)root/win32gdk:$Name:  $:$Id: TGWin32VirtualGLProxy.cxx,v 1.21 2006/06/06 11:49:01 couet Exp $
// Author: Valeriy Onuchin   05/08/04

/*************************************************************************
 * Copyright (C) 1995-2004,Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGWin32ProxyDefs.h"
#include "TGWin32VirtualGLProxy.h"
#include "TGWin32.h"
#include "TROOT.h"

static TVirtualGL *gKernelGL = 0;

//____________________________________________________________________________
TGWin32VirtualGLProxy::TGWin32VirtualGLProxy()
{
   //

   fMaxResponseTime = 15000;
}

//____________________________________________________________________________
TVirtualGL *TGWin32VirtualGLProxy::RealObject()
{
   //

   if (!gKernelGL) {
      gKernelGL = (TVirtualGL *)gROOT->GetListOfSpecials()->FindObject("gVirtualGL");
   }

   return gKernelGL;
}

RETURN_PROXY_OBJECT(VirtualGL)
RETURN_METHOD_ARG1(VirtualGL,Window_t,CreateGLWindow,Window_t,wind)
RETURN_METHOD_ARG1(VirtualGL,ULong_t,CreateContext,Window_t,wind)
VOID_METHOD_ARG1(VirtualGL,DeleteContext,ULong_t,ctx,1)
VOID_METHOD_ARG2(VirtualGL,MakeCurrent,Window_t,wind,ULong_t,ctx,1)
VOID_METHOD_ARG1(VirtualGL,SwapBuffers,Window_t,wind,1)
VOID_METHOD_ARG1(VirtualGL,NewGLModelView,Int_t,ilist,1)
VOID_METHOD_ARG2(VirtualGL,AddRotation,Double_t*,rotmatrix,Double_t*,extraangles,1)
VOID_METHOD_ARG1(VirtualGL,ClearColor,Int_t,colors,1)
VOID_METHOD_ARG4(VirtualGL,ClearGLColor,Float_t,red,Float_t,green,Float_t,blue,Float_t,alpha,1)
VOID_METHOD_ARG1(VirtualGL,ClearGL,UInt_t,bufbits,1)
RETURN_METHOD_ARG1(VirtualGL,Int_t,CreateGLLists,Int_t,range)
VOID_METHOD_ARG2(VirtualGL,DeleteGLLists,Int_t,ilist,Int_t,range,1)
VOID_METHOD_ARG0(VirtualGL,EndGLList,1)
VOID_METHOD_ARG1(VirtualGL,BeginGLCmd,EG3D2GLmode,mode,1)
VOID_METHOD_ARG1(VirtualGL,DisableGL,EG3D2GLmode,mode,1)
VOID_METHOD_ARG1(VirtualGL,EnableGL,EG3D2GLmode,mode,1)
VOID_METHOD_ARG2(VirtualGL,GetGL,EG3D2GLmode,modeu,UChar_t*,params,1)
VOID_METHOD_ARG2(VirtualGL,GetGL,EG3D2GLmode,moded,Double_t*,params,1)
VOID_METHOD_ARG2(VirtualGL,GetGL,EG3D2GLmode,modef,Float_t*,params,1)
VOID_METHOD_ARG2(VirtualGL,GetGL,EG3D2GLmode,modei,Int_t*,params,1)
RETURN_METHOD_ARG0(VirtualGL,Int_t,GetGLError)
VOID_METHOD_ARG0(VirtualGL,EndGLCmd,1)
VOID_METHOD_ARG0(VirtualGL,FlushGL,1)
VOID_METHOD_ARG1(VirtualGL,FrontGLFace,EG3D2GLmode,faceflag,1)
VOID_METHOD_ARG1(VirtualGL,MultGLMatrix,Double_t*,mat,1)
VOID_METHOD_ARG2(VirtualGL,NewGLList,UInt_t,ilist,EG3D2GLmode,mode,1)
VOID_METHOD_ARG2(VirtualGL,NewModelView,Double_t*,angles,Double_t*,delta,1)
VOID_METHOD_ARG3(VirtualGL,NewProjectionView,Double_t*,viewboxmin,Double_t*,viewboxmax,Bool_t,perspective,1)
VOID_METHOD_ARG3(VirtualGL,PaintCone,Float_t*,vertex,Int_t,ndiv,Int_t,nstacks,1)
VOID_METHOD_ARG3(VirtualGL,PaintPolyLine,Int_t,n,Float_t*,p,Option_t*,option,1)
VOID_METHOD_ARG3(VirtualGL,PaintPolyLine,Int_t,nd,Double_t*,pd,Option_t*,option,1)
VOID_METHOD_ARG2(VirtualGL,PaintGLPointsObject,const TPoints3DABC*,points,Option_t*,option,1)
VOID_METHOD_ARG1(VirtualGL,PaintBrik,Float_t*,vertex,1)
VOID_METHOD_ARG3(VirtualGL,PaintXtru,Float_t*,vertex,Int_t,nxy,Int_t,nz,1)
VOID_METHOD_ARG2(VirtualGL,PolygonGLMode,EG3D2GLmode,face,EG3D2GLmode,mode,1)
VOID_METHOD_ARG0(VirtualGL,PopGLMatrix,1)
VOID_METHOD_ARG0(VirtualGL,PushGLMatrix,1)
VOID_METHOD_ARG4(VirtualGL,RotateGL,Double_t,angle,Double_t,x,Double_t,y,Double_t,z,1)
VOID_METHOD_ARG3(VirtualGL,RotateGL,Double_t,Theta,Double_t,Phi,Double_t,Psi,1)
VOID_METHOD_ARG1(VirtualGL,RunGLList,Int_t,list,1)
VOID_METHOD_ARG3(VirtualGL,TranslateGL,Double_t,x,Double_t,y,Double_t,z,1)
VOID_METHOD_ARG1(VirtualGL,SetGLColor,Float_t*,rgb,1)
VOID_METHOD_ARG1(VirtualGL,SetGLColorIndex,Int_t,color,1)
VOID_METHOD_ARG1(VirtualGL,SetGLLineWidth,Float_t,width,1)
VOID_METHOD_ARG1(VirtualGL,SetGLPointSize,Float_t,size,1)
VOID_METHOD_ARG1(VirtualGL,SetStack,Double_t*,matrix,1)
VOID_METHOD_ARG1(VirtualGL,SetRootLight,Bool_t,flag,1)
VOID_METHOD_ARG1(VirtualGL,ShadeGLModel,EG3D2GLmode,mode,1)
VOID_METHOD_ARG2(VirtualGL,SetLineAttr,Color_t,color,Int_t,width,1)
VOID_METHOD_ARG3(VirtualGL,UpdateMatrix,Double_t*,translate,Double_t *,rotate,Bool_t,reflection,1)
VOID_METHOD_ARG1(VirtualGL,ClearGLDepth,Float_t,val,1)
VOID_METHOD_ARG1(VirtualGL,MatrixModeGL,EG3D2GLmode,mode,1)
VOID_METHOD_ARG0(VirtualGL,NewMVGL,1)
VOID_METHOD_ARG0(VirtualGL,NewPRGL,1)
VOID_METHOD_ARG6(VirtualGL,FrustumGL,Double_t,xmin,Double_t,xmax,Double_t,ymin,Double_t,ymax,Double_t,znear,Double_t,zfar,1)
VOID_METHOD_ARG3(VirtualGL,GLLight,EG3D2GLmode,name,EG3D2GLmode, prop, const Float_t*,lig_prop,1)
VOID_METHOD_ARG2(VirtualGL,LightModel,EG3D2GLmode,name,const Float_t*,lig_prop,1)
VOID_METHOD_ARG2(VirtualGL,LightModel,EG3D2GLmode,names,Int_t,lig_prop,1)
VOID_METHOD_ARG1(VirtualGL,CullFaceGL,EG3D2GLmode,mode,1)
VOID_METHOD_ARG4(VirtualGL,ViewportGL,Int_t,x,Int_t,y,Int_t,w,Int_t,h,1)
VOID_METHOD_ARG2(VirtualGL,MaterialGL,EG3D2GLmode,face,const Float_t*,mat_prop,1)
VOID_METHOD_ARG2(VirtualGL,MaterialGL,EG3D2GLmode,faces,Float_t,mat_prop,1)
VOID_METHOD_ARG1(VirtualGL,BeginGL, EG3D2GLmode, mode, 1)
VOID_METHOD_ARG0(VirtualGL,EndGL,1)
VOID_METHOD_ARG1(VirtualGL,SetGLVertex,const Double_t*,vertexd,1)
VOID_METHOD_ARG1(VirtualGL,SetGLVertex,Float_t*,vertexf,1)
VOID_METHOD_ARG1(VirtualGL,SetGLNormal,const Double_t*,normal,1)
VOID_METHOD_ARG1(VirtualGL,SetTrueColorMode,Bool_t,flag,1);
VOID_METHOD_ARG3(VirtualGL,PaintGLPoints,Int_t,n,Float_t*,p,Option_t*,option,1)
RETURN_METHOD_ARG0(VirtualGL,Bool_t,GetTrueColorMode)
RETURN_METHOD_ARG0(VirtualGL,Bool_t,GetRootLight)
VOID_METHOD_ARG3(VirtualGL, PaintPolyMarker, const Double_t *, v, Style_t, s, UInt_t, sz, 1)
VOID_METHOD_ARG6(VirtualGL, DrawSelectionBox, Double_t, xmin, Double_t, xmax, Double_t, ymin, Double_t, ymax, Double_t, zmin, Double_t, zmax, 1)
RETURN_METHOD_ARG0(VirtualGL, Int_t, ExitSelectionMode)
VOID_METHOD_ARG4(VirtualGL, EnterSelectionMode, UInt_t *, buff, Int_t, size, Event_t *, ev, Int_t *, viewport, 1)
VOID_METHOD_ARG1(VirtualGL, GLLoadName, UInt_t, name, 1)
VOID_METHOD_ARG5(VirtualGL, DrawFaceSet, const Double_t *, pnts, const Int_t *, pols, const Double_t *, normals, const Float_t *, mat, UInt_t, size, 1)
VOID_METHOD_ARG1(VirtualGL, DrawSphere, const Float_t *, color, 1)
VOID_METHOD_ARG1(VirtualGL, DrawViewer, TGLViewer *, viewer, 1)
RETURN_METHOD_ARG2(VirtualGL, Bool_t, SelectViewer, TGLViewer *, viewer, const TGLRect *, rect)
RETURN_METHOD_ARG4(VirtualGL, Bool_t, SelectManip, TGLManip *, manip, const TGLCamera *, camera, const TGLRect *, rect, const TGLBoundingBox *, sceneBox)
VOID_METHOD_ARG3(VirtualGL, CaptureViewer, TGLViewer *, viewer, Int_t, format, const char *, filePath, 1)

static TGLManager *gManager = 0;

//____________________________________________________________________________
TGLManager *TGWin32GLManagerProxy::RealObject()
{
   if (!gManager) {
      gManager = (TGLManager *)gROOT->GetListOfSpecials()->FindObject("gGLManager");
   }

   return gManager;
}

TGWin32GLManagerProxy::TGWin32GLManagerProxy()
{
}

RETURN_PROXY_OBJECT(GLManager)
RETURN_METHOD_ARG1(GLManager, Int_t, InitGLWindow, Window_t, winID)
RETURN_METHOD_ARG1(GLManager, Int_t, CreateGLContext, Int_t, winInd)
RETURN_METHOD_ARG5(GLManager, Bool_t, AttachOffScreenDevice, Int_t, winInd, Int_t, x, Int_t, y, UInt_t, w, UInt_t, h)
RETURN_METHOD_ARG5(GLManager, Bool_t, ResizeOffScreenDevice, Int_t, pixInd, Int_t, x, Int_t, y, UInt_t, w, UInt_t, h)
VOID_METHOD_ARG1(GLManager, SelectOffScreenDevice, Int_t, pixInd, 1)
RETURN_METHOD_ARG1(GLManager, Int_t, GetVirtualXInd, Int_t, pixInd)
VOID_METHOD_ARG2(GLManager, MarkForDirectCopy, Int_t, pixInd, Bool_t, direct, 1)
RETURN_METHOD_ARG1(GLManager, Bool_t, MakeCurrent, Int_t, devInd)
VOID_METHOD_ARG1(GLManager, Flush, Int_t, devInd, 1)
VOID_METHOD_ARG1(GLManager, ReadGLBuffer, Int_t, devInd, 1)
VOID_METHOD_ARG1(GLManager, DeleteGLContext, Int_t, devInd, 1)
VOID_METHOD_ARG2(GLManager, ExtractViewport, Int_t, pixInd, Int_t *, vp, 1)
VOID_METHOD_ARG1(GLManager, DrawViewer, TVirtualViewer3D *, vv, 1)
VOID_METHOD_ARG1(GLManager, PrintViewer, TVirtualViewer3D *, vv, 1)
VOID_METHOD_ARG1(GLManager, PaintSingleObject, TVirtualGLPainter *, p, 1)
VOID_METHOD_ARG3(GLManager, PanObject, TVirtualGLPainter *, p, Int_t, x, Int_t, y, 1)
RETURN_METHOD_ARG2(GLManager, Bool_t, SelectViewer, TVirtualViewer3D *, viewer, const TGLRect *, rect)
RETURN_METHOD_ARG3(GLManager, Bool_t, PlotSelected, TVirtualGLPainter *, plot, Int_t, x, Int_t, y)
RETURN_METHOD_ARG3(GLManager, char *, GetPlotInfo, TVirtualGLPainter *, plot, Int_t, x, Int_t, y)
RETURN_METHOD_ARG4(GLManager, Bool_t, SelectManip, TVirtualGLManip *, manip, const TGLCamera *, camera, const TGLRect *, rect, const TGLBoundingBox *, box)
