// @(#)root/win32gdk:$Name:  $:$Id: TGWin32VirtualGLProxy.cxx,v 1.1 2004/08/09 15:46:53 brun Exp $
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

   fMaxResponseTime = 1000;
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
VOID_METHOD_ARG1(VirtualGL,SwapLayerBuffers,Window_t,wind,1)
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
VOID_METHOD_ARG2(VirtualGL,GLLight,EG3D2GLmode,name,const Float_t*,lig_prop,1)
VOID_METHOD_ARG2(VirtualGL,LightModel,EG3D2GLmode,name,const Float_t*,lig_prop,1)
VOID_METHOD_ARG2(VirtualGL,LightModel,EG3D2GLmode,names,Int_t,lig_prop,1)
VOID_METHOD_ARG1(VirtualGL,CullFaceGL,EG3D2GLmode,mode,1)
VOID_METHOD_ARG4(VirtualGL,ViewportGL,Int_t,x,Int_t,y,Int_t,w,Int_t,h,1)
VOID_METHOD_ARG2(VirtualGL,MaterialGL,EG3D2GLmode,face,const Float_t*,mat_prop,1)
VOID_METHOD_ARG2(VirtualGL,MaterialGL,EG3D2GLmode,faces,Float_t,mat_prop,1)
VOID_METHOD_ARG0(VirtualGL,BeginGL,1)
VOID_METHOD_ARG0(VirtualGL,EndGL,1)
VOID_METHOD_ARG1(VirtualGL,SetGLVertex,const Double_t*,vertexd,1)
VOID_METHOD_ARG1(VirtualGL,SetGLVertex,Float_t*,vertexf,1)
VOID_METHOD_ARG1(VirtualGL,SetGLNormal,const Double_t*,normal,1)
RETURN_METHOD_ARG0(VirtualGL,GLUtesselator*,GLUNewTess)
VOID_METHOD_ARG1(VirtualGL,GLUDeleteTess,GLUtesselator *,t_obj,1)
VOID_METHOD_ARG1(VirtualGL,GLUTessCallback,GLUtesselator *,t_obj,1)
VOID_METHOD_ARG1(VirtualGL,GLUNextContour,GLUtesselator *,t_obj,1)
VOID_METHOD_ARG1(VirtualGL,GLUBeginPolygon,GLUtesselator *,t_obj,1)
VOID_METHOD_ARG1(VirtualGL,GLUEndPolygon,GLUtesselator *,t_obj,1)
VOID_METHOD_ARG2(VirtualGL,GLUTessVertex,GLUtesselator *,t_obj,const Double_t *,vertex,1)
VOID_METHOD_ARG1(VirtualGL,SetTrueColorMode,Bool_t,flag,1);
VOID_METHOD_ARG3(VirtualGL,PaintGLPoints,Int_t,n,Float_t*,p,Option_t*,option,1)
RETURN_METHOD_ARG0(VirtualGL,Bool_t,GetTrueColorMode)
RETURN_METHOD_ARG0(VirtualGL,Bool_t,GetRootLight)
