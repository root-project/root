// @(#)root/win32gdk:$Name:  $:$Id: TGWin32Proxy.cxx,v 1.7 2003/08/20 14:14:22 brun Exp $
// Author: Valeriy Onuchin  08/08/2003


/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32Proxy                                                         //
//                                                                      //
// This class is the proxy interface to the Win32 graphics system.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGWin32ProxyDefs.h"
#include "TGWin32Proxy.h"
#include "TGWin32.h"


////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGWin32Proxy::~TGWin32Proxy()
{
   // dtor

}

// canvas graphics
VOID_METHOD_ARG0_LOCK(TGWin32,SetFillAttributes)
VOID_METHOD_ARG0_LOCK(TGWin32,SetMarkerAttributes)
VOID_METHOD_ARG0_LOCK(TGWin32,SetLineAttributes)
VOID_METHOD_ARG0_LOCK(TGWin32,SetTextAttributes)
VOID_METHOD_ARG1_LOCK(TGWin32,ResetAttMarker,Option_t*,toption)
VOID_METHOD_ARG1_LOCK(TGWin32,ResetAttFill,Option_t*,option)
VOID_METHOD_ARG1_LOCK(TGWin32,ResetAttLine,Option_t*,option)
VOID_METHOD_ARG1_LOCK(TGWin32,ResetAttText,Option_t*,option)
VOID_METHOD_ARG1_LOCK(TGWin32,SetFillStyle,Style_t,style)
VOID_METHOD_ARG1_LOCK(TGWin32,SetTextAngle,Float_t,tangle)
VOID_METHOD_ARG1_LOCK(TGWin32,SetTextMagnitude,Float_t,mgn)
VOID_METHOD_ARG1_LOCK(TGWin32,SetTextSizePixels,Int_t,npixels)
VOID_METHOD_ARG1_LOCK(TGWin32,SetFillColor,Color_t,cindex)
VOID_METHOD_ARG1_LOCK(TGWin32,SetMarkerSize,Float_t,markersize)
VOID_METHOD_ARG1_LOCK(TGWin32,SetMarkerStyle,Style_t,markerstyle)
VOID_METHOD_ARG1_LOCK(TGWin32,SetMarkerColor,Color_t,cindex)
VOID_METHOD_ARG1(TGWin32,SetLineColor,Color_t,cindex,0)
VOID_METHOD_ARG1(TGWin32,SetLineStyle,Style_t,linestyle,0)
VOID_METHOD_ARG1(TGWin32,SetLineWidth,Width_t,width,0)
VOID_METHOD_ARG1(TGWin32,SetTextAlign,Short_t,talign,0)
VOID_METHOD_ARG1(TGWin32,SetTextSize,Float_t,textsize,1)
VOID_METHOD_ARG1(TGWin32,SetTextColor,Color_t,cindex,1)
VOID_METHOD_ARG1(TGWin32,SetTextFont,Font_t,fontnumber,1)
VOID_METHOD_ARG1(TGWin32,SelectWindow,Int_t,wid,0)
VOID_METHOD_ARG2(TGWin32,DrawFillArea,Int_t,n,TPoint*,xy,1)
VOID_METHOD_ARG2(TGWin32,DrawPolyLine,Int_t,n,TPoint*,xy,1)
VOID_METHOD_ARG2(TGWin32,DrawPolyMarker,Int_t,n,TPoint*,xy,1)
VOID_METHOD_ARG1(TGWin32,UpdateWindow,Int_t,mode,1)
VOID_METHOD_ARG1(TGWin32,SetOpacity,Int_t,percent,1)
VOID_METHOD_ARG5(TGWin32,DrawBox,Int_t,x1,Int_t,y1,Int_t,x2,Int_t,y2,TVirtualX::EBoxMode,mode,1)
VOID_METHOD_ARG6(TGWin32,DrawText,Int_t,x,Int_t,y,Float_t,angle,Float_t,mgn,const char*,text,TVirtualX::ETextMode,mode,1)
VOID_METHOD_ARG1(TGWin32,Update,Int_t,mode,1)
VOID_METHOD_ARG4(TGWin32,DrawLine,Int_t,x1,Int_t,y1,Int_t,x2,Int_t,y2,0)
VOID_METHOD_ARG2(TGWin32,GetCharacterUp,Float_t&,chupx,Float_t&,chupy,1)
RETURN_METHOD_ARG0(TGWin32,Float_t,GetTextMagnitude)
RETURN_METHOD_ARG0_CONST(TGWin32,Color_t,GetFillColor)
RETURN_METHOD_ARG0_CONST(TGWin32,Style_t,GetFillStyle)
RETURN_METHOD_ARG0_CONST(TGWin32,Bool_t,IsTransparent)
RETURN_METHOD_ARG0_CONST(TGWin32,Color_t,GetLineColor)
RETURN_METHOD_ARG0_CONST(TGWin32,Style_t,GetLineStyle)
RETURN_METHOD_ARG0_CONST(TGWin32,Width_t,GetLineWidth)
RETURN_METHOD_ARG0_CONST(TGWin32,Color_t,GetMarkerColor)
RETURN_METHOD_ARG0_CONST(TGWin32,Style_t,GetMarkerStyle)
RETURN_METHOD_ARG0_CONST(TGWin32,Size_t,GetMarkerSize)
RETURN_METHOD_ARG0_CONST(TGWin32,Short_t,GetTextAlign)
RETURN_METHOD_ARG0_CONST(TGWin32,Float_t,GetTextAngle)
RETURN_METHOD_ARG0_CONST(TGWin32,Color_t,GetTextColor)
RETURN_METHOD_ARG0_CONST(TGWin32,Font_t,GetTextFont)
RETURN_METHOD_ARG0_CONST(TGWin32,Float_t,GetTextSize)
VOID_METHOD_ARG1(TGWin32,Bell,Int_t,percent,1)
VOID_METHOD_ARG0(TGWin32,ClosePixmap,1)
VOID_METHOD_ARG0(TGWin32,CloseWindow,1)
VOID_METHOD_ARG0(TGWin32,SetDoubleBufferOFF,1)
VOID_METHOD_ARG0(TGWin32,SetDoubleBufferON,1)
VOID_METHOD_ARG0(TGWin32,glFlush,1)
VOID_METHOD_ARG0(TGWin32,glEndList,1)
VOID_METHOD_ARG0(TGWin32,glEnd,1)
VOID_METHOD_ARG0(TGWin32,glPushMatrix,1)
VOID_METHOD_ARG0(TGWin32,glPopMatrix,1)
VOID_METHOD_ARG0(TGWin32,glLoadIdentity,1)
VOID_METHOD_ARG1(TGWin32,SetClipOFF,Int_t,wid,1)
VOID_METHOD_ARG1(TGWin32,MapWindow,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,MapSubwindows,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,MapRaised,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,UnmapWindow,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,DestroyWindow,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,RaiseWindow,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,LowerWindow,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,DeleteGC,GContext_t,gc,1)
VOID_METHOD_ARG1(TGWin32,DeleteFont,FontStruct_t,fs,1)
VOID_METHOD_ARG1(TGWin32,DeletePixmap,Pixmap_t,pmap,1)
VOID_METHOD_ARG1(TGWin32,DeletePictureData,void*,data,1)
VOID_METHOD_ARG1(TGWin32,WMDeleteNotify,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,SetKeyAutoRepeat,Bool_t,on,1)
VOID_METHOD_ARG1(TGWin32,ClearWindow,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,SetInputFocus,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,FreeFontStruct,FontStruct_t,fs,1)
VOID_METHOD_ARG1(TGWin32,DestroyRegion,Region_t,reg,1)
VOID_METHOD_ARG1(TGWin32,FreeFontNames,char**,fontlist,1)
VOID_METHOD_ARG1(TGWin32,SetPrimarySelectionOwner,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,DeleteImage,Drawable_t,img,1)
VOID_METHOD_ARG1(TGWin32,IconifyWindow,Window_t,id,1)
VOID_METHOD_ARG1(TGWin32,wglDeleteContext,ULong_t,ctx,1)
VOID_METHOD_ARG1(TGWin32,glDrawBuffer,UInt_t,mode,1)
VOID_METHOD_ARG1(TGWin32,glClear,UInt_t,mode,1)
VOID_METHOD_ARG1(TGWin32,glDisable,UInt_t,mode,1)
VOID_METHOD_ARG1(TGWin32,glEnable,UInt_t,mode,1)
VOID_METHOD_ARG1(TGWin32,glFrontFace,UInt_t,mode,1)
VOID_METHOD_ARG1(TGWin32,glVertex3fv,const Float_t*,vert,1)
VOID_METHOD_ARG1(TGWin32,glIndexi,Int_t,index,1)
VOID_METHOD_ARG1(TGWin32,glPointSize,Float_t,size,1)
VOID_METHOD_ARG1(TGWin32,glLineWidth,Float_t,width,1)
VOID_METHOD_ARG1(TGWin32,glCallList,UInt_t,list,1)
VOID_METHOD_ARG1(TGWin32,glMatrixMode,UInt_t,mode,1)
VOID_METHOD_ARG1(TGWin32,glShadeModel,UInt_t,mode,1)
VOID_METHOD_ARG1(TGWin32,glNormal3fv,const Float_t*,norm,1)
VOID_METHOD_ARG1(TGWin32,glBegin,UInt_t,mode,1)
VOID_METHOD_ARG1(TGWin32,glCullFace,UInt_t,mode,1)
VOID_METHOD_ARG1(TGWin32,glClearIndex,Float_t,param,1)
VOID_METHOD_ARG1(TGWin32,glLoadMatrixd,const Double_t*,matrix,1)
VOID_METHOD_ARG1(TGWin32,glMultMatrixd,const Double_t*,matrix,1)
VOID_METHOD_ARG1(TGWin32,glColor3fv,const Float_t*,color,1)
VOID_METHOD_ARG2(TGWin32,Warp,Int_t,ix,Int_t,iy,1)
VOID_METHOD_ARG2(TGWin32,SetCharacterUp,Float_t,chupx,Float_t,chupy,1)
VOID_METHOD_ARG2(TGWin32,ChangeGC,GContext_t,gc,GCValues_t*,gval,1)
VOID_METHOD_ARG2(TGWin32,SetDoubleBuffer,Int_t,wid,Int_t,mode,1)
VOID_METHOD_ARG2(TGWin32,SetLineType,Int_t,n,Int_t*,dash,1)
VOID_METHOD_ARG2(TGWin32,SetCursor,Window_t,id,Cursor_t,curid,1)
VOID_METHOD_ARG2(TGWin32,SetWindowBackground,Window_t,id,ULong_t,color,1)
VOID_METHOD_ARG2(TGWin32,SetWindowBackgroundPixmap,Window_t,id,Pixmap_t,pxm,1)
VOID_METHOD_ARG2(TGWin32,ChangeWindowAttributes,Window_t,id,SetWindowAttributes_t*,attr,1)
VOID_METHOD_ARG2(TGWin32,FreeColor,Colormap_t,cmap,ULong_t,pixel,1)
VOID_METHOD_ARG2(TGWin32,SetWindowName,Window_t,id,char*,name,1)
VOID_METHOD_ARG2(TGWin32,SetWMTransientHint,Window_t,id,Window_t,main_id,1)
VOID_METHOD_ARG2(TGWin32,SetIconName,Window_t,id,char*,name,1)
VOID_METHOD_ARG2(TGWin32,SetIconPixmap,Window_t,id,Pixmap_t,pix,1)
VOID_METHOD_ARG2(TGWin32,SelectInput,Window_t,id,UInt_t,evmask,1)
VOID_METHOD_ARG2(TGWin32,SetForeground,GContext_t,gc,ULong_t,foreground,1)
VOID_METHOD_ARG2(TGWin32,SetWMState,Window_t,id,EInitialState,state,1)
VOID_METHOD_ARG2(TGWin32,wglMakeCurrent,Window_t,wind,ULong_t,ctx,1)
VOID_METHOD_ARG2(TGWin32,wglSwapLayerBuffers,Window_t,wind,UInt_t,mode,1)
VOID_METHOD_ARG2(TGWin32,glNewList,UInt_t,list,UInt_t,mode,1)
VOID_METHOD_ARG2(TGWin32,glDeleteLists,UInt_t,list,Int_t,sizei,1)
VOID_METHOD_ARG2(TGWin32,glPolygonMode,UInt_t,face,UInt_t,mode,1)
VOID_METHOD_ARG3(TGWin32,CopyPixmap,Int_t,wid,Int_t,xpos,Int_t,ypos,0)
VOID_METHOD_ARG3(TGWin32,SetClassHints,Window_t,id,char*,className,char*,resourceName,1)
VOID_METHOD_ARG3(TGWin32,SetWMPosition,Window_t,id,Int_t,x,Int_t,y,1)
VOID_METHOD_ARG3(TGWin32,SetWMSize,Window_t,id,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG3(TGWin32,ConvertPrimarySelection,Window_t,id,Atom_t,clipboard,Time_t,when,1)
VOID_METHOD_ARG3(TGWin32,ReadGIF,Int_t,x0,Int_t,y0,const char*,file,1)
VOID_METHOD_ARG3(TGWin32,RescaleWindow,Int_t,wid,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG3(TGWin32,MoveWindow,Window_t,id,Int_t,x,Int_t,y,1)
VOID_METHOD_ARG3(TGWin32,ResizeWindow,Window_t,id,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG3(TGWin32,glVertex3f,Float_t,x,Float_t,y,Float_t,z,1)
VOID_METHOD_ARG3(TGWin32,glTranslated,Double_t,x,Double_t,y,Double_t,z,1)
VOID_METHOD_ARG4(TGWin32,DrawSegments,Drawable_t,id,GContext_t,gc,Segment_t*,seg,Int_t,nseg,1)
VOID_METHOD_ARG4(TGWin32,SetMWMHints,Window_t,id,UInt_t,value,UInt_t,funcs,UInt_t,input,1)
VOID_METHOD_ARG4(TGWin32,SetDashes,GContext_t,gc,Int_t,offset,const char*,dash_list,Int_t,n,1)
VOID_METHOD_ARG4(TGWin32,WritePixmap,Int_t,wid,UInt_t,w,UInt_t,h,char*,pxname,1)
VOID_METHOD_ARG4(TGWin32,SetRGB,Int_t,cindex,Float_t,r,Float_t,g,Float_t,b,1)
VOID_METHOD_ARG4(TGWin32,PutPixel,Drawable_t,id,Int_t,x,Int_t,y,ULong_t,pixel,1)
VOID_METHOD_ARG4(TGWin32,GrabKey,Window_t,id,Int_t,keycode,UInt_t,modifier,Bool_t,grab,1)
VOID_METHOD_ARG4(TGWin32,FillPolygon,Window_t,id,GContext_t,gc,Point_t*,points,Int_t,npnt,1)
VOID_METHOD_ARG4(TGWin32,glViewport,Int_t,x0,Int_t,y0,Int_t,x1,Int_t,y1,1)
VOID_METHOD_ARG4(TGWin32,glClearColor,Float_t,red,Float_t,green,Float_t,blue,Float_t,alpha,1)
VOID_METHOD_ARG4(TGWin32,glRotated,Double_t,angle,Double_t,x,Double_t,y,Double_t,z,1)
VOID_METHOD_ARG5(TGWin32,MoveResizeWindow,Window_t,id,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG5(TGWin32,ChangeProperty,Window_t,id,Atom_t,property,Atom_t,type,UChar_t*,data,Int_t,len,1)
VOID_METHOD_ARG5(TGWin32,SetClipRegion,Int_t,wid,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG5(TGWin32,ClearArea,Window_t,id,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG5(TGWin32,SetClipRectangles,GContext_t,gc,Int_t,x,Int_t,y,Rectangle_t*,recs,Int_t,n,1)
VOID_METHOD_ARG6(TGWin32,GrabPointer,Window_t,id,UInt_t,evmask,Window_t,confine,Cursor_t,cursor,Bool_t,grab,Bool_t,owner_events,1)
VOID_METHOD_ARG6(TGWin32,DrawLine,Drawable_t,id,GContext_t,gc,Int_t,x1,Int_t,y1,Int_t,x2,Int_t,y2,1)
VOID_METHOD_ARG6(TGWin32,DrawString,Drawable_t,id,GContext_t,gc,Int_t,x,Int_t,y,const char*,s,Int_t,len,1)
VOID_METHOD_ARG6(TGWin32,FillRectangle,Drawable_t,id,GContext_t,gc,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG6(TGWin32,DrawRectangle,Drawable_t,id,GContext_t,gc,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG6(TGWin32,glFrustum,Double_t,min_0,Double_t,max_0,Double_t,min_1,Double_t,max_1,Double_t,dnear,Double_t,dfar,1)
VOID_METHOD_ARG6(TGWin32,glOrtho,Double_t,min_0,Double_t,max_0,Double_t,min_1,Double_t,max_1,Double_t,dnear,Double_t,dfar,1)
VOID_METHOD_ARG7(TGWin32,GrabButton,Window_t,id,EMouseButton,button,UInt_t,modifier,UInt_t,evmask,Window_t,confine,Cursor_t,cursor,Bool_t,grab,1)
VOID_METHOD_ARG7(TGWin32,DrawCellArray,Int_t,x1,Int_t,y1,Int_t,x2,Int_t,y2,Int_t,nx,Int_t,ny,Int_t*,ic,1)
VOID_METHOD_ARG7(TGWin32,SetWMSizeHints,Window_t,id,UInt_t,wmin,UInt_t,hmin,UInt_t,wmax,UInt_t,hmax,UInt_t,winc,UInt_t,hinc,1)
VOID_METHOD_ARG9(TGWin32,PutImage,Drawable_t,id,GContext_t,gc,Drawable_t,img,Int_t,dx,Int_t,dy,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG9(TGWin32,CopyArea,Drawable_t,src,Drawable_t,dest,GContext_t,gc,Int_t,src_x,Int_t,src_y,UInt_t,width,UInt_t,height,Int_t,dest_x,Int_t,dest_y,1)
VOID_METHOD_ARG2(TGWin32,glGetBooleanv,UInt_t,mode,UChar_t*,bRet,1)
VOID_METHOD_ARG2(TGWin32,glGetDoublev,UInt_t,mode,Double_t*,dRet,1)
VOID_METHOD_ARG2(TGWin32,glGetFloatv,UInt_t,mode,Float_t*,fRet,1)
VOID_METHOD_ARG2(TGWin32,glGetIntegerv,UInt_t,mode,Int_t*,iRet,1)
VOID_METHOD_ARG2(TGWin32,QueryColor,Colormap_t,cmap,ColorStruct_t&,color,1)
VOID_METHOD_ARG2(TGWin32,GetWindowAttributes,Window_t,id,WindowAttributes_t&,attr,1)
VOID_METHOD_ARG5(TGWin32,GetGeometry,Int_t,wid,Int_t&,x,Int_t&,y,UInt_t&,w,UInt_t&,h,1)
VOID_METHOD_ARG4(TGWin32,GetRGB,Int_t,index,Float_t&,r,Float_t&,g,Float_t&,b,1)
VOID_METHOD_ARG3(TGWin32,GetFontProperties,FontStruct_t,font,Int_t&,max_ascent,Int_t&,max_descent,1)
VOID_METHOD_ARG5(TGWin32,GetWindowSize,Drawable_t,id,Int_t&,x,Int_t&,y,UInt_t&,w,UInt_t&,h,1)
VOID_METHOD_ARG3(TGWin32,GetImageSize,Drawable_t,id,UInt_t&,width,UInt_t&,height,1)
VOID_METHOD_ARG3(TGWin32,UnionRectWithRegion,Rectangle_t*,rect,Region_t,src,Region_t,dest,1)
VOID_METHOD_ARG3(TGWin32,UnionRegion,Region_t,rega,Region_t,regb,Region_t,result,1)
VOID_METHOD_ARG3(TGWin32,IntersectRegion,Region_t,rega,Region_t,regb,Region_t,result,1)
VOID_METHOD_ARG3(TGWin32,SubtractRegion,Region_t,rega,Region_t,regb,Region_t,result,1)
VOID_METHOD_ARG3(TGWin32,XorRegion,Region_t,rega,Region_t,regb,Region_t,result,1)
VOID_METHOD_ARG2(TGWin32,GetRegionBox,Region_t,reg,Rectangle_t*,rect,1)
VOID_METHOD_ARG3(TGWin32,CopyGC,GContext_t,org,GContext_t,dest,Mask_t,mask,1)
VOID_METHOD_ARG3(TGWin32,GetTextExtent,UInt_t&,w,UInt_t&,h,char*,mess,1)
VOID_METHOD_ARG7(TGWin32,TranslateCoordinates,Window_t,src,Window_t,dest,Int_t,src_x,Int_t,src_y,Int_t&,dest_x,Int_t&,dest_y,Window_t&,child,1)
VOID_METHOD_ARG8(TGWin32,QueryPointer,Window_t,id,Window_t&,rootw,Window_t&,childw,Int_t&,root_x,Int_t&,root_y,Int_t&,win_x,Int_t&,win_y,UInt_t&,mask,1)
VOID_METHOD_ARG0(TGWin32,ClearWindow,1)
VOID_METHOD_ARG1(TGWin32,SetDrawMode,TVirtualX::EDrawMode,mode,1)
VOID_METHOD_ARG3(TGWin32,MoveWindow,Int_t,wid,Int_t,x,Int_t,y,1)
VOID_METHOD_ARG1(TGWin32,ResizeWindow,Int_t,winid,1)
VOID_METHOD_ARG2(TGWin32,SetCursor,Int_t,win,ECursor,cursor,1)
VOID_METHOD_ARG2(TGWin32,QueryPointer,Int_t&,ix,Int_t&,iy,1)
VOID_METHOD_ARG5(TGWin32,GetPasteBuffer,Window_t,id,Atom_t,atom,TString&,text,Int_t&,nchar,Bool_t,del,1)
VOID_METHOD_ARG1(TGWin32,GetPlanes,Int_t&,planes,1)
VOID_METHOD_ARG2(TGWin32,GetGCValues,GContext_t,gc,GCValues_t&,gval,1)
RETURN_METHOD_ARG0(TGWin32,Window_t,GetInputFocus)
RETURN_METHOD_ARG0(TGWin32,Window_t,GetPrimarySelectionOwner)
RETURN_METHOD_ARG0(TGWin32,Region_t,CreateRegion)
RETURN_METHOD_ARG0(TGWin32,Int_t,glGetError)
RETURN_METHOD_ARG0_CONST(TGWin32,Display_t,GetDisplay)
RETURN_METHOD_ARG0_CONST(TGWin32,Visual_t,GetVisual)
RETURN_METHOD_ARG0_CONST(TGWin32,Int_t,GetScreen)
RETURN_METHOD_ARG0_CONST(TGWin32,Int_t,GetDepth)
RETURN_METHOD_ARG0_CONST(TGWin32,Colormap_t,GetColormap)
RETURN_METHOD_ARG0_CONST(TGWin32,Bool_t,HasTTFonts)
RETURN_METHOD_ARG0_CONST(TGWin32,Handle_t,GetNativeEvent)
RETURN_METHOD_ARG0_CONST(TGWin32,Window_t,GetDefaultRootWindow)
RETURN_METHOD_ARG1(TGWin32,const char*,DisplayName,const char*,dpyName)
RETURN_METHOD_ARG1(TGWin32,Bool_t,Init,void*,display)
RETURN_METHOD_ARG1(TGWin32,Int_t,GetDoubleBuffer,Int_t,wid)
RETURN_METHOD_ARG1(TGWin32,Window_t,GetWindowID,Int_t,wid)
RETURN_METHOD_ARG1(TGWin32,Int_t,InitWindow,ULong_t,window)
RETURN_METHOD_ARG1(TGWin32,Int_t,WriteGIF,char*,name)
RETURN_METHOD_ARG1(TGWin32,FontStruct_t,LoadQueryFont,const char*,font_name)
RETURN_METHOD_ARG1(TGWin32,FontH_t,GetFontHandle,FontStruct_t,fs)
RETURN_METHOD_ARG1(TGWin32,Cursor_t,CreateCursor,ECursor,cursor)
RETURN_METHOD_ARG1(TGWin32,FontStruct_t,GetFontStruct,FontH_t,fh)
RETURN_METHOD_ARG1(TGWin32,Int_t,KeysymToKeycode,UInt_t,keysym)
RETURN_METHOD_ARG1(TGWin32,Int_t,OpenDisplay,const char*,dpyName)
RETURN_METHOD_ARG1(TGWin32,Bool_t,EmptyRegion,Region_t,reg)
RETURN_METHOD_ARG1(TGWin32,ULong_t,GetWinDC,Window_t,wind)
RETURN_METHOD_ARG1(TGWin32,ULong_t,wglCreateContext,Window_t,wind)
RETURN_METHOD_ARG1(TGWin32,UInt_t,glGenLists,UInt_t,list)
RETURN_METHOD_ARG2(TGWin32,Int_t,OpenPixmap,UInt_t,w,UInt_t,h)
RETURN_METHOD_ARG2(TGWin32,Atom_t,InternAtom,const char*,atom_name,Bool_t,only_if_exist)
RETURN_METHOD_ARG2(TGWin32,GContext_t,CreateGC,Drawable_t,id,GCValues_t*,gval)
RETURN_METHOD_ARG2(TGWin32,Bool_t,EqualRegion,Region_t,rega,Region_t,regb)
RETURN_METHOD_ARG2(TGWin32,Drawable_t,CreateImage,UInt_t,width,UInt_t,height)
RETURN_METHOD_ARG3(TGWin32,Int_t,ResizePixmap,Int_t,wid,UInt_t,w,UInt_t,h)
RETURN_METHOD_ARG3(TGWin32,Int_t,TextWidth,FontStruct_t,font,const char*,s,Int_t,len)
RETURN_METHOD_ARG3(TGWin32,Region_t,PolygonRegion,Point_t*,points,Int_t,np,Bool_t,winding)
RETURN_METHOD_ARG3(TGWin32,Bool_t,PointInRegion,Int_t,x,Int_t,y,Region_t,reg)
RETURN_METHOD_ARG3(TGWin32,Int_t,RequestString,Int_t,x,Int_t,y,char*,text)
RETURN_METHOD_ARG3(TGWin32,Window_t,CreateGLWindow,Window_t,wind,Visual_t,visual,Int_t,depth)
RETURN_METHOD_ARG4(TGWin32,Pixmap_t,CreateBitmap,Drawable_t,id,const char*,bitmap,UInt_t,width,UInt_t,height)
RETURN_METHOD_ARG7(TGWin32,Pixmap_t,CreatePixmap,Drawable_t,id,const char*,bitmap,UInt_t,width,UInt_t,height,ULong_t,forecolor,ULong_t,backcolor,Int_t,depth)
RETURN_METHOD_ARG11(TGWin32,Window_t,CreateWindow,Window_t,parent,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,UInt_t,border,Int_t,depth,UInt_t,clss,void*,visual,SetWindowAttributes_t*,attr,UInt_t,wtype)
RETURN_METHOD_ARG3(TGWin32,char**,ListFonts,char*,fontname,Int_t,mx,Int_t&,count)
RETURN_METHOD_ARG4(TGWin32,Int_t,RequestLocator,Int_t,mode,Int_t,ctyp,Int_t&,x,Int_t&,y)
RETURN_METHOD_ARG3(TGWin32,Bool_t,ParseColor,Colormap_t,cmap,const char*,cname,ColorStruct_t&,color)
RETURN_METHOD_ARG2(TGWin32,Bool_t,AllocColor,Colormap_t,cmap,ColorStruct_t&,color)
RETURN_METHOD_ARG5(TGWin32,Bool_t,CreatePictureFromFile,Drawable_t,id,const char*,filename,Pixmap_t&,pict,Pixmap_t&,pict_mask,PictureAttributes_t&,attr)
RETURN_METHOD_ARG5(TGWin32,Bool_t,CreatePictureFromData,Drawable_t,id,char**,data,Pixmap_t&,pict,Pixmap_t&,pict_mask,PictureAttributes_t&,attr)
RETURN_METHOD_ARG2(TGWin32,Bool_t,ReadPictureDataFromFile,const char*,filename,char***,ret_data)
RETURN_METHOD_ARG2(TGWin32,Int_t,SetTextFont,char*,fontname,TVirtualX::ETextSetMode,mode)
RETURN_METHOD_ARG3(TGWin32,Pixmap_t,CreatePixmap,Drawable_t,wid,UInt_t,w,UInt_t,h)
RETURN_METHOD_ARG1(TGWin32,ULong_t,GetPixel,Color_t,cindex)

RETURN_METHOD_ARG3(TGWin32,Bool_t,CheckEvent,Window_t,id,EGEventType,type,Event_t&,ev)
VOID_METHOD_ARG2(TGWin32,SendEvent,Window_t,id,Event_t*,ev,1)
VOID_METHOD_ARG1(TGWin32,NextEvent,Event_t&,event,1)
//RETURN_METHOD_ARG0(TGWin32,Int_t,EventsPending)

//VOID_METHOD_ARG1(TGWin32,CreateOpenGLContext,Int_t,wid,1)
//VOID_METHOD_ARG1(TGWin32,DeleteOpenGLContext,Int_t,wid,1)
//VOID_METHOD_ARG1(TGWin32,RemoveWindow,ULong_t,qwid,1)
//RETURN_METHOD_ARG1(TGWin32,ExecCommand,UInt_t,TGWin32Command*,code)
//RETURN_METHOD_ARG3(TGWin32,Int_t,AddWindow,ULong_t,qwid,UInt_t,w,UInt_t,h)


//////////////////////// some non-standard methods /////////////////////////////
/*
//______________________________________________________________________________
Bool_t TGWin32Proxy::CheckEvent(Window_t id,EGEventType type,Event_t& ev)
{
   //

   return TGWin32::Instance()->CheckEvent(id,type,ev);
}

//______________________________________________________________________________
void TGWin32Proxy::NextEvent(Event_t& event)
{
   //

   TGWin32::Instance()->NextEvent(event);
}

//______________________________________________________________________________
void TGWin32Proxy::SendEvent(Window_t id,Event_t* ev)
{
   //

   TGWin32::Instance()->SendEvent(id,ev);
}
*/


//______________________________________________________________________________
void p2TGWin32ProxyEventsPending(void* in)
{
   //

  struct tmp {
      Int_t ret;
   };
   tmp *p = (tmp*)in;
   p->ret = TGWin32::Instance()->EventsPending();
}

Int_t TGWin32Proxy::EventsPending()
{
   //

   DEBUG_PROFILE_PROXY_START(EventsPending)
   Int_t ret;
   struct tmp {
      Int_t ret;
   };
   fParam = new tmp;
   fCallBack = &p2TGWin32ProxyEventsPending;
   Bool_t batch = ForwardCallBack(1);
   ret  = ((tmp*)fParam)->ret;
   if (!batch) delete fParam;
   if (gDebugProxy) {
      if (debug) {
         double dt = GetMilliSeconds() - start;
         i++; total++;
         t += dt;
         total_time += dt;
         //if (gDebugValue==kDebugTrace) printf(#method " %d\n",i);
      }
   }
   return ret;
}

//______________________________________________________________________________
void TGWin32Proxy::CloseDisplay()
{
   //

   if (gDebug) printf("CloseDisplay\n");
   SendExitMessage();
}

//______________________________________________________________________________
Window_t TGWin32Proxy::GetParent(Window_t id) const
{
   // might be thread unsafe (?)

   return (Window_t)gdk_window_get_parent((GdkWindow *) id);
}

//______________________________________________________________________________
ULong_t TGWin32Proxy::GetPixel(Drawable_t id, Int_t x, Int_t y)
{
   //

   ULong_t ret;
   DEBUG_PROFILE_PROXY_START(GetPixel)
   Lock();
   ret =  TGWin32::GetPixel(id,x,y);
   DEBUG_PROFILE_PROXY_STOP(GetPixel)
   Unlock();
   return ret;
}

//______________________________________________________________________________
void TGWin32Proxy::LookupString(Event_t * event, char *buf, Int_t buflen,
                                UInt_t & keysym)
{
   // Convert the keycode from the event structure to a key symbol (according
   // to the modifiers specified in the event structure and the current
   // keyboard mapping). In buf a null terminated ASCII string is returned
   // representing the string that is currently mapped to the key code.

   DEBUG_PROFILE_PROXY_START(LookupString)
   TGWin32::Instance()->LookupString(event,buf,buflen,keysym);
   DEBUG_PROFILE_PROXY_STOP(LookupString)
}
