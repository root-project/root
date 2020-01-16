// @(#)root/win32gdk:$Id$
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
#include "TGWin32VirtualXProxy.h"
#include "TGWin32.h"
#include "TList.h"

TVirtualX *TGWin32VirtualXProxy::fgRealObject = 0;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TVirtualX *TGWin32VirtualXProxy::RealObject()
{
   return fgRealObject;
}

RETURN_PROXY_OBJECT(VirtualX)
VOID_METHOD_ARG0(VirtualX,SetFillAttributes,1)
VOID_METHOD_ARG0(VirtualX,SetMarkerAttributes,1)
VOID_METHOD_ARG0(VirtualX,SetLineAttributes,1)
VOID_METHOD_ARG0(VirtualX,SetTextAttributes,1)
VOID_METHOD_ARG1(VirtualX,ResetAttMarker,Option_t*,toption,1)
VOID_METHOD_ARG1(VirtualX,ResetAttFill,Option_t*,option,1)
VOID_METHOD_ARG1(VirtualX,ResetAttLine,Option_t*,option,1)
VOID_METHOD_ARG1(VirtualX,ResetAttText,Option_t*,option,1)
VOID_METHOD_ARG1(VirtualX,SetFillStyle,Style_t,style,1)
VOID_METHOD_ARG1(VirtualX,SetTextAngle,Float_t,tangle,1)
VOID_METHOD_ARG1(VirtualX,SetTextMagnitude,Float_t,mgn,1)
VOID_METHOD_ARG1(VirtualX,SetTextSizePixels,Int_t,npixels,1)
VOID_METHOD_ARG1(VirtualX,SetFillColor,Color_t,cindex,1)
VOID_METHOD_ARG1(VirtualX,SetMarkerSize,Float_t,markersize,1)
VOID_METHOD_ARG1(VirtualX,SetMarkerStyle,Style_t,markerstyle,1)
VOID_METHOD_ARG1(VirtualX,SetMarkerColor,Color_t,cindex,1)
VOID_METHOD_ARG1(VirtualX,SetLineColor,Color_t,cindex,1)
VOID_METHOD_ARG1(VirtualX,SetLineStyle,Style_t,linestyle,1)
VOID_METHOD_ARG1(VirtualX,SetLineWidth,Width_t,width,1)
VOID_METHOD_ARG1(VirtualX,SetTextAlign,Short_t,talign,1)
VOID_METHOD_ARG1(VirtualX,SetTextSize,Float_t,textsize,1)
VOID_METHOD_ARG1(VirtualX,SetTextColor,Color_t,cindex,1)
VOID_METHOD_ARG1(VirtualX,SetTextFont,Font_t,fontnumber,1)
VOID_METHOD_ARG1(VirtualX,SelectWindow,Int_t,wid,0)
VOID_METHOD_ARG2(VirtualX,DrawFillArea,Int_t,n,TPoint*,xy,1)
VOID_METHOD_ARG2(VirtualX,DrawPolyLine,Int_t,n,TPoint*,xy,1)
VOID_METHOD_ARG2(VirtualX,DrawPolyMarker,Int_t,n,TPoint*,xy,1)
VOID_METHOD_ARG1(VirtualX,UpdateWindow,Int_t,mode,1)
VOID_METHOD_ARG1(VirtualX,SetOpacity,Int_t,percent,1)
VOID_METHOD_ARG5(VirtualX,DrawBox,Int_t,x1,Int_t,y1,Int_t,x2,Int_t,y2,TVirtualX::EBoxMode,mode,1)
VOID_METHOD_ARG6(VirtualX,DrawText,Int_t,x,Int_t,y,Float_t,angle,Float_t,mgn,const char*,text,TVirtualX::ETextMode,mode,1)
VOID_METHOD_ARG1(VirtualX,Update,Int_t,mode,1)
VOID_METHOD_ARG4(VirtualX,DrawLine,Int_t,x1,Int_t,y1,Int_t,x2,Int_t,y2,0)
VOID_METHOD_ARG2(VirtualX,GetCharacterUp,Float_t&,chupx,Float_t&,chupy,1)
RETURN_METHOD_ARG0(VirtualX,Float_t,GetTextMagnitude)
RETURN_METHOD_ARG0_CONST(VirtualX,Color_t,GetFillColor)
RETURN_METHOD_ARG0_CONST(VirtualX,Style_t,GetFillStyle)
RETURN_METHOD_ARG0_CONST(VirtualX,Bool_t,IsTransparent)
RETURN_METHOD_ARG0_CONST(VirtualX,Color_t,GetLineColor)
RETURN_METHOD_ARG0_CONST(VirtualX,Style_t,GetLineStyle)
RETURN_METHOD_ARG0_CONST(VirtualX,Width_t,GetLineWidth)
RETURN_METHOD_ARG0_CONST(VirtualX,Color_t,GetMarkerColor)
RETURN_METHOD_ARG0_CONST(VirtualX,Style_t,GetMarkerStyle)
RETURN_METHOD_ARG0_CONST(VirtualX,Size_t,GetMarkerSize)
RETURN_METHOD_ARG0_CONST(VirtualX,Style_t,GetMarkerStyleBase)
RETURN_METHOD_ARG0_CONST(VirtualX,Width_t,GetMarkerLineWidth)
RETURN_METHOD_ARG0_CONST(VirtualX,Short_t,GetTextAlign)
RETURN_METHOD_ARG0_CONST(VirtualX,Float_t,GetTextAngle)
RETURN_METHOD_ARG0_CONST(VirtualX,Color_t,GetTextColor)
RETURN_METHOD_ARG0_CONST(VirtualX,Font_t,GetTextFont)
RETURN_METHOD_ARG0_CONST(VirtualX,Float_t,GetTextSize)
VOID_METHOD_ARG1(VirtualX,Bell,Int_t,percent,1)
VOID_METHOD_ARG0(VirtualX,ClosePixmap,1)
VOID_METHOD_ARG0(VirtualX,CloseWindow,1)
VOID_METHOD_ARG0(VirtualX,SetDoubleBufferOFF,1)
VOID_METHOD_ARG0(VirtualX,SetDoubleBufferON,1)
VOID_METHOD_ARG1(VirtualX,SetClipOFF,Int_t,wid,1)
VOID_METHOD_ARG1(VirtualX,MapWindow,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,MapSubwindows,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,MapRaised,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,UnmapWindow,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,DestroyWindow,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,DestroySubwindows,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,RaiseWindow,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,LowerWindow,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,DeleteGC,GContext_t,gc,1)
VOID_METHOD_ARG1(VirtualX,DeleteFont,FontStruct_t,fs,1)
VOID_METHOD_ARG1(VirtualX,DeletePixmap,Pixmap_t,pmap,1)
VOID_METHOD_ARG1(VirtualX,DeletePictureData,void*,data,1)
VOID_METHOD_ARG1(VirtualX,WMDeleteNotify,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,SetKeyAutoRepeat,Bool_t,on,1)
VOID_METHOD_ARG1(VirtualX,ClearWindow,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,SetInputFocus,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,FreeFontStruct,FontStruct_t,fs,1)
VOID_METHOD_ARG1(VirtualX,DestroyRegion,Region_t,reg,1)
VOID_METHOD_ARG1(VirtualX,FreeFontNames,char**,fontlist,1)
VOID_METHOD_ARG1(VirtualX,SetPrimarySelectionOwner,Window_t,id,1)
VOID_METHOD_ARG1(VirtualX,DeleteImage,Drawable_t,img,1)
VOID_METHOD_ARG1(VirtualX,IconifyWindow,Window_t,id,1)
VOID_METHOD_ARG3(VirtualX,Warp,Int_t,ix,Int_t,iy,Window_t,id,1)
VOID_METHOD_ARG2(VirtualX,SetCharacterUp,Float_t,chupx,Float_t,chupy,1)
VOID_METHOD_ARG2(VirtualX,ChangeGC,GContext_t,gc,GCValues_t*,gval,1)
VOID_METHOD_ARG2(VirtualX,SetDoubleBuffer,Int_t,wid,Int_t,mode,1)
VOID_METHOD_ARG2(VirtualX,SetLineType,Int_t,n,Int_t*,dash,1)
VOID_METHOD_ARG2(VirtualX,SetCursor,Window_t,id,Cursor_t,curid,1)
VOID_METHOD_ARG2(VirtualX,SetWindowBackground,Window_t,id,ULong_t,color,1)
VOID_METHOD_ARG2(VirtualX,SetWindowBackgroundPixmap,Window_t,id,Pixmap_t,pxm,1)
VOID_METHOD_ARG2(VirtualX,ChangeWindowAttributes,Window_t,id,SetWindowAttributes_t*,attr,1)
VOID_METHOD_ARG2(VirtualX,FreeColor,Colormap_t,cmap,ULong_t,pixel,1)
VOID_METHOD_ARG2(VirtualX,SetWindowName,Window_t,id,char*,name,1)
VOID_METHOD_ARG2(VirtualX,SetWMTransientHint,Window_t,id,Window_t,main_id,1)
VOID_METHOD_ARG2(VirtualX,SetIconName,Window_t,id,char*,name,1)
VOID_METHOD_ARG2(VirtualX,SetIconPixmap,Window_t,id,Pixmap_t,pix,1)
VOID_METHOD_ARG2(VirtualX,SelectInput,Window_t,id,UInt_t,evmask,1)
VOID_METHOD_ARG2(VirtualX,SetForeground,GContext_t,gc,ULong_t,foreground,1)
VOID_METHOD_ARG2(VirtualX,SetWMState,Window_t,id,EInitialState,state,1)
VOID_METHOD_ARG3(VirtualX,CopyPixmap,Int_t,wid,Int_t,xpos,Int_t,ypos,0)
VOID_METHOD_ARG3(VirtualX,SetClassHints,Window_t,id,char*,className,char*,resourceName,1)
VOID_METHOD_ARG3(VirtualX,SetWMPosition,Window_t,id,Int_t,x,Int_t,y,1)
VOID_METHOD_ARG3(VirtualX,SetWMSize,Window_t,id,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG3(VirtualX,ConvertPrimarySelection,Window_t,id,Atom_t,clipboard,Time_t,when,1)
VOID_METHOD_ARG3(VirtualX,ReadGIF,Int_t,x0,Int_t,y0,const char*,file,1)
VOID_METHOD_ARG3(VirtualX,RescaleWindow,Int_t,wid,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG3(VirtualX,MoveWindow,Window_t,id,Int_t,x,Int_t,y,1)
VOID_METHOD_ARG3(VirtualX,ResizeWindow,Window_t,id,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG4(VirtualX,DrawSegments,Drawable_t,id,GContext_t,gc,Segment_t*,seg,Int_t,nseg,1)
VOID_METHOD_ARG4(VirtualX,SetMWMHints,Window_t,id,UInt_t,value,UInt_t,funcs,UInt_t,input,1)
VOID_METHOD_ARG4(VirtualX,SetDashes,GContext_t,gc,Int_t,offset,const char*,dash_list,Int_t,n,1)
VOID_METHOD_ARG4(VirtualX,WritePixmap,Int_t,wid,UInt_t,w,UInt_t,h,char*,pxname,1)
VOID_METHOD_ARG4(VirtualX,SetRGB,Int_t,cindex,Float_t,r,Float_t,g,Float_t,b,1)
VOID_METHOD_ARG4(VirtualX,PutPixel,Drawable_t,id,Int_t,x,Int_t,y,ULong_t,pixel,1)
VOID_METHOD_ARG4(VirtualX,GrabKey,Window_t,id,Int_t,keycode,UInt_t,modifier,Bool_t,grab,1)
VOID_METHOD_ARG4(VirtualX,FillPolygon,Window_t,id,GContext_t,gc,Point_t*,points,Int_t,npnt,1)
VOID_METHOD_ARG4(VirtualX,ReparentWindow,Window_t,id,Window_t,pid,Int_t,x,Int_t,y,1)
VOID_METHOD_ARG5(VirtualX,MoveResizeWindow,Window_t,id,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG5(VirtualX,ChangeProperty,Window_t,id,Atom_t,property,Atom_t,type,UChar_t*,data,Int_t,len,1)
VOID_METHOD_ARG5(VirtualX,SetClipRegion,Int_t,wid,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG5(VirtualX,ClearArea,Window_t,id,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG5(VirtualX,SetClipRectangles,GContext_t,gc,Int_t,x,Int_t,y,Rectangle_t*,recs,Int_t,n,1)
VOID_METHOD_ARG6(VirtualX,GrabPointer,Window_t,id,UInt_t,evmask,Window_t,confine,Cursor_t,cursor,Bool_t,grab,Bool_t,owner_events,1)
VOID_METHOD_ARG6(VirtualX,DrawLine,Drawable_t,id,GContext_t,gc,Int_t,x1,Int_t,y1,Int_t,x2,Int_t,y2,1)
VOID_METHOD_ARG6(VirtualX,DrawString,Drawable_t,id,GContext_t,gc,Int_t,x,Int_t,y,const char*,s,Int_t,len,1)
VOID_METHOD_ARG6(VirtualX,FillRectangle,Drawable_t,id,GContext_t,gc,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG6(VirtualX,DrawRectangle,Drawable_t,id,GContext_t,gc,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG7(VirtualX,GrabButton,Window_t,id,EMouseButton,button,UInt_t,modifier,UInt_t,evmask,Window_t,confine,Cursor_t,cursor,Bool_t,grab,1)
VOID_METHOD_ARG7(VirtualX,DrawCellArray,Int_t,x1,Int_t,y1,Int_t,x2,Int_t,y2,Int_t,nx,Int_t,ny,Int_t*,ic,1)
VOID_METHOD_ARG7(VirtualX,SetWMSizeHints,Window_t,id,UInt_t,wmin,UInt_t,hmin,UInt_t,wmax,UInt_t,hmax,UInt_t,winc,UInt_t,hinc,1)
VOID_METHOD_ARG9(VirtualX,PutImage,Drawable_t,id,GContext_t,gc,Drawable_t,img,Int_t,dx,Int_t,dy,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,1)
VOID_METHOD_ARG9(VirtualX,CopyArea,Drawable_t,src,Drawable_t,dest,GContext_t,gc,Int_t,src_x,Int_t,src_y,UInt_t,width,UInt_t,height,Int_t,dest_x,Int_t,dest_y,1)
VOID_METHOD_ARG2(VirtualX,QueryColor,Colormap_t,cmap,ColorStruct_t&,color,1)
VOID_METHOD_ARG2(VirtualX,GetWindowAttributes,Window_t,id,WindowAttributes_t&,attr,1)
VOID_METHOD_ARG5(VirtualX,GetGeometry,Int_t,wid,Int_t&,x,Int_t&,y,UInt_t&,w,UInt_t&,h,1)
VOID_METHOD_ARG4(VirtualX,GetRGB,Int_t,index,Float_t&,r,Float_t&,g,Float_t&,b,1)
VOID_METHOD_ARG3(VirtualX,GetFontProperties,FontStruct_t,font,Int_t&,max_ascent,Int_t&,max_descent,1)
VOID_METHOD_ARG5(VirtualX,GetWindowSize,Drawable_t,id,Int_t&,x,Int_t&,y,UInt_t&,w,UInt_t&,h,1)
VOID_METHOD_ARG3(VirtualX,GetImageSize,Drawable_t,id,UInt_t&,width,UInt_t&,height,1)
VOID_METHOD_ARG3(VirtualX,UnionRectWithRegion,Rectangle_t*,rect,Region_t,src,Region_t,dest,1)
VOID_METHOD_ARG3(VirtualX,UnionRegion,Region_t,rega,Region_t,regb,Region_t,result,1)
VOID_METHOD_ARG3(VirtualX,IntersectRegion,Region_t,rega,Region_t,regb,Region_t,result,1)
VOID_METHOD_ARG3(VirtualX,SubtractRegion,Region_t,rega,Region_t,regb,Region_t,result,1)
VOID_METHOD_ARG3(VirtualX,XorRegion,Region_t,rega,Region_t,regb,Region_t,result,1)
VOID_METHOD_ARG2(VirtualX,GetRegionBox,Region_t,reg,Rectangle_t*,rect,1)
VOID_METHOD_ARG3(VirtualX,CopyGC,GContext_t,org,GContext_t,dest,Mask_t,mask,1)
VOID_METHOD_ARG3(VirtualX,GetTextExtent,UInt_t&,w,UInt_t&,h,char*,mess,1)
VOID_METHOD_ARG7(VirtualX,TranslateCoordinates,Window_t,src,Window_t,dest,Int_t,src_x,Int_t,src_y,Int_t&,dest_x,Int_t&,dest_y,Window_t&,child,1)
VOID_METHOD_ARG8(VirtualX,QueryPointer,Window_t,id,Window_t&,rootw,Window_t&,childw,Int_t&,root_x,Int_t&,root_y,Int_t&,win_x,Int_t&,win_y,UInt_t&,mask,1)
VOID_METHOD_ARG0(VirtualX,ClearWindow,1)
VOID_METHOD_ARG1(VirtualX,SetDrawMode,TVirtualX::EDrawMode,mode,1)
VOID_METHOD_ARG3(VirtualX,MoveWindow,Int_t,wid,Int_t,x,Int_t,y,1)
VOID_METHOD_ARG1(VirtualX,ResizeWindow,Int_t,winid,1)
VOID_METHOD_ARG2(VirtualX,SetCursor,Int_t,win,ECursor,cursor,1)
VOID_METHOD_ARG2(VirtualX,QueryPointer,Int_t&,ix,Int_t&,iy,1)
VOID_METHOD_ARG5(VirtualX,GetPasteBuffer,Window_t,id,Atom_t,atom,TString&,text,Int_t&,nchar,Bool_t,del,1)
VOID_METHOD_ARG1(VirtualX,GetPlanes,Int_t&,planes,1)
VOID_METHOD_ARG2(VirtualX,GetGCValues,GContext_t,gc,GCValues_t&,gval,1)
RETURN_METHOD_ARG0(VirtualX,Window_t,GetInputFocus)
RETURN_METHOD_ARG0(VirtualX,Window_t,GetPrimarySelectionOwner)
RETURN_METHOD_ARG0(VirtualX,Region_t,CreateRegion)
RETURN_METHOD_ARG0_CONST(VirtualX,Display_t,GetDisplay)
RETURN_METHOD_ARG0_CONST(VirtualX,Visual_t,GetVisual)
RETURN_METHOD_ARG0_CONST(VirtualX,Int_t,GetScreen)
RETURN_METHOD_ARG0_CONST(VirtualX,Int_t,GetDepth)
RETURN_METHOD_ARG0_CONST(VirtualX,Colormap_t,GetColormap)
RETURN_METHOD_ARG0_CONST(VirtualX,Bool_t,HasTTFonts)
RETURN_METHOD_ARG0_CONST(VirtualX,Handle_t,GetNativeEvent)
RETURN_METHOD_ARG0_CONST(VirtualX,Window_t,GetDefaultRootWindow)
RETURN_METHOD_ARG1(VirtualX,const char*,DisplayName,const char*,dpyName)
RETURN_METHOD_ARG1(VirtualX,Bool_t,Init,void*,display)
RETURN_METHOD_ARG1(VirtualX,Int_t,GetDoubleBuffer,Int_t,wid)
RETURN_METHOD_ARG1(VirtualX,Window_t,GetWindowID,Int_t,wid)
RETURN_METHOD_ARG1(VirtualX,Int_t,InitWindow,ULong_t,window)
RETURN_METHOD_ARG1(VirtualX,Int_t,WriteGIF,char*,name)
RETURN_METHOD_ARG1(VirtualX,FontStruct_t,LoadQueryFont,const char*,font_name)
RETURN_METHOD_ARG1(VirtualX,FontH_t,GetFontHandle,FontStruct_t,fs)
RETURN_METHOD_ARG1(VirtualX,Cursor_t,CreateCursor,ECursor,cursor)
RETURN_METHOD_ARG1(VirtualX,FontStruct_t,GetFontStruct,FontH_t,fh)
RETURN_METHOD_ARG1(VirtualX,Int_t,KeysymToKeycode,UInt_t,keysym)
RETURN_METHOD_ARG1(VirtualX,Int_t,OpenDisplay,const char*,dpyName)
RETURN_METHOD_ARG1(VirtualX,Bool_t,EmptyRegion,Region_t,reg)
RETURN_METHOD_ARG2(VirtualX,Int_t,OpenPixmap,UInt_t,w,UInt_t,h)
RETURN_METHOD_ARG2(VirtualX,Atom_t,InternAtom,const char*,atom_name,Bool_t,only_if_exist)
RETURN_METHOD_ARG2(VirtualX,GContext_t,CreateGC,Drawable_t,id,GCValues_t*,gval)
RETURN_METHOD_ARG2(VirtualX,Bool_t,EqualRegion,Region_t,rega,Region_t,regb)
RETURN_METHOD_ARG2(VirtualX,Drawable_t,CreateImage,UInt_t,width,UInt_t,height)
RETURN_METHOD_ARG3(VirtualX,Int_t,ResizePixmap,Int_t,wid,UInt_t,w,UInt_t,h)
RETURN_METHOD_ARG3(VirtualX,Int_t,TextWidth,FontStruct_t,font,const char*,s,Int_t,len)
RETURN_METHOD_ARG3(VirtualX,Region_t,PolygonRegion,Point_t*,points,Int_t,np,Bool_t,winding)
RETURN_METHOD_ARG3(VirtualX,Bool_t,PointInRegion,Int_t,x,Int_t,y,Region_t,reg)
RETURN_METHOD_ARG3(VirtualX,Int_t,RequestString,Int_t,x,Int_t,y,char*,text)
RETURN_METHOD_ARG4(VirtualX,Pixmap_t,CreateBitmap,Drawable_t,id,const char*,bitmap,UInt_t,width,UInt_t,height)
RETURN_METHOD_ARG7(VirtualX,Pixmap_t,CreatePixmap,Drawable_t,id,const char*,bitmap,UInt_t,width,UInt_t,height,ULong_t,forecolor,ULong_t,backcolor,Int_t,depth)
RETURN_METHOD_ARG11(VirtualX,Window_t,CreateWindow,Window_t,parent,Int_t,x,Int_t,y,UInt_t,w,UInt_t,h,UInt_t,border,Int_t,depth,UInt_t,clss,void*,visual,SetWindowAttributes_t*,attr,UInt_t,wtype)
RETURN_METHOD_ARG3(VirtualX,char**,ListFonts,const char*,fontname,Int_t,mx,Int_t&,count)
RETURN_METHOD_ARG4(VirtualX,Int_t,RequestLocator,Int_t,mode,Int_t,ctyp,Int_t&,x,Int_t&,y)
RETURN_METHOD_ARG3(VirtualX,Bool_t,ParseColor,Colormap_t,cmap,const char*,cname,ColorStruct_t&,color)
RETURN_METHOD_ARG2(VirtualX,Bool_t,AllocColor,Colormap_t,cmap,ColorStruct_t&,color)
RETURN_METHOD_ARG5(VirtualX,Bool_t,CreatePictureFromFile,Drawable_t,id,const char*,filename,Pixmap_t&,pict,Pixmap_t&,pict_mask,PictureAttributes_t&,attr)
RETURN_METHOD_ARG5(VirtualX,Bool_t,CreatePictureFromData,Drawable_t,id,char**,data,Pixmap_t&,pict,Pixmap_t&,pict_mask,PictureAttributes_t&,attr)
RETURN_METHOD_ARG2(VirtualX,Bool_t,ReadPictureDataFromFile,const char*,filename,char***,ret_data)
RETURN_METHOD_ARG2(VirtualX,Int_t,SetTextFont,char*,fontname,TVirtualX::ETextSetMode,mode)
RETURN_METHOD_ARG3(VirtualX,Pixmap_t,CreatePixmap,Drawable_t,wid,UInt_t,w,UInt_t,h)
RETURN_METHOD_ARG1(VirtualX,ULong_t,GetPixel,Color_t,cindex)
RETURN_METHOD_ARG5(VirtualX,unsigned char*,GetColorBits,Drawable_t,wid,Int_t,x,Int_t,y,UInt_t,width,UInt_t,height)
RETURN_METHOD_ARG3(VirtualX,Pixmap_t,CreatePixmapFromData,unsigned char*,bits,UInt_t,width,UInt_t,height)
RETURN_METHOD_ARG3(VirtualX,Int_t,AddWindow,ULong_t,qwid,UInt_t,w,UInt_t,h)
VOID_METHOD_ARG1(VirtualX,RemoveWindow,ULong_t,qwid,1)
VOID_METHOD_ARG4(VirtualX,ShapeCombineMask,Window_t,id,Int_t,x,Int_t,y,Pixmap_t,mask,1)

VOID_METHOD_ARG2(VirtualX,DeleteProperty,Window_t,win,Atom_t&,prop,1)
RETURN_METHOD_ARG11(VirtualX,Int_t,GetProperty,Window_t,win,Atom_t,prop,Long_t,offset,Long_t,length,Bool_t,del,Atom_t,req_type,Atom_t*,act_type,Int_t*,act_format,ULong_t*,nitems,ULong_t*,bytes,unsigned char**,prop_list)
VOID_METHOD_ARG3(VirtualX,ChangeActivePointerGrab,Window_t,win,UInt_t,mask,Cursor_t,cur,1)
VOID_METHOD_ARG5(VirtualX,ConvertSelection,Window_t,win,Atom_t&,sel,Atom_t&,target,Atom_t&,prop,Time_t&,stamp,1)
RETURN_METHOD_ARG2(VirtualX,Bool_t,SetSelectionOwner,Window_t,win,Atom_t&,prop)
VOID_METHOD_ARG6(VirtualX,ChangeProperties,Window_t,id,Atom_t,property,Atom_t,type,Int_t,format,UChar_t*,data,Int_t,len,1)
VOID_METHOD_ARG2(VirtualX,SetDNDAware,Window_t,id,Atom_t*,typelist,1)
VOID_METHOD_ARG3(VirtualX,SetTypeList,Window_t,win,Atom_t,prop,Atom_t*,typelist,1);
RETURN_METHOD_ARG6(VirtualX,Window_t,FindRWindow,Window_t,win,Window_t,dragwin,Window_t,input,int,x,int,y,int,maxd);
RETURN_METHOD_ARG2(VirtualX,Bool_t,IsDNDAware,Window_t,win,Atom_t*,typelist);

//VOID_METHOD_ARG1(VirtualX,CreateOpenGLContext,Int_t,wid,1)
//VOID_METHOD_ARG1(VirtualX,DeleteOpenGLContext,Int_t,wid,1)
//VOID_METHOD_ARG1(VirtualX,RemoveWindow,ULong_t,qwid,1)
//RETURN_METHOD_ARG1(VirtualX,ExecCommand,UInt_t,TGWin32Command*,code)
//RETURN_METHOD_ARG3(VirtualX,Int_t,AddWindow,ULong_t,qwid,UInt_t,w,UInt_t,h)

//////////////////////// some non-standard methods /////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///

void TGWin32VirtualXProxy::CloseDisplay()
{
   if (gDebug) printf("CloseDisplay\n");
   fgRealObject->CloseDisplay();
}

////////////////////////////////////////////////////////////////////////////////
/// might be thread unsafe (?)

Window_t TGWin32VirtualXProxy::GetParent(Window_t id) const
{
   return (Window_t)gdk_window_get_parent((GdkWindow *) id);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert the keycode from the event structure to a key symbol (according
/// to the modifiers specified in the event structure and the current
/// keyboard mapping). In buf a null terminated ASCII string is returned
/// representing the string that is currently mapped to the key code.

void TGWin32VirtualXProxy::LookupString(Event_t * event, char *buf, Int_t buflen,
                                UInt_t & keysym)
{
   DEBUG_PROFILE_PROXY_START(LookupString)
   fgRealObject->LookupString(event,buf,buflen,keysym);
   DEBUG_PROFILE_PROXY_STOP(LookupString)
}

////////////////////////////////////////////////////////////////////////////////
/// Return number of pending events.

Int_t TGWin32VirtualXProxy::EventsPending()
{  
   return fgRealObject->EventsPending();
}

////////////////////////////////////////////////////////////////////////////////
/// Process next event in the queue - if any.

void TGWin32VirtualXProxy::NextEvent(Event_t & event)
{
   fgRealObject->NextEvent(event);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if there is for window "id" an event of type "type".

Bool_t TGWin32VirtualXProxy::CheckEvent(Window_t id, EGEventType type, Event_t &ev)
{
   return fgRealObject->CheckEvent(id,type,ev);
}

////////////////////////////////////////////////////////////////////////////////
/// Send event ev to window id.

void TGWin32VirtualXProxy::SendEvent(Window_t id, Event_t *ev)
{
   fgRealObject->SendEvent(id,ev);
}

////////////////////////////////////////////////////////////////////////////////
/// Return kTRUE if we are inside cmd/server thread.

Bool_t TGWin32VirtualXProxy::IsCmdThread() const 
{
   return fgRealObject->IsCmdThread();
}

////////////////////////////////////////////////////////////////////////////////
/// Return a pointer to the current window.

Window_t TGWin32VirtualXProxy::GetCurrentWindow() const 
{
   return fgRealObject->GetCurrentWindow(); 
}

