// @(#)root/qt:$Name:$:$Id:$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQtThread                                                            //
//                                                                      //
// Interface to low level Qt GUI. This class gives access to basic      //
// Qt graphics, pixmap, text and font handling routines.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <qapplication.h>

#include "TQtThread.h"
#include "TQtApplication.h"
#include "TQtEvent.h"
#include "TWaitCondition.h"

#define BASECLASS TGQt
#define THREADCLASS TQtThread

#include "TQtThreadStub.h"

//______________________________________________________________________________
//
//                  /---- Methods used for GUI -----
//______________________________________________________________________________

//______________________________________________________________________________
  RETURNACTION1(Window_t,GetWindowID,Int_t,wid);
//______________________________________________________________________________
  SENDACTION1(SetOpacity,Int_t,percent);
//______________________________________________________________________________
  VOIDACTION2(GetWindowAttributes,Window_t, id, WindowAttributes_t &,attr);
//______________________________________________________________________________
  VOIDACTION1(MapWindow,Window_t,id);
//______________________________________________________________________________
  VOIDACTION1(MapSubwindows,Window_t, id);
//______________________________________________________________________________
   SENDACTION1(MapRaised,Window_t, id);
//______________________________________________________________________________
   VOIDACTION1(UnmapWindow,Window_t, id);
//______________________________________________________________________________
   SENDACTION1(DestroyWindow,Window_t, id);
//______________________________________________________________________________
   SENDACTION1(RaiseWindow,Window_t, id);
//______________________________________________________________________________
   SENDACTION1(LowerWindow,Window_t, id);
//______________________________________________________________________________
   SENDACTION3(MoveWindow,Window_t, id, Int_t,x, Int_t,y);
//______________________________________________________________________________
   SENDACTION5(MoveResizeWindow,Window_t, id, Int_t,x, Int_t,y, UInt_t,w, UInt_t,h);
//______________________________________________________________________________
   SENDACTION3(ResizeWindow,Window_t, id, UInt_t,w, UInt_t,h);
//______________________________________________________________________________
   SENDACTION2(SetWindowBackground,Window_t, id, ULong_t,color);
//______________________________________________________________________________
   SENDACTION2(SetWindowBackgroundPixmap,Window_t, id, Pixmap_t,pxm);
//______________________________________________________________________________
   RETURNACTION11(Window_t, CreateWindow,Window_t, parent, Int_t,x, Int_t,y,
                                 UInt_t,w, UInt_t,h, UInt_t,border,
                                 Int_t,depth, UInt_t,clss,
                                 void *,visual, SetWindowAttributes_t *,attr,
                                 UInt_t,wtype);
//______________________________________________________________________________
  RETURNACTION1(Int_t,OpenDisplay,const char *,dpyName);
//______________________________________________________________________________
  SENDACTION0(CloseDisplay);
//______________________________________________________________________________
  RETURNACTION2(Atom_t,InternAtom,const char *,atom_name, Bool_t, only_if_exist);
//______________________________________________________________________________
//  RETURNACTION1(Window_t,GetParent,Window_t, id);
//______________________________________________________________________________
Window_t TQtThread::GetParent(Window_t /*id*/ ) const {return 0;}
//______________________________________________________________________________
  RETURNACTION1(FontStruct_t, LoadQueryFont,const char *,font_name);
//______________________________________________________________________________
  RETURNACTION1(FontH_t,GetFontHandle,FontStruct_t, fs);
//______________________________________________________________________________
  SENDACTION1(DeleteFont,FontStruct_t, fs);
//______________________________________________________________________________
  RETURNACTION2(GContext_t, CreateGC,Drawable_t, id, GCValues_t *,gval);
//______________________________________________________________________________
  VOIDACTION2(ChangeGC,GContext_t, gc, GCValues_t *,gval);
//______________________________________________________________________________
  VOIDACTION3(CopyGC,GContext_t, org, GContext_t, dest, Mask_t, mask);
//______________________________________________________________________________
  VOIDACTION1(DeleteGC,GContext_t, gc);
//______________________________________________________________________________
  RETURNACTION1(Cursor_t,CreateCursor,ECursor, cursor);
//______________________________________________________________________________
  SENDACTION2(SetCursor,Window_t, id, Cursor_t, curid);
//______________________________________________________________________________
  RETURNACTION3(Pixmap_t,CreatePixmap,Drawable_t,id, UInt_t,w, UInt_t,h);
//______________________________________________________________________________
  RETURNACTION7(Pixmap_t,CreatePixmap,Drawable_t, id, const char *,bitmap, UInt_t,width,
                                UInt_t,height, ULong_t, forecolor, ULong_t, backcolor,
                                Int_t,depth);
//______________________________________________________________________________
  RETURNACTION4(Pixmap_t,CreateBitmap,Drawable_t,id, const char *,bitmap,
                                UInt_t,width, UInt_t,height);
//______________________________________________________________________________
  SENDACTION1(DeletePixmap,Pixmap_t,pmap);
#ifdef BUG2DEBUG
//______________________________________________________________________________
  RETURNACTION5(Bool_t,CreatePictureFromFile,Drawable_t,id, const char *,filename,
                                              Pixmap_t &,pict, Pixmap_t &,pict_mask,
                                              PictureAttributes_t &,attr);
//______________________________________________________________________________
  RETURNACTION5(Bool_t,CreatePictureFromData,Drawable_t,id, char **,data,
                                              Pixmap_t &,pict, Pixmap_t &,pict_mask,
                                              PictureAttributes_t &,attr);
#endif
//______________________________________________________________________________
  RETURNACTION2(Bool_t,ReadPictureDataFromFile,const char *,filename, char ***,ret_data);
#if BUG2DEBUG
//______________________________________________________________________________
  SENDACTION1(DeletePictureData,void *,data);
#endif
//______________________________________________________________________________
  SENDACTION4(SetDashes,GContext_t,gc, Int_t,offset, const char *,dash_list,
                                  Int_t,n);
//______________________________________________________________________________
  RETURNACTION3(Bool_t,ParseColor,Colormap_t,cmap, const char *,cname, ColorStruct_t &,color);
//______________________________________________________________________________
  RETURNACTION2(Bool_t,AllocColor,Colormap_t,cmap, ColorStruct_t &,color);
//______________________________________________________________________________
  VOIDACTION2(QueryColor,Colormap_t,cmap, ColorStruct_t &,color);
//______________________________________________________________________________
//  RETURNACTION0(Int_t,EventsPending);
//______________________________________________________________________________
//  VOIDACTION1(NextEvent,Event_t &,event);
//______________________________________________________________________________
  SENDACTION1(Bell,Int_t,percent);
//______________________________________________________________________________
  VOIDACTION9(CopyArea,Drawable_t,src, Drawable_t,dest, GContext_t,gc,
                                 Int_t,src_x, Int_t,src_y, UInt_t,width,
                                 UInt_t,height, Int_t,dest_x, Int_t,dest_y);
//______________________________________________________________________________
  VOIDACTION2(ChangeWindowAttributes,Window_t, id, SetWindowAttributes_t *,attr);
#if BUG2DEBUG
//______________________________________________________________________________
  VOIDACTION5(ChangeProperty,Window_t, id, Atom_t,property, Atom_t,type,
                                       UChar_t *,data, Int_t,len);
#endif
//______________________________________________________________________________
  VOIDACTION6(DrawLine,Drawable_t,id, GContext_t,gc, Int_t,x1, Int_t,y1, Int_t,x2, Int_t,y2);
//______________________________________________________________________________
  SENDACTION5(ClearArea,Window_t, id, Int_t,x, Int_t,y, UInt_t,w, UInt_t,h);
//______________________________________________________________________________
  RETURNACTION3(Bool_t,CheckEvent,Window_t, id, EGEventType, type, Event_t &,evnt);
//______________________________________________________________________________
//  VOIDACTION2(SendEvent,Window_t, id, Event_t *,evnt);
//______________________________________________________________________________
  SENDACTION1(WMDeleteNotify,Window_t, id);
//______________________________________________________________________________
  SENDACTION1(SetKeyAutoRepeat,Bool_t,on);
//______________________________________________________________________________
  SENDACTION4(GrabKey,Window_t, id, Int_t,keycode, UInt_t,modifier, Bool_t,grab);
//______________________________________________________________________________
  VOIDACTION7(GrabButton,Window_t, id, EMouseButton, button, UInt_t,modifier,
                                   UInt_t,evmask, Window_t, confine, Cursor_t,cursor,
                                   Bool_t,grab);
//______________________________________________________________________________
  VOIDACTION6(GrabPointer,Window_t, id, UInt_t,evmask, Window_t, confine,
                                    Cursor_t,cursor, Bool_t,grab,
                                    Bool_t,owner_events);
//______________________________________________________________________________
  SENDACTION2(SetWindowName,Window_t, id, char *,name);
//______________________________________________________________________________
  SENDACTION2(SetIconName,Window_t, id, char *,name);
//______________________________________________________________________________
  SENDACTION3(SetClassHints,Window_t, id, char *,className, char *,resourceName);
//______________________________________________________________________________
  SENDACTION4(SetMWMHints,Window_t, id, UInt_t,value, UInt_t,funcs, UInt_t,input);
//______________________________________________________________________________
  SENDACTION3(SetWMPosition,Window_t, id, Int_t,x, Int_t,y);
//______________________________________________________________________________
  SENDACTION3(SetWMSize,Window_t, id, UInt_t,w, UInt_t,h);
//______________________________________________________________________________
  SENDACTION7(SetWMSizeHints,Window_t, id, UInt_t,wmin, UInt_t,hmin,
                             UInt_t, wmax, UInt_t,hmax, UInt_t,winc, UInt_t,hinc);
//______________________________________________________________________________
  SENDACTION2(SetWMState,Window_t, id, EInitialState, state);
//______________________________________________________________________________
  SENDACTION2(SetWMTransientHint,Window_t, id, Window_t, main_id);
//______________________________________________________________________________
  VOIDACTION6(DrawString,Drawable_t,id, GContext_t,gc, Int_t,x, Int_t,y,
                                   const char *,s, Int_t,len);
//______________________________________________________________________________
  RETURNACTION3(Int_t,TextWidth,FontStruct_t,font, const char *,s, Int_t,len);
//______________________________________________________________________________
  VOIDACTION3(GetFontProperties,FontStruct_t,font, Int_t&,max_ascent, Int_t&,max_descent);
//______________________________________________________________________________
  VOIDACTION2(GetGCValues,GContext_t,gc, GCValues_t &,gval);
//______________________________________________________________________________
//  RETURNACTION1(FontStruct_t,GetFontStruct,FontH_t,fh);
//______________________________________________________________________________
  SENDACTION1(ClearWindow,Window_t, id);
//______________________________________________________________________________
  RETURNACTION1(Int_t,KeysymToKeycode,UInt_t,keysym);
//______________________________________________________________________________
  VOIDACTION6(FillRectangle,Drawable_t,id, GContext_t,gc, Int_t,x, Int_t,y,
                                      UInt_t,w, UInt_t,h);
//______________________________________________________________________________
  VOIDACTION6(DrawRectangle,Drawable_t,id, GContext_t,gc, Int_t,x, Int_t,y,
                                      UInt_t,w, UInt_t,h);
//______________________________________________________________________________
  VOIDACTION4(DrawSegments,Drawable_t,id, GContext_t,gc, Segment_t *,seg, Int_t,nseg);
//______________________________________________________________________________
  SENDACTION2(SelectInput,Window_t, id, UInt_t,evmask);
//______________________________________________________________________________
  SENDACTION1(SetInputFocus,Window_t, id);
//______________________________________________________________________________
  RETURNACTION0(Window_t,GetPrimarySelectionOwner);
//______________________________________________________________________________
  SENDACTION1(SetPrimarySelectionOwner,Window_t, id);
//______________________________________________________________________________
  SENDACTION3(ConvertPrimarySelection,Window_t, id, Atom_t,clipboard, Time_t,when);
//______________________________________________________________________________
  SENDACTION4(LookupString,Event_t *,event, char *, buf, Int_t,buflen, UInt_t&,keysym);
//______________________________________________________________________________
  SENDACTION5(GetPasteBuffer,Window_t, id, Atom_t,atom, TString &,text, Int_t&,nchar,
                                       Bool_t,del);
//______________________________________________________________________________
  VOIDACTION7(TranslateCoordinates,Window_t, src, Window_t, dest, Int_t,src_x,
                         Int_t,src_y, Int_t&,dest_x, Int_t&,dest_y, Window_t &,child);
//______________________________________________________________________________
  VOIDACTION5(GetWindowSize,Drawable_t,id, Int_t&,x, Int_t&,y, UInt_t&,w, UInt_t&,h);
//______________________________________________________________________________
  SENDACTION4(FillPolygon,Window_t, id, GContext_t,gc, Point_t*,points, Int_t,npnt);
//______________________________________________________________________________
  VOIDACTION8(QueryPointer,Window_t, id, Window_t &,rootw, Window_t &,childw,
                                     Int_t&,root_x, Int_t&,root_y, Int_t&,win_x,
                                    Int_t&,win_y, UInt_t&,mask);
//______________________________________________________________________________
  VOIDACTION2(SetForeground,GContext_t,gc, ULong_t,foreground);
//______________________________________________________________________________
  VOIDACTION5(SetClipRectangles,GContext_t,gc, Int_t,x, Int_t,y, Rectangle_t *,recs, Int_t,n);
//______________________________________________________________________________
  SENDACTION1(Update,Int_t,mode);
