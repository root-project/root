// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtThread.cxx,v 1.12 2004/07/30 14:12:07 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

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

ClassImp(TQtThread);

//______________________________________________________________________________
TQtThread::TQtThread(){;}
//______________________________________________________________________________
TQtThread::TQtThread(const Text_t *name, const Text_t *title) : TGQt(name, title)
{
  CreateQtApplicationImp();
  Init();
}
//______________________________________________________________________________
TQtThread::~TQtThread(){;}
//______________________________________________________________________________
    RETURNACTION1(Bool_t,Init,void *,display);
//______________________________________________________________________________
    VOIDACTION0(ClearWindow);
//______________________________________________________________________________
    VOIDACTION0(ClosePixmap);
//______________________________________________________________________________
    VOIDACTION0(CloseWindow);
//______________________________________________________________________________
    VOIDACTION3(CopyPixmap,Int_t, wid, Int_t, xpos, Int_t, ypos);
//______________________________________________________________________________
    SENDACTION1(CreateOpenGLContext,Int_t, wid);    // Create OpenGL context for win windows (for "selected" Window by default)
//______________________________________________________________________________
    SENDACTION1(DeleteOpenGLContext,Int_t,wid);    // Create OpenGL context for win windows (for "selected" Window by default)
//______________________________________________________________________________
    VOIDACTION5(DrawBox,Int_t, x1, Int_t, y1, Int_t, x2, Int_t, y2, TVirtualX::EBoxMode, mode);
//______________________________________________________________________________
    VOIDACTION7(DrawCellArray,Int_t,x1, Int_t,y1, Int_t,x2, Int_t,y2, Int_t,nx, Int_t,ny, Int_t*,ic);
//______________________________________________________________________________
    VOIDACTION2(DrawFillArea,Int_t,n, TPoint *,xy);
//______________________________________________________________________________
    VOIDACTION4(DrawLine,Int_t,x1, Int_t,y1, Int_t,x2, Int_t,y2);
//______________________________________________________________________________
    VOIDACTION2(DrawPolyLine,Int_t,n, TPoint *,xy);
//______________________________________________________________________________
    VOIDACTION2(DrawPolyMarker,Int_t,n, TPoint *,xy);
//______________________________________________________________________________
    VOIDACTION6(DrawText,Int_t,x, Int_t,y, Float_t, angle, Float_t, mgn, const char *,text, TVirtualX::ETextMode, mode);
//______________________________________________________________________________
    VOIDACTION2(GetCharacterUp,Float_t &,chupx, Float_t &,chupy);
//______________________________________________________________________________
    RETURNACTION1(Int_t,GetDoubleBuffer,Int_t,wid);
//______________________________________________________________________________
    VOIDACTION5(GetGeometry,Int_t, wid, Int_t &,x, Int_t &,y, UInt_t &,w, UInt_t &,h);
//______________________________________________________________________________
    const char *TQtThread::DisplayName(const char *n) { return TGQt::DisplayName(n); }
//______________________________________________________________________________
    Handle_t  TQtThread::GetNativeEvent() const { return  TGQt::GetNativeEvent();}
//______________________________________________________________________________
    RETURNACTION1(ULong_t,GetPixel,Color_t,cindex);
//______________________________________________________________________________
    void TQtThread::GetPlanes(Int_t &nplanes){ TGQt::GetPlanes(nplanes);}
//______________________________________________________________________________
    void TQtThread::GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b)
    { TGQt::GetRGB(index,r,g,b); }
//______________________________________________________________________________
    VOIDACTION3(GetTextExtent,UInt_t&,w, UInt_t&,h, char *,mess);
//______________________________________________________________________________
    Float_t TQtThread::GetTextMagnitude(){ return TGQt::GetTextMagnitude() ;} 
//______________________________________________________________________________
    Bool_t  TQtThread::HasTTFonts() const { return HasTTFonts() ;}
//______________________________________________________________________________
    RETURNACTION1(Int_t,InitWindow,ULong_t, window);
//______________________________________________________________________________
    RETURNACTION3(Int_t,AddWindow,ULong_t, qwid, UInt_t, w, UInt_t, h);
//______________________________________________________________________________
    VOIDACTION1(RemoveWindow,ULong_t, qwid);
//______________________________________________________________________________
    VOIDACTION3(MoveWindow,Int_t,wid, Int_t,x, Int_t,y);
//______________________________________________________________________________
    RETURNACTION2(Int_t,OpenPixmap,UInt_t,w, UInt_t,h);
//______________________________________________________________________________
    VOIDACTION1(PutByte,Byte_t,b);
//______________________________________________________________________________
    VOIDACTION2(QueryPointer,Int_t&,ix, Int_t&,iy);
//______________________________________________________________________________
    RETURNACTION4(Pixmap_t,ReadGIF,Int_t, x0, Int_t, y0, const char *,file, Window_t, id);
//______________________________________________________________________________
    RETURNACTION4(Int_t,RequestLocator,Int_t,mode, Int_t,ctyp, Int_t&,x, Int_t&,y);
//______________________________________________________________________________
    RETURNACTION3(Int_t,RequestString,Int_t,x,Int_t,y,char *,text);
//______________________________________________________________________________
    VOIDACTION3(RescaleWindow,Int_t,wid, UInt_t,w, UInt_t,h);
//______________________________________________________________________________
    RETURNACTION3(Int_t,ResizePixmap,Int_t,wid, UInt_t,w, UInt_t,h);
//______________________________________________________________________________
    VOIDACTION1(ResizeWindow,Int_t,wid);
//______________________________________________________________________________
    VOIDACTION1(SelectWindow,Int_t,wid);
//______________________________________________________________________________
    void  TQtThread::SelectPixmap(Int_t qpixid){ TGQt::SelectPixmap(qpixid );}
//______________________________________________________________________________
    VOIDACTION2(SetCharacterUp,Float_t, chupx, Float_t, chupy);
//______________________________________________________________________________
    VOIDACTION1(SetClipOFF,Int_t,wid);
//______________________________________________________________________________
    VOIDACTION5(SetClipRegion,Int_t,wid, Int_t,x, Int_t,y, UInt_t,w, UInt_t,h);
//______________________________________________________________________________
    SENDACTION2(SetCursor,Int_t,win, ECursor, cursor);
//______________________________________________________________________________
    VOIDACTION2(SetDoubleBuffer,Int_t,wid, Int_t,mode);
//______________________________________________________________________________
    void  TQtThread::SetDoubleBufferOFF(){ TGQt::SetDoubleBufferOFF(); }
//_____________________________________________________________________________
    void  TQtThread::SetDoubleBufferON() { TGQt::SetDoubleBufferON();  }

//______________________________________________________________________________
    VOIDACTION1(SetDrawMode,TVirtualX::EDrawMode,mode);
//______________________________________________________________________________
    VOIDACTION1(SetFillColor,Color_t, cindex);
//______________________________________________________________________________
    VOIDACTION1(SetFillStyle,Style_t, style);
//______________________________________________________________________________
    VOIDACTION2(SetFillStyleIndex, Int_t,style, Int_t,fasi);
//______________________________________________________________________________
    VOIDACTION1(SetLineColor,Color_t, cindex);
//______________________________________________________________________________
    VOIDACTION2(SetLineType,Int_t,n, Int_t *,dash);
//______________________________________________________________________________
    VOIDACTION1(SetLineStyle,Style_t, linestyle);
//______________________________________________________________________________
    VOIDACTION1(SetLineWidth,Width_t,width);
//______________________________________________________________________________
    VOIDACTION1(SetMarkerColor, Color_t, cindex);
//______________________________________________________________________________
    VOIDACTION1(SetMarkerSize,Float_t, markersize);
//______________________________________________________________________________
    VOIDACTION1(SetMarkerStyle,Style_t, markerstyle);
//______________________________________________________________________________
    VOIDACTION3(SetMarkerType, Int_t,type, Int_t,n, TPoint *,xy );
//______________________________________________________________________________
    VOIDACTION4(SetRGB,Int_t,cindex, Float_t, r, Float_t, g, Float_t, b);
//______________________________________________________________________________
    VOIDACTION1(SetTextAlign,Short_t, talign);
//______________________________________________________________________________
    VOIDACTION1(SetTextColor,Color_t, cindex);
//______________________________________________________________________________
    RETURNACTION2(Int_t,SetTextFont,char *,fontname, TVirtualX::ETextSetMode ,mode);
//______________________________________________________________________________
    VOIDACTION1(SetTextFont,Font_t, fontnumber);
//______________________________________________________________________________
    VOIDACTION1(SetTextMagnitude,Float_t, mgn);
//______________________________________________________________________________
    VOIDACTION1(SetTextSize,Float_t, textsize);
//______________________________________________________________________________
    VOIDACTION1(SetTitle,const char *,title);
//______________________________________________________________________________
    VOIDACTION1(UpdateWindow,Int_t,mode);
//______________________________________________________________________________
    VOIDACTION3(Warp,Int_t,ix, Int_t,iy, Window_t,id);
//______________________________________________________________________________
    RETURNACTION1(Int_t,WriteGIF,char *,name);
//______________________________________________________________________________
    VOIDACTION4(WritePixmap,Int_t,wid, UInt_t,w, UInt_t,h, char *,pxname);

//______________________________________________________________________________
    RETURNACTION1(Int_t,LoadQt, const char *,shareLibFileName);
//______________________________________________________________________________
UInt_t   TQtThread::ExecCommand(TGWin32Command *code)
 { return TGQt::ExecCommand(code); }

//______________________________________________________________________________
bool TQtThread::event(QEvent *e)
{
  if (e) {
    if (e->type() < QEvent::User) return FALSE;
    ((TQtEvent *)e)->ExecuteCB();
  }
  return TRUE;
}

