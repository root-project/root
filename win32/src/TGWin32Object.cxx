// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   10/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*The   G W i n 3 2 O b j e c t  class *-*-*-*-*-*-*-*-*
//*-*                =====================================
//*-*
//*-*  Basic interface to the WIN32 graphics system
//*-*
//*-*   This is an implemenation of the GWin32 Class.
//*-*
//*-*  It is done as separate file to work around of the clashes between
//*-*  MS macro names and Root names of GWin32 member functions
//*-*
//*-*  This code was initially developped in the context of HIGZ and PAW
//*-*  by Valery Fine to port the package X11INT (by Olivier Couet)
//*-*  to Windows NT.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#include "TGWin32Object.h"

#ifndef ROOT_TGWin32WindowsObject
#include "TGWin32WindowsObject.h"
#endif

#ifndef ROOT_TGWin32PixmapObject
#include "TGWin32PixmapObject.h"
#endif

#ifndef ROOT_TCanvas
#include "TCanvas.h"
#endif

#ifndef ROOT_TVirtualPad
#include "TVirtualPad.h"
#endif

#ifndef ROOT_TMath
#include "TMath.h"
#endif

#ifndef ROOT_TPoint
#include "TPoint.h"
#endif

#ifndef ROOT_TGWin32Brush
#include "TGWin32Brush.h"
#endif

#ifndef ROOT_TGWin32Pen
#include "TGWin32Pen.h"
#endif

// #include <windows.h>


// #include <wingdi.h>
#include <commctrl.h>
#undef GetTextAlign

#define CleanCommand(type) if (lParam) {delete (type *)lParam; lParam = 0;}

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#define SafeSwitchW32(_WinFunc) if (flpMirror) flpMirror->##_WinFunc ;    \
                                if (fMasterIsActive) flpMasterObject->##_WinFunc

#define ExecSwitchW32(_WinFunc,_Command) if (flpMirror) flpMirror->##_WinFunc ;    \
                                        if (fMasterIsActive)    \
                                        flpMasterObject->ExecCommand(##_Command)

//*-*  Only the Master object can be queried

#define ReturnQuery(_WinFunc)   return flpMasterObject ?                   \
                                       (flpMasterObject->##_WinFunc) : 0


// ClassImp(TGWin32Switch)

//______________________________________________________________________________
TGWin32Switch::TGWin32Switch(){
    fMasterIsActive = -1;
}

//______________________________________________________________________________
TGWin32Switch::TGWin32Switch(TGWin32Object *master, TGWin32Switch *mirror, Bool_t ownmaster){

//    fDoubleBuffer   = 0;
    fMasterIsActive = 1;
    flpMasterObject = master;
    flpMirror       = mirror;
    fOwnMasterFlag  = ownmaster;

}

//______________________________________________________________________________
TGWin32Switch::TGWin32Switch(TGWin32Object *master, Bool_t ownmaster){

//    fDoubleBuffer   = 0;
    fMasterIsActive = 1;
    flpMasterObject = master;
    fOwnMasterFlag  = ownmaster;
    flpMirror       = 0;
}

//______________________________________________________________________________
TGWin32Switch::~TGWin32Switch(){
    if (fMasterIsActive == -1 ) return;
    Delete();
}
//______________________________________________________________________________
void  TGWin32Switch::W32_Clear() {
    TGWin32Clear *CodeOp = new TGWin32Clear;
    ExecSwitchW32(W32_Clear() ,CodeOp);
}
//______________________________________________________________________________
void  TGWin32Switch::W32_Close()
    {  SafeSwitchW32(W32_Close());}
//______________________________________________________________________________
void  TGWin32Switch::W32_CopyTo(TGWin32Object *obj, int xpos, int ypos){
    SafeSwitchW32(W32_CopyTo(obj, xpos, ypos));
}
//______________________________________________________________________________
void  TGWin32Switch::W32_CreateOpenGL()
{
   SafeSwitchW32(W32_CreateOpenGL());
}
//______________________________________________________________________________
void  TGWin32Switch::W32_DeleteOpenGL()
{
//   TGWin32OpenGL *CodeOp = new TGWin32OpenGL(kDeleteGL);
//   ExecSwitchW32(W32_Clear() ,CodeOp);
}

//______________________________________________________________________________
void  TGWin32Switch::W32_DrawBox(int x1, int y1, int x2, int y2, TVirtualX::EBoxMode mode){
    TGWin32Box *CodeOp = new TGWin32Box (x1,y1,x2,y2,mode);
    ExecSwitchW32(W32_DrawBox(x1, y1, x2, y2, mode) ,CodeOp);
}
//______________________________________________________________________________
void  TGWin32Switch::W32_DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny, int *ic){
//*-*-*-*-*-*-*-*-*-*-*Draw a cell array*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================
//*-*  x1,y1        : left down corner
//*-*  x2,y2        : right up corner
//*-*  nx,ny        : array size
//*-*  ic           : array
//*-*
//*-*  Draw a cell array. The drawing is done with the pixel presicion
//*-*  if (X2-X1)/NX (or Y) is not a exact pixel number the position of
//*-*  the top rigth corner may be wrong.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    TGWin32Cell *CodeOp = new TGWin32Cell(x1, y1, x2, y2, nx, ny, ic);
    ExecSwitchW32(W32_DrawCellArray(x1, y1, x2, y2, nx, ny, ic) ,CodeOp);
}
//______________________________________________________________________________
void  TGWin32Switch::W32_DrawFillArea(int n, TPoint *xy){
      TGWin32DrawPolyLine *CodeOp = new TGWin32DrawPolyLine(n,(POINT *)xy,IX_FLARE);
      ExecSwitchW32(W32_DrawFillArea(n, xy),CodeOp);
}
//______________________________________________________________________________
void  TGWin32Switch::W32_DrawLine(int x1, int y1, int x2, int y2){
    POINT xy[2] = { {x1,y1}, {x2,y2}};
    TGWin32DrawPolyLine *CodeOp = new TGWin32DrawPolyLine(2,xy);

    ExecSwitchW32(W32_DrawLine(x1, y1, x2, y2),CodeOp);

 }
//______________________________________________________________________________
void  TGWin32Switch::W32_DrawPolyLine(int n, TPoint *xy)
    {
      TGWin32DrawPolyLine *CodeOp = new TGWin32DrawPolyLine(n,(POINT *)xy);
      ExecSwitchW32(W32_DrawPolyLine(n, xy),CodeOp);

    }

//______________________________________________________________________________
void  TGWin32Switch::W32_DrawPolyMarker(int n, TPoint *xy)
{
      TGWin32DrawPolyLine *CodeOp = new TGWin32DrawPolyLine(n,(POINT *)xy, IX_MARKE);
      ExecSwitchW32(W32_DrawPolyMarker(n, xy),CodeOp);
}
//______________________________________________________________________________
void  TGWin32Switch::W32_DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode mode)
{
      TGWin32DrawText *CodeOp = new TGWin32DrawText(x, y, text, mode);
      ExecSwitchW32(W32_DrawText(x, y, angle, mgn, text, mode),CodeOp);
}
//______________________________________________________________________________
void  TGWin32Switch::W32_GetCharacterUp(Float_t &chupx, Float_t &chupy)
    {SafeSwitchW32(W32_GetCharacterUp(chupx, chupy));}
//______________________________________________________________________________
Int_t TGWin32Switch::W32_GetDoubleBuffer()
    {ReturnQuery(W32_GetDoubleBuffer());}
//______________________________________________________________________________
void  TGWin32Switch::W32_GetGeometry(int &x, int &y, unsigned int &w, unsigned int &h)
    {SafeSwitchW32(W32_GetGeometry(x, y, w, h));}
//______________________________________________________________________________
void  TGWin32Switch::W32_GetPixel(int y, int width, Byte_t *scline)
    {SafeSwitchW32(W32_GetPixel(y, width, scline));}
//______________________________________________________________________________
void  TGWin32Switch::W32_GetRGB(int index, float &r, float &g, float &b){
#define BIGGEST_RGB_VALUE 255
      TGWin32GetColor *CodeOp = new TGWin32GetColor(index);
      ExecSwitchW32(W32_GetRGB(index, r, g, b),CodeOp);

      r = (float)CodeOp->Red()/BIGGEST_RGB_VALUE;
      g = (float)CodeOp->Green()/BIGGEST_RGB_VALUE;
      b = (float)CodeOp->Blue()/BIGGEST_RGB_VALUE;

      delete CodeOp;

//    {SafeSwitchW32(W32_GetRGB(index, r, g, b));}
}
//______________________________________________________________________________
void  TGWin32Switch::W32_GetTextExtent(unsigned int &w, unsigned int &h, char *mess){

//*-*-*-*-*-*-*-*-*-*-*Return the size of a character string*-*-*-*-*-*-*-*-*-*
//*-*                  =====================================
//*-*  iw          : text width
//*-*  ih          : text height
//*-*  mess        : message
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   TGWin32DrawText *CodeOp = new TGWin32DrawText(0, 0, mess,TVirtualX::kClear,IX_TXTL);
   ExecSwitchW32(W32_GetTextExtent(w, h, mess),CodeOp);
   w = CodeOp->GetX();
   h = CodeOp->GetY();
   delete CodeOp;
}
//______________________________________________________________________________
void  TGWin32Switch::W32_Move(Int_t x, Int_t y)
    {SafeSwitchW32(W32_Move(x, y));}
//______________________________________________________________________________
void  TGWin32Switch::W32_PutByte(Byte_t b)
    {SafeSwitchW32(W32_PutByte(b));}
//______________________________________________________________________________
void  TGWin32Switch::W32_QueryPointer(int &ix, int &iy)
    {SafeSwitchW32(W32_QueryPointer(ix, iy));}
//______________________________________________________________________________
Int_t TGWin32Switch::W32_RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y)
    {ReturnQuery(W32_RequestLocator(mode, ctyp, x, y));}
//______________________________________________________________________________
Int_t TGWin32Switch::W32_RequestString(int x, int y, char *text)
    {ReturnQuery(W32_RequestString(x, y, text));}
//______________________________________________________________________________
Int_t TGWin32Switch::W32_Rescale(int wid, unsigned int w, unsigned int h){

//*-*  Rescale may entail creating a new object wich is a copy of the previous one

//    ((TGWin32Object *)wid)->Rescale(w,h);

    if (flpMirror) flpMirror->W32_Rescale((int) flpMirror,w,h);
    if (flpMasterObject) {
      TGWin32Object *winobj = flpMasterObject->Rescale(w,h);
      if (winobj != flpMasterObject) {
        TGWin32Object *o = flpMasterObject;
        flpMasterObject = winobj; // link the new object
        delete o;                 // delete old one
        return 1;
      }
    }
    return 0;
}

//______________________________________________________________________________
void  TGWin32Switch::W32_Resize()
    {SafeSwitchW32(W32_Resize());}
//______________________________________________________________________________
void  TGWin32Switch::W32_Select()
    {SafeSwitchW32(W32_Select());}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetCharacterUp(Float_t chupx, Float_t chupy)
    {   printf(" W32_SetCharacterUp(..) hasn't been implemented yet, Sorry! \n");
        SafeSwitchW32(W32_SetCharacterUp(chupx, chupy));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetClipOFF(){

  TGWin32Clear *CodeOp = new TGWin32Clear(IX_NOCLI);
  ExecSwitchW32(W32_SetClipOFF(),CodeOp);
  delete CodeOp;

}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetClipRegion(int x, int y, unsigned int w, unsigned int h){

  TGWin32Clip *CodeOp = new TGWin32Clip(w, h, x, y);
  ExecSwitchW32(W32_SetClipRegion(x, y, w, h),CodeOp);

}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetCursor(ECursor cursor)
    {SafeSwitchW32(W32_SetCursor(cursor));}

//______________________________________________________________________________
    void      TGWin32Switch::W32_SetDoubleBuffer(int mode)
    {SafeSwitchW32(W32_SetDoubleBuffer(mode));}

//______________________________________________________________________________
    void      TGWin32Switch::W32_SetDoubleBufferOFF()
    {W32_SetDoubleBuffer(0);};

//______________________________________________________________________________
    void      TGWin32Switch::W32_SetDoubleBufferON()
    { W32_SetDoubleBuffer(1); }

//______________________________________________________________________________
void  TGWin32Switch::W32_SetDrawMode(TVirtualX::EDrawMode mode){
  TGWin32DrawMode *CodeOp = new TGWin32DrawMode(mode);
  ExecSwitchW32(W32_SetDrawMode(mode),CodeOp);
}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetFillColor(Color_t cindex)
    {   printf(" W32_SetFillColor(..) hasn't been implemented yet, Sorry! \n");
        SafeSwitchW32(W32_SetFillColor(cindex));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetFillStyle(Style_t style)
    {SafeSwitchW32(W32_SetFillStyle(style));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetLineColor(Color_t cindex)
    {SafeSwitchW32(W32_SetLineColor(cindex));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetLineType(int n, int *dash)
    {SafeSwitchW32(W32_SetLineType(n, dash));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetLineStyle(Style_t linestyle)
    {SafeSwitchW32(W32_SetLineStyle(linestyle));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetLineWidth(Width_t width)
    {SafeSwitchW32(W32_SetLineWidth(width));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetMarkerColor( Color_t cindex)
    {SafeSwitchW32(W32_SetMarkerColor(cindex));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetMarkerSize(Float_t markersize)
    {SafeSwitchW32(W32_SetMarkerSize(markersize));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetMarkerStyle(Style_t markerstyle)
    {SafeSwitchW32(W32_SetMarkerStyle(markerstyle));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetRGB(int cindex, float r, float g, float b)
    {SafeSwitchW32(W32_SetRGB(cindex, r, g, b));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetTextAlign(Short_t talign){
}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetTextColor(Color_t cindex)
    {SafeSwitchW32(W32_SetTextColor(cindex));}
//______________________________________________________________________________
Int_t TGWin32Switch::W32_SetTextFont(char *fontname, TVirtualX::ETextSetMode mode)
    {ReturnQuery(W32_SetTextFont(fontname, mode));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetTextFont(Int_t fontnumber)
    {SafeSwitchW32(W32_SetTextFont(fontnumber));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetTextSize(Float_t textsize)
    {SafeSwitchW32(W32_SetTextSize(textsize));}
//______________________________________________________________________________
void  TGWin32Switch::W32_SetTitle(const char *title)
    {SafeSwitchW32(W32_SetTitle(title));}
//______________________________________________________________________________
void  TGWin32Switch::W32_Update(int mode)
    {SafeSwitchW32(W32_Update(mode));}
//______________________________________________________________________________
void  TGWin32Switch::W32_Warp(int ix, int iy)
    {SafeSwitchW32(W32_Warp(ix, iy));}

//______________________________________________________________________________
void  TGWin32Switch::W32_WriteGIF(char *name)
    {SafeSwitchW32(W32_WriteGIF(name));}
//______________________________________________________________________________
void  TGWin32Switch::W32_WritePixmap(unsigned int w, unsigned int h, char *pxname)
    {SafeSwitchW32(W32_WritePixmap(w, h, pxname));}

//______________________________________________________________________________
void TGWin32Switch::Delete()
{
   if (flpMirror)       SafeDelete(flpMirror);
//*-*  Delete the windows object too if this owns flpMasterObject
   if (flpMasterObject && fOwnMasterFlag) SafeDelete(flpMasterObject);
}
//______________________________________________________________________________
TGWin32Object *TGWin32Switch::GetMasterObject()
{
 return flpMasterObject;
}

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*       T G W i n 3 2 O b j e c t   i m p l e m e n t a t i o n
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

// ClassImp(TGWin32Object)

//______________________________________________________________________________
TGWin32Object::TGWin32Object()
{
  fTypeFlag = -1;
}

//______________________________________________________________________________
TGWin32Object::~TGWin32Object(){

  if (fTypeFlag == -1) return;
  Delete();
}

//______________________________________________________________________________
void TGWin32Object::Delete(){

  if (fObjectClipRegion) { DeleteObject(fObjectClipRegion); fObjectClipRegion = 0; }
  if (fObjectDC)         { DeleteDC(fObjectDC); fObjectDC = 0; }
  if (fBufferObj)        { delete fBufferObj; fBufferObj = 0; }

}

//______________________________________________________________________________
LRESULT  TGWin32Object::CallCallback(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
    return fWinAction(hwnd, uMsg, wParam, lParam);
}
//______________________________________________________________________________
int TGWin32Object::CharWidth(char ch)
{
    int SizeOfChar;
    GetCharWidth32(fObjectDC, (UINT) ch, (UINT) ch,  &SizeOfChar);
    return SizeOfChar;
 }

//______________________________________________________________________________
void TGWin32Object::Win32CreateCallbacks()
{

// fWinAction[(UINT)WM_LBUTTONDOWN] = TGWin32WindowsObject::OnMouseButton;
//  fWinAction[WM_MBUTTONDOWN] = &OnMouseButton;
//  fWinAction[WM_RBUTTONDOWN] = &OnMouseButton;

//  fWinAction[WM_MOUSEMOVE]   = &OnMouseButton;
//  fWinAction[WM_LBUTTONUP]   = &OnMouseButton;
//  fWinAction[WM_MBUTTONUP]   = &OnMouseButton;
//  fWinAction[WM_RBUTTONUP]   = &OnMouseButton;


 fWinAction.AddCallBack(WM_LBUTTONDOWN,(CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);
 fWinAction.AddCallBack(WM_MBUTTONDOWN,(CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);
 fWinAction.AddCallBack(WM_RBUTTONDOWN,(CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);

 fWinAction.AddCallBack(WM_LBUTTONUP,  (CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);
 fWinAction.AddCallBack(WM_MBUTTONUP,  (CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);
 fWinAction.AddCallBack(WM_RBUTTONUP,  (CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);

 fWinAction.AddCallBack(WM_LBUTTONDBLCLK,(CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);
 fWinAction.AddCallBack(WM_MBUTTONDBLCLK,(CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);
 fWinAction.AddCallBack(WM_RBUTTONDBLCLK,(CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);

 fWinAction.AddCallBack(WM_MOUSEMOVE,  (CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);
 fWinAction.AddCallBack(WM_MOUSEACTIVATE,  (CallBack_t)TGWin32WindowsObject::OnMouseActivateCB,this);
 fWinAction.AddCallBack(WM_CONTEXTMENU,(CallBack_t)TGWin32WindowsObject::OnMouseButtonCB,this);

 fWinAction.AddCallBack(WM_PAINT,      (CallBack_t)TGWin32WindowsObject::OnPaintCB,this);

 fWinAction.AddCallBack(WM_SETFOCUS,   (CallBack_t)TGWin32WindowsObject::OnSetFocusCB,this);
 fWinAction.AddCallBack(WM_KILLFOCUS,  (CallBack_t)TGWin32WindowsObject::OnKillFocusCB,this);
 fWinAction.AddCallBack(WM_CHAR,       (CallBack_t)TGWin32WindowsObject::OnCharCB,this);
 fWinAction.AddCallBack(WM_KEYDOWN,    (CallBack_t)TGWin32WindowsObject::OnKeyDownCB,this);


//-

  fWinAction.AddCallBack(WM_ACTIVATE,  (CallBack_t)TGWin32WindowsObject::OnActivateCB,this);
  fWinAction.AddCallBack(WM_CREATE,    (CallBack_t)TGWin32WindowsObject::OnCreateCB,this);
  fWinAction.AddCallBack(WM_CLOSE,     (CallBack_t)TGWin32WindowsObject::OnCloseCB,this);
  fWinAction.AddCallBack(WM_COMMAND,   (CallBack_t)TGWin32WindowsObject::OnCommandCB,this);
  fWinAction.AddCallBack(WM_ERASEBKGND,(CallBack_t)TGWin32WindowsObject::OnEraseBkgndCB,this);
  fWinAction.AddCallBack(WM_EXITSIZEMOVE, (CallBack_t)TGWin32WindowsObject::OnExitSizeMoveCB,this);

  fWinAction.AddCallBack(WM_PALETTECHANGED, (CallBack_t)TGWin32WindowsObject::OnPaletteChangedCB,this);

  fWinAction.AddCallBack(WM_SIZE,      (CallBack_t)TGWin32WindowsObject::OnSizeCB,this);
  fWinAction.AddCallBack(WM_SIZING,    (CallBack_t)TGWin32WindowsObject::OnSizingCB,this);
  fWinAction.AddCallBack(WM_GETMINMAXINFO, (CallBack_t)TGWin32WindowsObject::OnGetMinMaxInfoCB,this);
  fWinAction.AddCallBack(WM_DESTROY,   (CallBack_t)TGWin32WindowsObject::OnSysCommandCB,this);
  fWinAction.AddCallBack(WM_NOTIFY,    (CallBack_t)TGWin32WindowsObject::OnNotifyCB,this);

  fWinAction.AddCallBack(IX11_ROOT_MSG,(CallBack_t)TGWin32Object::OnRootActCB,this); // TVirtualX Draw operations
  fWinAction.AddCallBack(IX11_ROOT_Input,(CallBack_t)TGWin32WindowsObject::OnRootInputCB,this); // TVirtualX Get Locator
  fWinAction.AddCallBack(ROOT_HOOK,    (CallBack_t)TGWin32WindowsObject::OnRootHookCB,this);    // To call mathod via WinProc

}

//______________________________________________________________________________
void TGWin32Object::W32_CreateOpenGL()
{
#if 0
    if (fObjectDC && !fOpenGLRC)
    {
        fOpenGLRC = new TVirtualXOpenGL();
        fOpenGLRC->CreateContext((Int_t)(fWin32Mother->GetSelectedWindow()));
        fOpenGLRC->MakeCurrent();
    }
#endif
}

//______________________________________________________________________________
void TGWin32Object::W32_Select()
{
//    if (fOpenGLRC)
//        fOpenGLRC->MakeCurrent();
}

//______________________________________________________________________________
LRESULT APIENTRY
                 TGWin32Object::OnRootAct(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  DWORD rvalue;
  HGDIOBJ hbr,hpn;
  HDC hdc;

//*-*   Check, whether the double buffer mode was set for the object

//*-*  Very special case CopyTo is buffered even fDoubleBuffer == 0;
  if (fBufferObj &&
      (    fDoubleBuffer && ((TGWin32Command *)lParam)->GetBuffered()
      ||  ((TGWin32Command *)lParam)->GetBuffered() == -1 ) )
  {
      //                       fBufferObjSec.WriteLock();
      LRESULT res = fBufferObj->OnRootAct(hwnd,uMsg, wParam, lParam);
      //                       fBufferObjSec.ReleaseWriteLock();
      return res;
  }

  if (fWin32Mother->fhdCommonPalette)
  {
      //            SetSystemPaletteUse(fObjectDC,SYSPAL_NOSTATIC);

#if 0
            SetSystemPaletteUse(fObjectDC,SYSPAL_STATIC);
#endif
//            printf(" Common palette = %d DC=%x\n",RealizePalette(fObjectDC),fObjectDC);
            HPALETTE hPal = SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,FALSE);
//            HPALETTE hPal = SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,TRUE);
            if (hPal != fWin32Mother->fhdCommonPalette) DeleteObject(hPal);

            RealizePalette(fObjectDC);
  }
  hbr = SelectObject(fObjectDC,fWin32Mother->fWin32Brush->GetBrush());
  hpn = SelectObject(fObjectDC,fWin32Mother->fWin32Pen->GetWin32Pen());

  rvalue = (DWORD)RootAct(hwnd,uMsg, wParam, lParam);

//*-*                  Restore all attributes

  SelectObject(fObjectDC,hpn);
  SelectObject(fObjectDC,hbr);
  return rvalue;

}

//______________________________________________________________________________
LRESULT APIENTRY
                 TGWin32Object::RootAct(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{

  switch (HIWORD(wParam)) {

     case ROOT_Control:
         switch (LOWORD(wParam))
            {
              case IX_OPNDS:  // Open WIN32 display
                       break;
              case IX_OPNWI:  // Open WIN32 window
                       break;
              case IX_SELWI:  // Select the current X11 window
                        break;
              case IX_CLSWI:  // Close an WIN32 window
                    CleanCommand(TGWin32Command);
//*-*  First delete the double buffer and clear fDoubleBuffer flag

                   if (fBufferObj) {
                     SafeDelete(fBufferObj);
                     fDoubleBuffer = 0;
                   }
                   DestroyWindow( hwnd );
                   return 0;
               case IX_CLSDS:  // Close an WIN32 session
                   DestroyWindow( hwnd );
                   PostQuitMessage(0);
                   break;

               case IX_SETHN:  // Set WIN32 host name
                     return TRUE;
               case IX_CLRWI:  // Clear an WIN32 object
                   {
                     HRGN TempClip;
                     RECT rect;
                     HBRUSH CurBrush = CreateSolidBrush(WHITE_ROOT_COLOR);


//*-*                To clear object one has to cancel clipping temporary

                      if (fObjectClipRegion) SelectClipRgn(fObjectDC,NULL);
//*-*
//*-*               One should distiguish a real window and a pixmap
//*-*
                      if (hwnd) {
                        GetClientRect(hwnd,&rect);
                        DPtoLP(fObjectDC,(POINT *)(&rect),2);}
                      else {
                        BITMAP  Bitmap_buffer;
                        GetObject(((TGWin32PixmapObject *)this)->GetBitmap(), sizeof(BITMAP),&Bitmap_buffer);
                        rect.left   = 0;
                        rect.top    = 0;
                        rect.right  = Bitmap_buffer.bmWidth;
                        rect.bottom = Bitmap_buffer.bmHeight;
                      }

                      FillRect(fObjectDC,&rect, CurBrush );
                      DeleteObject(CurBrush);

//*-*               Reset clipping if any

                      if (fObjectClipRegion) SelectClipRgn(fObjectDC,fObjectClipRegion);

                    return TRUE;
                    }

               case IX_RSCWI:  // Resize an ROOT window
                     return TRUE;
               case IX_SETBUF:
                     fDoubleBuffer = ((TGWin32SetDoubleBuffer *)lParam)->GetBuffer();
                     CleanCommand(TGWin32SetDoubleBuffer);
                     return TRUE;
                     break;
               case IX_SETSTATUS:
                    {
                   // Create the status window.
                     HWND shwnd = CreateWindowEx(
                                          0,                       // no extended styles
                                          STATUSCLASSNAME,         // name of status window class
                                          (LPCTSTR) NULL,          // no text when first created
                                          SBARS_SIZEGRIP |         // includes a sizing grip
                                          WS_CHILD,                // creates a child window
                                          0, 0, 0, 0,              // ignores size and position
                                          hwnd,                    // handle to parent window
                                          (HMENU)ID_STATUSBAR,     // child window identifier
                                          GetWin32ObjectMother()->GetWin32Instance(),  // handle to the application instance
                                          NULL);                    // no window creation data
                      ((TGWin32CreateStatusBar *)lParam)->SetWindow(shwnd);
                      ShowWindow(shwnd,SW_HIDE);
                      ((TGWin32CreateStatusBar *)lParam)->Release();
                      return 0;
                     }
               case IX_GETBUF:
                     ((TGWin32GetDoubleBuffer *)lParam)->SetBuffer(fDoubleBuffer);
                     ((TGWin32GetDoubleBuffer *)lParam)->Release();
                     return TRUE;
               case IX_CLIP :  // Define the X11 clipping rectangle
                    {
                     LPPOINT ClipRectPoint;
                     LPRECT  ClipRectFromPoint;

                     ClipRectPoint     = (LPPOINT) lParam;
                     ClipRectFromPoint = (LPRECT) lParam;
                     LPtoDP(fObjectDC,ClipRectPoint, 2);

                     if (fObjectClipRegion) {
                                      DeleteObject(fObjectClipRegion);
                                      fObjectClipRegion = 0;
                     }
//*-*   Select clippping for Display */
                     SelectClipRgn(fObjectDC,
                                   fObjectClipRegion = CreateRectRgnIndirect( ClipRectFromPoint));
                     return (LRESULT) TRUE;
                    }
                case IX_NOCLI:  // Deactivate the ROOT clipping rectangle
                    {
                     HDC     hdc;

                     if (fObjectClipRegion) {
                        DeleteObject(fObjectClipRegion);
                        SelectClipRgn(fObjectDC,NULL);
                        fObjectClipRegion = (HRGN) NULL;
                     }
                     return (LRESULT) TRUE;
                    }
                 default:
                    return TRUE;
               }
             break;


//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      ROOT output graphics primitives
//*-*
     case  ROOT_Primitive:
        switch (LOWORD(wParam))
        {
//*-*
//*-*   Draw PolyLine
//*-*
         case IX_LINE :      // Draw a line through all points
          {
           int n            = ((TGWin32DrawPolyLine *)lParam)->GetNumber();
           POINT *lpTPoint = ((TGWin32DrawPolyLine *)lParam)->GetPoints();

           CleanCommand(TGWin32DrawPolyLine);

           if ( n > 1 )
               return  Polyline(fObjectDC,(CONST POINT *)lpTPoint,n);
           else
               return SetPixelV(fObjectDC,lpTPoint->x,lpTPoint->y,
                                fWin32Mother->fWin32Pen->GetColor());
          }

//*-*
//*-*   Draw ROOT Markers
//*-*
          case IX_MARKE:      // Draw a marker at each point
               return  Wnd_MARKE(hwnd, uMsg, wParam, lParam);

//*-*
//*-*   Draw ROOT Mafilled area described with a polygon
//*-*
          case IX_FLARE:      // Fill area described by polygon
             return Wnd_FLARE(hwnd, uMsg, wParam, lParam);
//*-*
//*-*   Draw ROOT Box
//*-*
          case IX_BOX  :      // Draw a box
                return Wnd_BOX(hwnd, uMsg, wParam, lParam);
//*-*
//*-*   Draw ROOT cell array
//*-*
          case IX_CA   :      // Draw a cell array
               return Wnd_CA(hwnd, uMsg, wParam, lParam);
          default:
                return TRUE;
            }
            break;
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*   ROOT output  text  primitives
//*-*
     case  ROOT_Text:
         {
           TEXTMETRIC *tm;
           HGDIOBJ hdf;

//*-*      Save default values and set the current one

           hdf    = SelectObject(fObjectDC,fWin32Mother->fhdCommonFont);

//*-*      Update text metric's

           tm = (TEXTMETRIC *)malloc(sizeof(TEXTMETRIC));
              GetTextMetrics(fObjectDC,tm);
              fdwCharX = tm->tmAveCharWidth;
              fdwCharY = tm->tmHeight;
              fdwAscent= tm->tmAscent;
           free(tm);

           switch (LOWORD(wParam))
             {
              case IX_TEXT :      // Draw a text string using the current font
               {
                 int x        = ((TGWin32DrawText *)lParam)->GetX();
                 int y        = ((TGWin32DrawText *)lParam)->GetY();
                 const char *lpText = ((TGWin32DrawText *)lParam)->GetText();
                 TVirtualX::ETextMode mode = ((TGWin32DrawText *)lParam)->GetMode();

                 CleanCommand(TGWin32DrawText);

                 int y_shift = 0,
                     x_shift = 0;
                 double t_rotate;

//*-*            Save default values  and set current one

                 COLORREF clrref = SetTextColor(fObjectDC,ROOTColorIndex(fWin32Mother->fTextColor));
                 int txtA   = SetTextAlign(fObjectDC,fWin32Mother->fdwCommonTextAlign);

//*-*            therefore Windows font hasn't a  "vertical center attribute"
//*-*                        one should emulate it by hand

                 if (fWin32Mother->fTextAlignV == 2) {
                     t_rotate = cos(0.1*((fWin32Mother->fROOTFont).lfEscapement));
                     y_shift  = (fdwCharY/2)*t_rotate;
                     x_shift  = (fdwCharY/2)*sqrt(1.0-t_rotate*t_rotate);
                 }
                 Int_t savemode = SetBkMode(fObjectDC,mode == TVirtualX::kOpaque ? OPAQUE : TRANSPARENT);
                 ExtTextOut(fObjectDC,
                            x-x_shift, y-y_shift,
                            0, NULL,
                            lpText, strlen(lpText),
                            NULL);

//*-*                     Restore default values

                 SetBkMode(fObjectDC,savemode);
                 SetTextColor(fObjectDC,clrref);
                 SetTextAlign(fObjectDC,txtA );
                 break;
                }

              case IX_TXTL :      // Return the width and height of character string in the current font
                 {
                   const char *lpText = ((TGWin32DrawText *)lParam)->GetText();

                   SIZE text_size;
                   GetTextExtentPoint(fObjectDC,
                                      lpText,strlen(lpText), &text_size);
                   ((TGWin32DrawText *)lParam)->SetSize(&text_size);
                   break;
                  }
                 default:
                           break;
               }

//*-*                     Restore Default font

              SelectObject(fObjectDC,hdf);

              tm = (TEXTMETRIC *)malloc(sizeof(TEXTMETRIC));
                 GetTextMetrics(fObjectDC,tm);
                 fdwCharX = tm->tmAveCharWidth;
                 fdwCharY = tm->tmHeight;
                 fdwAscent= tm->tmAscent;
              free(tm);

              break;

            }

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*     ROOT output attributes
//*-*
      case ROOT_Attribute:
        switch (LOWORD(wParam))
         {
           case IX_DRMDE:          // Set drawing mode
            {
             int mode   = ((TGWin32DrawMode *)lParam)->GetMode();
             SetROP2(fObjectDC,mode);
             CleanCommand(TGWin32DrawMode);
             return TRUE;
            }
           case IX_SETMENU:
             HMENU menu = ((TGWin32AddMenu *)lParam)->GetMenuHandle();
             CleanCommand(TGWin32AddMenu);
             TGWin32WindowsObject *wobj = (TGWin32WindowsObject *)this;

        //     RECT rect;
        //     GetClientRect(hwnd,&rect);
             if (!SetMenu(hwnd,menu)) {
                Int_t err = GetLastError();
                Printf("SetMenu error  %d ", err);
             }
             else {
//*-*
//*-*        Correct the window size to hold the coming menu bar
//*-*
          //   ClientToScreen(hwnd,(POINT *)&rect.left);
          //   ClientToScreen(hwnd,(POINT *)&rect.right);
          //   AdjustWindowRectEx(&rect,wobj->fDwStyle,TRUE,wobj->fDwExtStyle);
          //   wobj->W32_Set(rect.left,rect.top,rect.right-rect.left+1,rect.bottom-rect.top+1);
             }
            return TRUE;
         }
         break;
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*     ROOT marker style
//*-*
      case ROOT_Marker:
        switch (LOWORD(wParam))
        {
       case IX_SYNC :      // ROOT synchronization

       default:
             return TRUE;
       }
                   break;
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      ROOT inquiry routines
//*-*
       case ROOT_Inquiry:
         switch (LOWORD(wParam))
          {
           case IX_GETGE:       // Returns position and size of Window
             if (hwnd) GetClientRect(hwnd,(LPRECT)lParam);
             return TRUE;
           case IX_GETWI:       // Returns the X11 window identifier
           case IX_GETPL:       // Returns the maximal number of planes of the display
           case IX_GETCOL:      // Returns the X11 colour representation
             {
              int ci          = ((TGWin32GetColor *)lParam)->GetCIndex();
              PALETTEENTRY  *lpRGB = ((TGWin32GetColor *)lParam)->GetPalPointer();

              HPALETTE hpl = (HPALETTE)GetCurrentObject(fObjectDC,OBJ_PAL);
              GetPaletteEntries(hpl,ci,1,lpRGB);
              break;
             }
           default:
             return TRUE;
          }
        break;

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      Pixmap manipulation
//*-*
      case ROOT_Pixmap:
        switch (LOWORD(wParam))
         {
           case IX_UPDWI:  // Update an ROOT window
             if (!fDoubleBuffer)  break;
           case IX_CPPX :       // Copy the pixmap
             Wnd_CPPX(hwnd, uMsg, wParam, lParam);
             break;
           case IX_WRPX :       // Write the pixmap
//           Wnd_WRPX(hwnd, uMsg, wParam, lParam);
             break;
           case IX_WIPX :       // Copy the area in the current window
//==>             Wnd_WIPX(hwnd, uMsg, wParam, lParam);
             break;
           default:
                return TRUE;
         }
         return TRUE;
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*           OpenGL command

     case ROOT_OpenGL:
         switch (LOWORD(wParam))
         {
              case GL_MAKECURRENT:  // Make GL context the current one
#if 0
                  printf("fOpenGLRC = %x \n", fOpenGLRC);
                  if (fOpenGLRC)
                  {
                     Bool_t res = wglMakeCurrent(fObjectDC,fOpenGLRC->GetRC());
                     if (!res) printf(" Error: TGWin32Object::MakeCurrent Error code =  %d, Thread id = %d \n",
                     GetLastError(),GetCurrentThreadId());
                  }
                  ((TGWin32GLCommand *)lParam)->Release();
#endif

//                      fOpenGLRC->MakeCurrentCB();
                  break;
              default:
                  return TRUE;
         }
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*           Dummies

      case ROOT_Dummies:
         switch (LOWORD(wParam))
         {
           case IX_S2BUF:
//==>             Wnd_S2BUF(hwnd, uMsg, wParam, lParam);
             break;
           case IX_SDSWI:
           default:
                     return TRUE;
          }
       default:
                     return TRUE;
    }
            return TRUE;
  }   /*  ROOT_Act */

//______________________________________________________________________________
 LRESULT APIENTRY
       TGWin32Object::Wnd_BOX(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
 {
  RECT box;

  box.left   = ((TGWin32Box *)lParam)->GetX1();
  box.bottom = ((TGWin32Box *)lParam)->GetY1();
  box.right  = ((TGWin32Box *)lParam)->GetX2();
  box.top    = ((TGWin32Box *)lParam)->GetY2();

  int mode = ((TGWin32Box *)lParam)->GetMode();

  HGDIOBJ NewBrush, lpCurPen;
  COLORREF CurTextColor;

  CleanCommand(TGWin32Box);

  if ((box.right - box.left) + (box.top - box.bottom) <= 2)
  {
      TGWin32Brush *lpBrush = fWin32Mother->fWin32Brush;
      SetPixelV(fObjectDC,box.left, box.bottom,lpBrush->GetColor() );
  }
  else if (mode == 0) {
     if (box.left == box.right | box.bottom == box.top) {
//*-*          Draw a line  instead the rectangle            */
         MoveToEx(fObjectDC,box.left,box.bottom,NULL);
         LineTo  (fObjectDC,box.right,box.top);
     }
     else {
         HBRUSH CurBrush = (HBRUSH) SelectObject(fObjectDC, GetStockObject(HOLLOW_BRUSH));
         Rectangle(fObjectDC,box.left, box.bottom, box.right, box.top);
         DeleteObject(SelectObject(fObjectDC,CurBrush));
     }
  }
  else {
//                             CurBrush = GetCurrentObject(CurrentDC,OBJ_BRUSH);

     TGWin32Brush *lpBrush = fWin32Mother->fWin32Brush;

     if (lpBrush->GetStyle() == BS_PATTERN) {
 //        CurTextColor = SetTextColor(fObjectDC,lpBrush->GetColor());
         CurTextColor = SetBkColor(fObjectDC,lpBrush->GetColor());

         FillRect(fObjectDC, &box, lpBrush->GetBrush());
 //        SetTextColor(fObjectDC,CurTextColor);
         SetBkColor(fObjectDC,CurTextColor);
     }
     else
         FillRect(fObjectDC, &box, lpBrush->GetBrush());
  }
  return TRUE;
 }

//______________________________________________________________________________
 LRESULT APIENTRY
       TGWin32Object::Wnd_CA(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
   {
    int i,j,icol,ix,iy,w,h,hh,current_icol;

    int x1  = ((TGWin32Cell *)lParam)->GetX1();
    int y1  = ((TGWin32Cell *)lParam)->GetY1();
    int x2  = ((TGWin32Cell *)lParam)->GetX2();
    int y2  = ((TGWin32Cell *)lParam)->GetY2();

    int nx  = ((TGWin32Cell *)lParam)->GetNx();
    int ny  = ((TGWin32Cell *)lParam)->GetNy();

    int *ic =((TGWin32Cell *)lParam)->GetCells();

    HBRUSH CurCABrush = NULL;

    CleanCommand(TGWin32Cell);

    current_icol = -1;
    w            = TMath::Max((x2-x1)/(nx),1);
    h            = TMath::Max((y1-y2)/(ny),1);
    ix           = x1;

    if (w+h == 2)
    {
//*-*  The size of the box is equal a single pixel
        for ( i=x1; i<x1+nx; i++){
            for (j = 0; j<ny; j++){
                icol = ic[i+(nx*j)];
                SetPixelV(fObjectDC,i,y1+j,ROOTColorIndex(icol));
            }
        }
    }
    else
    {
//*-* The shape of the box is a rectangle
        RECT box;
        box.bottom = y1;
        box.top    = y1;
        box.left   = x1;
        box.right  = box.left+h;
        for ( i=0; i<nx; i++ ) {
            box.top -= h;
            for ( j=0; j<ny; j++ ) {
                icol = ic[i+(nx*j)];
                if(icol != current_icol){
                    if (CurCABrush != NULL) DeleteObject(CurCABrush);
                    CurCABrush = CreateSolidBrush(ROOTColorIndex(icol));
                    current_icol = icol;
                }
                FillRect(fObjectDC, &box, CurCABrush);
                box.bottom = box.top;
                box.top -= h;
            }
            box.left = box.right;
            box.right += w;
        }
    }

    DeleteObject(CurCABrush);
    return TRUE;
   }

//______________________________________________________________________________
LRESULT APIENTRY
         TGWin32Object::Wnd_CPPX(HWND  hwnd, UINT   uMsg, WPARAM wParam, LPARAM lParam)
{

//*-*  Define the target object

   TGWin32Object *WinSource = ((TGWin32CopyTo *)lParam)->GetSource();
//*-*  WinSource = 0 is a special case, try fBufferObj;
   if (!WinSource) WinSource = fBufferObj;
   if (!WinSource) return TRUE;  // Nothing to copy

   HDC hDC = WinSource->fObjectDC;
   POINT *TargetPosition = ((TGWin32CopyTo *)lParam)->GetPointsTo();
   POINT *SourcePosition = ((TGWin32CopyTo *)lParam)->GetPointsFrom();

   HBITMAP hb;

//*-* Define the size of the source object;

   if (WinSource->GetObjectType())    // Source is a pixmap
     hb = ((TGWin32PixmapObject *)WinSource)->GetBitmap();

   else if (GetObjectType())    // Target is a pixmap
     hb = ((TGWin32PixmapObject *)this)->GetBitmap();

   BITMAP Bitmap_buffer;
   GetObject(hb, sizeof(BITMAP),&Bitmap_buffer);

   int w = Bitmap_buffer.bmWidth;
   int h = Bitmap_buffer.bmHeight;
   BitBlt(fObjectDC,TargetPosition->x,TargetPosition->y, w,h,
                hDC,SourcePosition->x,SourcePosition->y,SRCCOPY);

   CleanCommand(TGWin32CopyTo);

   return TRUE;
}
//______________________________________________________________________________
 LRESULT APIENTRY
       TGWin32Object::Wnd_FLARE(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
 {
   int n           = ((TGWin32DrawPolyLine *)lParam)->GetNumber();
   POINT *lpTPoint = ((TGWin32DrawPolyLine *)lParam)->GetPoints();
   TGWin32Brush *lpBrush = fWin32Mother->fWin32Brush;

   COLORREF CurTextColor;
   HGDIOBJ lpCurPen;
   int bord = 0;  // must be changed !!!

   CleanCommand(TGWin32DrawPolyLine);

   if (!bord & lpBrush->GetStyle() != BS_HOLLOW)
               lpCurPen = SelectObject(fObjectDC, GetStockObject(NULL_PEN));


   if (lpBrush->GetStyle()  == BS_PATTERN) {
 //            CurTextColor = SetTextColor(fObjectDC,lpBrush->GetColor());
             CurTextColor = SetBkColor(fObjectDC,lpBrush->GetColor());
             Polygon(fObjectDC,lpTPoint,n);
 //            SetTextColor(fObjectDC,CurTextColor);
             CurTextColor = SetBkColor(fObjectDC,CurTextColor);
   }
   else {
             Polygon(fObjectDC,lpTPoint,n);
   }
   if (!bord & lpBrush->GetStyle() != BS_HOLLOW)
             DeleteObject(SelectObject(fObjectDC,lpCurPen));

   return TRUE;
}

//______________________________________________________________________________
 LRESULT APIENTRY
      TGWin32Object::Wnd_MARKE(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
  {
  int n     = ((TGWin32DrawPolyLine *)lParam)->GetNumber();
  POINT *xy = ((TGWin32DrawPolyLine *)lParam)->GetPoints();
  POINT *mxy = fWin32Mother->fWin32Marker->GetNodes();

  TGWin32Marker *CurMarker = fWin32Mother->fWin32Marker;

  int m;
  COLORREF mColor;
  int ownBrush;

  CleanCommand(TGWin32DrawPolyLine);

                  /* Set marker Color */
  mColor  = ROOTColorIndex(fWin32Mother->GetMarkerColor());

  if( CurMarker->GetNumber() <= 0 )
     for (m=0; m < n; m++)  SetPixelV(fObjectDC, xy[m].x,xy[m].y, mColor);

  else {
    int r = CurMarker->GetNumber() / 2;
    HGDIOBJ  CurBrush, CurPen;

    CurPen   = SelectObject(fObjectDC, CreatePen(PS_SOLID,0,mColor));

    switch (CurMarker -> GetType()) {
      case 1:
      case 3:
     default:
          ownBrush = TRUE;
          CurBrush = SelectObject(fObjectDC, CreateSolidBrush(mColor));
          break;
      case 0:
      case 2:
          ownBrush = TRUE;
          CurBrush = SelectObject(fObjectDC, GetStockObject(HOLLOW_BRUSH));
          break;
      case 4:
          ownBrush = FALSE;
          break;
      }

    for( m = 0; m < n; m++ ) {
      int i;

      switch( CurMarker->GetType() ) {

      case 0:        /* hollow circle */
      case 1:        /* filled circle */
         Ellipse( fObjectDC,
              xy[m].x - r, xy[m].y - r,
              xy[m].x + r, xy[m].y + r);
         break;

      case 2:        /* hollow polygon */
      case 3:        /* filled polygon */
        for( i = 0; i < CurMarker->GetNumber(); i++ ) {
            mxy[i].x += xy[m].x;
            mxy[i].y += xy[m].y;
                                        }
        Polygon(fObjectDC,CurMarker->GetNodes(),CurMarker->GetNumber());
        for( i = 0; i < CurMarker->GetNumber(); i++ ) {
          mxy[i].x -= xy[m].x;
          mxy[i].y -= xy[m].y;
         }
      break;

      case 4:        /* segmented line */
      for( i = 0; i < CurMarker->GetNumber(); i += 2 )
       {
        MoveToEx(fObjectDC,xy[m].x + mxy[i].x, xy[m].y + mxy[i].y,NULL);
        LineTo(fObjectDC,xy[m].x + mxy[i+1].x, xy[m].y + mxy[i+1].y);
       }
       break;
      }
    }


    if (ownBrush) DeleteObject(SelectObject(fObjectDC, CurBrush));
    DeleteObject( SelectObject(fObjectDC, CurPen ));
   }
   return (LRESULT)TRUE;
 }



#ifndef WIN32
    virtual void      W32_SetCharacterUp(Float_t chupx, Float_t chupy) = 0;
    virtual void      W32_SetClipOFF() = 0;
    virtual void      W32_SetClipRegion(int x, int y, unsigned int w, unsigned int h) = 0;
    virtual void      W32_SetCursor(ECursor cursor) = 0;
    virtual void      W32_SetDoubleBuffer(int mode) = 0;
    virtual void      W32_SetDoubleBufferOFF() = 0;
    virtual void      W32_SetDoubleBufferON()  = 0;
    virtual void      W32_SetDrawMode(TVirtualX::EDrawMode mode)  = 0;
    virtual void      W32_SetFillColor(Color_t cindex) = 0;
    virtual void      W32_SetFillStyle(Style_t style)  = 0;
#endif
//______________________________________________________________________________
void  TGWin32Object::W32_SetLineColor(Color_t cindex){
   Error("W32_SetLineColor","ROOT error");
}

//______________________________________________________________________________
void  TGWin32Object::W32_SetLineType(int n, int *dash){
   Error("W32_SetLineType","ROOT error");
}

//______________________________________________________________________________
void TGWin32Object::W32_SetLineStyle( Style_t lstyle )
{
//*-*-*-*-*-*-*-*-*-*-*Set line style*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Error("W32_SetLineStyle","ROOT error");
}

//______________________________________________________________________________
void  TGWin32Object::W32_SetLineWidth(Width_t width){
   Error("W32_SetLineWidth","ROOT error");
}

#ifndef WIN32
    virtual void      W32_SetMarkerColor( Color_t cindex) = 0;
    virtual void      W32_SetMarkerSize(Float_t markersize) = 0;
    virtual void      W32_SetMarkerStyle(Style_t markerstyle) = 0;
    virtual void      W32_SetTextAlign(Short_t talign=11) = 0;
    virtual void      W32_SetTextColor(Color_t cindex) = 0;
    virtual Int_t     W32_SetTextFont(char *fontname, TVirtualX::ETextSetMode mode) = 0;
    virtual void      W32_SetTextFont(Int_t fontnumber) = 0;
    virtual void      W32_SetTextSize(Float_t textsize) = 0;
#endif

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

// ClassImp(TGWin32PixmapObject)

//______________________________________________________________________________
TGWin32PixmapObject::TGWin32PixmapObject() : TGWin32Object() {;}

//______________________________________________________________________________
HBITMAP TGWin32PixmapObject::CreateDIBSurface(HDC hWndDC,Int_t w, Int_t h)
{
//***********************************************************************
// Function:    CreateDIBSurface
//
// Purpose:             creates a DIB section as the drawing surface for gl calls
//
// Parameters:
//        HDC                           Device Context to create the DIBSection for
//
// Returns:
//        HBITMAP
//
//**********************************************************************

   return    CreateCompatibleBitmap(hWndDC, w,  h);

#define WIDTHBYTES(bits)  (((bits) + 31)/32 * 4)

        HBITMAP    hBmRet = NULL;
//      BITMAPINFO *pbi = (BITMAPINFO *)render.biInfo;
        BITMAPINFO pbi;
    LPVOID     lpBits;
        ZeroMemory(&pbi, sizeof(BITMAPINFO));

        if (!hWndDC) return NULL;

    pbi.bmiHeader.biSize           = sizeof(BITMAPINFOHEADER);
    pbi.bmiHeader.biWidth          = w;
    pbi.bmiHeader.biHeight         = h;
        pbi.bmiHeader.biPlanes         = 1;
    pbi.bmiHeader.biBitCount       = GetDeviceCaps(hWndDC, PLANES) * GetDeviceCaps(hWndDC, BITSPIXEL);
    pbi.bmiHeader.biCompression    = BI_RGB;
    pbi.bmiHeader.biSizeImage      = WIDTHBYTES((DWORD)pbi.bmiHeader.biWidth * pbi.bmiHeader.biBitCount) * pbi.bmiHeader.biHeight;

    printf(" BitCount = %d, SizeImage = %d \n", pbi.bmiHeader.biBitCount,pbi.bmiHeader.biSizeImage);
    hBmRet = CreateDIBSection(hWndDC, &pbi, DIB_RGB_COLORS,
                                        &lpBits, NULL, (DWORD)0);

    int ierr = GetLastError();
    printf(" hWndDC %x \n",hWndDC);
    printf(" Create bitmap for %d thread with bitmap %x ierr =  %d \n", GetCurrentThreadId(),hBmRet,ierr);
    if (hBmRet==0) printf(" error %d \n", GetLastError());
        return hBmRet;
#undef WIDTHBYTES
}

//______________________________________________________________________________
TGWin32PixmapObject::TGWin32PixmapObject(TGWin32 *lpTGWin32, UInt_t w, UInt_t h){

///////////////////////////////////////////////////////////////////
//    Creat a new pixmap.                                        //
//   Int_t w,h : Width and height of the pixmap.                 //
///////////////////////////////////////////////////////////////////


  fTypeFlag     = 0;
  fWin32Mother  = lpTGWin32;
  fIsPixmap     = 1;
  fModified     = 1;

  fDoubleBuffer = 0;
  fBufferObj    = 0;

  fWinSize.cx  = w;
  fWinSize.cy  = h;

  TGWin32Object *winobj = fWin32Mother->GetMasterObjectbyId(0);
//  fObjectDC = CreateCompatibleDC(winobj->GetWin32DC());
  fObjectDC = CreateCompatibleDC(NULL);

//*-*
//*-*  Set and adjust a client area of the window object
//*-*

  SetMapMode (fObjectDC,MM_ISOTROPIC);

  SetWindowExtEx(fObjectDC, fWinSize.cx, fWinSize.cy, NULL);

  SetBkMode  (fObjectDC,TRANSPARENT);
  SetTextAlign(fObjectDC,TA_BASELINE | TA_LEFT | TA_NOUPDATECP);

  SetViewportExtEx (fObjectDC, fWinSize.cx, fWinSize.cy,  NULL);


//*-* Look for a real Screen DC to make pixmap compatible.

  fRootPixmap = CreateDIBSurface(winobj->GetWin32DC(), fWinSize.cx, fWinSize.cy);

//*-*  The version below generates the black/white images
//*-*  fRootPixmap = CreateDIBSurface(fObjectDC, fWinSize.cx, fWinSize.cy);

  if (fRootPixmap)
  {

    SelectObject(fObjectDC,fRootPixmap);

    if (fWin32Mother->fhdCommonPalette)
    {
//      UINT ncolor = SetDIBColorTable(fObjectDC, 0, fWin32Mother->fMaxCol,
//                                      (RGBQUAD *)(fWin32Mother->flpPalette->palPalEntry));
//        printf(" The %d colors have been set. error = %d  \n", ncolor, GetLastError());

//      HGDIOBJ hPal = SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,TRUE);
      HGDIOBJ hPal = SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,FALSE);
      if (!hPal) printf("Error SelectPalette for Pixmap - %d \n", GetLastError());
      else
      {
          if (hPal != fWin32Mother->fhdCommonPalette) DeleteObject(hPal);
          RealizePalette(fObjectDC);
      }
    }
  }
  else {
    printf("*** TGWin32PixmapObject: Create Bitmap error - %d  %x\n", GetLastError(),fRootPixmap);
    W32_Close();
  }
}

//______________________________________________________________________________
TGWin32PixmapObject::~TGWin32PixmapObject()
{
    if (fTypeFlag == -1) return;
    W32_Close();
}
//______________________________________________________________________________
void  TGWin32PixmapObject::W32_Clear()
{
RECT rect;
BITMAP Bitmap_buffer;
HBRUSH CurBrush;

    GetObject(fRootPixmap, sizeof(BITMAP),&Bitmap_buffer);
    rect.left   = 0;
    rect.top    = 0;
    rect.right  = Bitmap_buffer.bmWidth-1;
    rect.bottom = Bitmap_buffer.bmHeight-1;
    FillRect(fObjectDC,
             &rect, CurBrush = CreateSolidBrush(WHITE_ROOT_COLOR));
    DeleteObject(CurBrush);
}

//______________________________________________________________________________
void  TGWin32PixmapObject::W32_Close(){
     if (fRootPixmap) {DeleteObject(fRootPixmap); fRootPixmap = 0; }
     Delete();
}
//______________________________________________________________________________
void  TGWin32PixmapObject::W32_CopyTo(TGWin32Object *wobj, int xpos, int ypos){
    Printf(" **** TGWin32PixmapObject::W32_CopyTo is not implemented \n");
    return;
    TGWin32CopyTo *CodeOp = new TGWin32CopyTo(wobj,xpos,ypos,0,0);

    SendMessage(((TGWin32WindowsObject *)wobj)->GetWindow(),
                  IX11_ROOT_MSG,
                  (WPARAM)CodeOp->GetCOP(),
                  (LPARAM)(CodeOp));

}

//______________________________________________________________________________
void  TGWin32PixmapObject::W32_DrawBox(int x1, int y1, int x2, int y2, TVirtualX::EBoxMode mode){ }
void  TGWin32PixmapObject::W32_DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny, int *ic){ }
void  TGWin32PixmapObject::W32_DrawFillArea(int n, TPoint *xy){ }
void  TGWin32PixmapObject::W32_DrawLine(int x1, int y1, int x2, int y2){ }
void  TGWin32PixmapObject::W32_DrawPolyLine(int n, TPoint *xy){ }
void  TGWin32PixmapObject::W32_DrawPolyMarker(int n, TPoint *xy){ }
void  TGWin32PixmapObject::W32_DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode mode){ }
void  TGWin32PixmapObject::W32_GetCharacterUp(Float_t &chupx, Float_t &chupy){ }
Int_t TGWin32PixmapObject::W32_GetDoubleBuffer(){return fDoubleBuffer;}
//______________________________________________________________________________
void  TGWin32PixmapObject::W32_GetGeometry(int &x, int &y, unsigned int &w, unsigned int &h){
   x = fPosition.x;  // Take the origin of the pixmap
   y = fPosition.y;

   BITMAP Bitmap_buffer;
   GetObject(fRootPixmap, sizeof(BITMAP),&Bitmap_buffer);

   w = Bitmap_buffer.bmWidth;
   h = Bitmap_buffer.bmHeight;

}
//______________________________________________________________________________
void  TGWin32PixmapObject::W32_GetPixel(int y, int width, Byte_t *scline){ }
void  TGWin32PixmapObject::W32_GetRGB(int index, float &r, float &g, float &b){ }
void  TGWin32PixmapObject::W32_GetTextExtent(unsigned int &w, unsigned int &h, char *mess){ }
void  TGWin32PixmapObject::W32_Move(Int_t x, Int_t y){ }
void  TGWin32PixmapObject::W32_PutByte(Byte_t b){ }
void  TGWin32PixmapObject::W32_QueryPointer(int &ix, int &iy){ }
Int_t TGWin32PixmapObject::W32_RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y){return 0;}
Int_t TGWin32PixmapObject::W32_RequestString(int x, int y, char *text){return 0;}
//______________________________________________________________________________
TGWin32Object *TGWin32PixmapObject::Rescale(unsigned int w, unsigned int h){
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Resize a pixmap*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          ===============
//*-*  w,h : Width and height of the pixmap.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//*-*  Check the new size agaist the old one

       if ( fWinSize.cx >= w-1 && fWinSize.cx <= w+1 &&
            fWinSize.cy >= h-1 && fWinSize.cy <= h+1) return this; // Do nothing

//*-*  Create a new pixmap object

       TGWin32PixmapObject *newpixmap = new  TGWin32PixmapObject(fWin32Mother,w,h);
//       newpixmap->SetOpenGLRC(fOpenGLRC);

//*-*   ATTENTION !!! We don't delete the old pixmap in here and it still occupies the memory

//*-*  Copy old one into a new one

//*-*  This operation is useless so far  ==>  ((TGWin32Object *)newpixmap)->W32_CopyTo(this,w,h);

//*-*  Change the Viewport of the OpenGL view if any
//       if (fOpenGLRC)
//                fOpenGLRC->SetViewPort(0,0,w,h);

       return newpixmap;

}
//______________________________________________________________________________
void  TGWin32PixmapObject::W32_Resize(){ }
//______________________________________________________________________________
void  TGWin32PixmapObject::W32_Select()
{
#if 0
    if (fOpenGLRC) {
         fOpenGLRC->MakeCurrent();
    }
#endif
}

//______________________________________________________________________________
void  TGWin32PixmapObject::W32_SetCharacterUp(Float_t chupx, Float_t chupy){ }
void  TGWin32PixmapObject::W32_SetClipOFF(){ }
void  TGWin32PixmapObject::W32_SetClipRegion(int x, int y, unsigned int w, unsigned int h){ }
void  TGWin32PixmapObject::W32_SetCursor(ECursor cursor){ }
void  TGWin32PixmapObject::W32_SetDoubleBuffer(int mode){ }
void  TGWin32PixmapObject::W32_SetDoubleBufferOFF(){ }
void  TGWin32PixmapObject::W32_SetDoubleBufferON(){ }
void  TGWin32PixmapObject::W32_SetDrawMode(TVirtualX::EDrawMode mode){ }
void  TGWin32PixmapObject::W32_SetFillColor(Color_t cindex){ }
void  TGWin32PixmapObject::W32_SetFillStyle(Style_t style){ }
void  TGWin32PixmapObject::W32_SetLineColor(Color_t cindex){ }
void  TGWin32PixmapObject::W32_SetLineType(int n, int *dash){ }
void  TGWin32PixmapObject::W32_SetLineStyle(Style_t linestyle){ }
void  TGWin32PixmapObject::W32_SetLineWidth(Width_t width){ }
void  TGWin32PixmapObject::W32_SetMarkerColor( Color_t cindex){ }
void  TGWin32PixmapObject::W32_SetMarkerSize(Float_t markersize){ }
void  TGWin32PixmapObject::W32_SetMarkerStyle(Style_t markerstyle){ }
void  TGWin32PixmapObject::W32_SetRGB(int cindex, float r, float g, float b){ }
void  TGWin32PixmapObject::W32_SetTextAlign(Short_t talign){ }
void  TGWin32PixmapObject::W32_SetTextColor(Color_t cindex){ }
Int_t TGWin32PixmapObject::W32_SetTextFont(char *fontname, TVirtualX::ETextSetMode mode){return 0;}
void  TGWin32PixmapObject::W32_SetTextFont(Int_t fontnumber){ }
void  TGWin32PixmapObject::W32_SetTextSize(Float_t textsize){ }
void  TGWin32PixmapObject::W32_SetTitle(const char *title){ }
void  TGWin32PixmapObject::W32_Update(int mode)
{
#if 0
    if (fOpenGLRC)
         fOpenGLRC->W32_Update(0);
#endif
}

void  TGWin32PixmapObject::W32_Warp(int ix, int iy){ }
void  TGWin32PixmapObject::W32_WriteGIF(char *name){ }
void  TGWin32PixmapObject::W32_WritePixmap(unsigned int w, unsigned int h, char *pxname){ }

Int_t TGWin32PixmapObject::ExecCommand(TGWin32Command *code){

//*-*  To be universal we should call CallCallback function, but this appoach more fast.
 Modified();
 return OnRootAct(NULL, IX11_ROOT_MSG,
                    (WPARAM)code->GetCOP(),
                    (LPARAM)code);
}

HBITMAP  TGWin32PixmapObject::GetBitmap(){
//*-* Return the Handle of the BITMAP data structure
   return fRootPixmap;
}

void  TGWin32PixmapObject::Win32CreateObject(){ }

