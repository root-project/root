// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   11/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGWin32WindowsObject.h"

#include "TGWin32PixmapObject.h"
#include "TCanvas.h"
#include "TCanvasImp.h"
#include "TWin32Canvas.h"
#include "TROOT.h"
#include "TGWin32StatusBar.h"
#include "Buttons.h"
#include "TInterpreter.h"
#include "TWin32SimpleEditCtrl.h"

// ClassImp(TGWin32WindowsObject)

////////////////////////////////////////////////////////////////////
//                                                                //
//  TGWin32WindowsObject                                          //
//                                                                //
//  It defines behaviour of the INTERACTIVE objects of WIN32 GDI  //
//  For instance "real" windows                                   //
//                                                                //
////////////////////////////////////////////////////////////////////


//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*  T G W i n 3 2 W i n d o w s O b j e c t   i m p e m e n t a t i o n
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//______________________________________________________________________________
 LRESULT APIENTRY TGWin32WindowsObject::OnRootInput(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
  {
   switch (LOWORD(wParam)) {
      case IX_REQLO:
        return OnRootMouse(hwnd,uMsg, wParam, lParam);
      case IX_REQST:      // Request a string input
        return OnRootTextInput(hwnd, uMsg, wParam, lParam);
      default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
      }

  }
//______________________________________________________________________________
 LRESULT APIENTRY TGWin32WindowsObject::OnRootMouse(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
  {
//     SetSystemPaletteUse(fObjectDC,SYSPAL_NOSTATIC);
//     RealizePalette(fObjectDC);

     fButton_Press =  0;
     fButton_Up    =  0;
     switch (uMsg) {
       case WM_RBUTTONUP:
         fButton_Up++;
       case WM_MBUTTONUP:
         fButton_Up++;
       case WM_LBUTTONUP:
         fButton_Up++;
         fButton_Press++;
       case WM_RBUTTONDOWN:
         fButton_Press++;
       case WM_MBUTTONDOWN:
         fButton_Press++;
       case WM_LBUTTONDOWN:
         fButton_Press++;
       case WM_MOUSEMOVE:

              /* Clean an old position */

        DrawROOTCursor(flpROOTMouse->GetType());

             /* Plot new cursor position */

        fLoc.x = (LONG) (MAKEPOINTS(lParam).x);
        fLoc.y = (LONG) (MAKEPOINTS(lParam).y);
        DPtoLP(fObjectDC,&fLoc,1);
        DrawROOTCursor(flpROOTMouse->GetType());

  /*  Reset system cursor near the bord id frame */

         if (flpROOTMouse->GetMode()) {
           if      (fButton_Press == 0)
                       fButton_Press = -1;
           else if (fButton_Press == 4)
                       fButton_Press = fButton_Up+10;
           RestoreROOT(flpROOTMouse->GetType());
           flpROOTMouse->SetXY(&fLoc);
           flpROOTMouse->SetButton(fButton_Press);

//           RestoreDC(CurrentDC,-1);
           flpROOTMouse->Release();

         }
         else if (fButton_Press > 0 & fButton_Up ==0) {
           RestoreROOT(flpROOTMouse->GetType());
           flpROOTMouse->SetXY(&fLoc);
           flpROOTMouse->SetButton(fButton_Press);
           flpROOTMouse->Release();
          }

         break;
       case IX11_ROOT_Input:
         flpROOTMouse = (TGWin32GetLocator *)lParam;
         fLoc.x  = flpROOTMouse->GetX(); fLoc.y  = flpROOTMouse->GetY();
         fLocp.x = flpROOTMouse->GetX(); fLocp.y = flpROOTMouse->GetY();

         ROOTCursorInit(hwnd,flpROOTMouse->GetType());
         break;
       default:
         break;
       }
        return 0;
}
//______________________________________________________________________________
void TGWin32WindowsObject::DrawROOTCursor(int ctyp)
{
 int radius, CurMxX, CurMxY;
 POINT loc,locp;

 loc.x = fLoc.x;
 loc.y = fLoc.y;

 locp.x = fLocp.x;
 locp.y = fLocp.y;

 CurMxX = fWin32WindowSize.right;
 CurMxY = fWin32WindowSize.bottom;

 SetCursor(fWin32Mother->fCursors[kPointer]);
 switch ( ctyp ) {

    case 1 :   //  Default ROOT window cursor is CROSS  -> do nothing here */
            break;

    case 2 : MoveToEx(fObjectDC,0,     loc.y,NULL);
             LineTo  (fObjectDC,CurMxX,loc.y);

             MoveToEx(fObjectDC,loc.x,0,    NULL);
             LineTo  (fObjectDC,loc.x,CurMxY);

             break;

    case 3 : radius = (int) sqrt((double)((loc.x-locp.x)*(loc.x-locp.x)+
                                          (loc.y-locp.y)*(loc.y-locp.y)));
             Pie(fObjectDC,locp.x-radius,locp.y-radius,
                           locp.x+radius,locp.y+radius,
                           locp.x-radius,locp.y-radius,
                           locp.x-radius,locp.y-radius);
             break;

     case 4 : MoveToEx(fObjectDC,loc.x, loc.y, NULL);
              LineTo  (fObjectDC,locp.x,locp.y);
              break;

     case 5 : Rectangle(fObjectDC,locp.x, locp.y, loc.x, loc.y);
              break;
     default:
             break;
 }
}

//______________________________________________________________________________
void TGWin32WindowsObject::ROOTCursorInit(HWND hwnd, int ctyp){

  SaveDC(fObjectDC);

//*-*  Create Brush or Pen to draw ROOT graphics cursor

   SelectObject(fObjectDC,fWin32Mother->fhdCursorPen);
   if (ctyp == 3 | ctyp == 5)
           SelectObject(fObjectDC, fWin32Mother->fhdCursorBrush);

//*-*  Suspend current clipping

   SelectClipRgn(fObjectDC, NULL);


//*-*  Set a new mix mode to XOR

   SetROP2(fObjectDC,R2_NOT);

//*-*  Set Cursor on the screen

   DrawROOTCursor(ctyp);
   fMouseInit = ctyp;
}

//______________________________________________________________________________
void TGWin32WindowsObject::RestoreROOT(int ctyp)
  {
    fMouseInit = 0;
//*-*  Clean cursor off the screen
    DrawROOTCursor(ctyp);
    RestoreDC(fObjectDC,-1);
  }

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnRootEditInput
         (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
#define ixg(x) (fix0+x)
#define iyg(y) (fiy0+y)
  int i;

  if (uMsg != IX11_ROOT_Input) return 0;

  flpROOTString = (TGWin32GetString *)lParam;

//*-*       Write  TEXT to the pointed position on the screen
  fix0 =  flpROOTString->GetX();
  fiy0 =  flpROOTString->GetY();
  fiXText = 0;
  fiYText = 0;
//*-*        Init of string input
 // flpInstr = StrDup(flpROOTString->GetTextPointer());
  flpInstr = (char *)flpROOTString->GetTextPointer();
  fnCur    = 0;
  flStr    = 0;
  fInsert  = 1;

  if (!(flpInstr && lstrlen(flpInstr)))
  {
          flpROOTString->Release();
 //         delete [] flpInstr; flpInstr = 0;
          return 0;
  }
  fLenLine = 0;

  if (fEditCtrl) {
          UnRegisterControlItem(fEditCtrl);
          delete fEditCtrl;
//          delete [] flpInstr; flpInstr = 0;
          fEditCtrl = 0;
  }
  // Convert from the pixel to the relative cooridunat
  UInt_t iw,ih;
  Float_t x,y,w,h;
  RECT winsize;
  GetClientRect(fhwndRootWindow,&winsize);
  x = Float_t(fix0)/winsize.right;
//           y = Float_t(winsize.bottom-fiy0)/winsize.bottom -(fdwAscent);
  y = (Float_t(fiy0)/winsize.bottom);
  gVirtualX->GetTextExtent(iw,ih,flpInstr);
  w = Float_t(iw)/winsize.right;
  h = Float_t(ih)/winsize.bottom;
  // Clean buffer. Delete all right blanks in there
  int len_text = flpInstr ? lstrlen(flpInstr) : 0;
  if (len_text > 0 ) {
          int il=len_text;
          while (il && (flpInstr[il-1] == ' ' || flpInstr[il-1] == 0))
          { flpInstr[il] = 0; il--;}
  }
  // set size in WM_SIZE message
  fEditCtrl = new TWin32SimpleEditCtrl(this,flpInstr,len_text,x,y,w,h);

  //*-*  Set text font

  SendMessage(fEditCtrl->GetWindow(),WM_SETFONT, (WPARAM) fWin32Mother->fhdCommonFont, FALSE);

  /* Add text to the window. */

  SendMessage(fEditCtrl->GetWindow(), WM_SETTEXT, 0, (LPARAM) flpInstr);
  SetFocus(fEditCtrl->GetWindow());

  fSetTextInput = TRUE;
 // delete [] flpInstr; flpInstr = 0;
  return 0;
}


//______________________________________________________________________________
 LRESULT APIENTRY
         TGWin32WindowsObject::OnRootTextInput
                               (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
 {
     // This method reads the input string via WIN32 EDIT Control
     // if (!fEditCtrl) return OnRootEditInput(hwnd, uMsg, wParam, lParam);

     switch (uMsg) {
       case WM_CHAR:
         switch(wParam) {
           case 0x0A: /* line feed       */
           case 0x0D: /* carriage return */
                     flpROOTString->IncBreakKey();
           case 0x1B: /* escape          */
               {
                     flpROOTString->IncBreakKey();

                     fSetTextInput = FALSE;
                                         fEditCtrl->GetText(flpInstr);
                     ::ShowWindow(fEditCtrl->GetWindow(),SW_HIDE);
                     // delete fEditCtrl; fEditCtrl = 0;
                     flpROOTString->Release();
               }
               return 0;
           default:
               return  DefWindowProc(hwnd,uMsg, wParam, lParam);
               break;
         };
         case IX11_ROOT_Input:
             return OnRootEditInput(hwnd, uMsg, wParam, lParam);
         default:
             break;
     }
     return  DefWindowProc(hwnd,uMsg, wParam, lParam);
 }


//______________________________________________________________________________
TGWin32WindowsObject::TGWin32WindowsObject() : TGWin32Object() {;}

//______________________________________________________________________________
TGWin32WindowsObject::TGWin32WindowsObject(TGWin32 *lpTGWin32, Int_t x, Int_t y, UInt_t w, UInt_t h){
 CreateWindowsObject(lpTGWin32, x, y, w, h);
}

//______________________________________________________________________________
TGWin32WindowsObject::TGWin32WindowsObject(TGWin32 *lpTGWin32, UInt_t w, UInt_t h){
 CreateWindowsObject(lpTGWin32, 0, 0, w, h);
}

//______________________________________________________________________________
void TGWin32WindowsObject::CreateWindowsObject(TGWin32 *lpTGWin32, Int_t x, Int_t y, UInt_t w, UInt_t h){

  fPaintFlag   = 0; // Object ought accept SendMessage command
                    // =1 means object must be paint directly (is set from within WinProc)
  fWin32Mother = lpTGWin32;
  fIsPixmap    = 0;
  fTypeFlag    = 0;
  fButton      = 0;
  fMouseActive = kTRUE;
//  fCanvas      = 0;
  fMenu        = 0;  // No menu at this point
  fContextMenu = 0;
  fStaticMenuItems = 0;
  fStatusBar   = 0;  // No status bar at this point

  fDwStyle     = WS_OVERLAPPEDWINDOW | WS_VISIBLE; // window style
  fDwExtStyle  = WS_EX_CONTEXTHELP | WS_EX_OVERLAPPEDWINDOW;

  fDoubleBuffer = 0;  // There are no double buffer and its flag is disable
  fBufferObj    = 0;

  fXMouse = 0;
  fYMouse = 0;

//*-*
//*-* Set callback function for all desired events
//*-*

  Win32CreateCallbacks();

  if (!fIsPixmap) {


    fWinSize.cx = w;
    fWinSize.cy = h;

 //   int lx = x ? x : CW_USEDEFAULT;  // Set x-default start position for the window
 //    int ly = y ? y : CW_USEDEFAULT;  // Set x-default start position for the window

    int lx = x ;  // Set x-default start position for the window
    int ly = y ;  // Set x-default start position for the window

    RECT adjust = { lx, ly, w, h };

    AdjustWindowRectEx(&adjust,fDwStyle,TRUE,fDwExtStyle);

    fPosition.x = x ? adjust.left : CW_USEDEFAULT ;
    fPosition.y = y ? adjust.top  : CW_USEDEFAULT ;

    fSizeFull.cx = adjust.right;
    fSizeFull.cy = adjust.bottom;


//*-*-     Create ROOT window

    fhSemaphore = CreateSemaphore(NULL, 0, 1, NULL);

    while(!PostThreadMessage(fWin32Mother->fIDThread,
           IX11_ROOT_MSG,MAKEWPARAM(IX_OPNWI,ROOT_Control),(DWORD)this)){
            printf(" PostThreadMessage for  %d error %d \n",fWin32Mother->fIDThread, GetLastError());}
    if (GetCurrentThreadId() == fWin32Mother->fIDThread)
        Error("CreateWindowsObject"," Dead lock !!!");
    else
         WaitForSingleObject(fhSemaphore, INFINITE);
    CloseHandle(fhSemaphore);

//    ShowWindow(fhwndRootWindow,SW_SHOWDEFAULT);
//    ShowWindow(fhwndRootWindow,SW_SHOWNORMAL);
//    UpdateWindow(fhwndRootWindow);
  }
}


//______________________________________________________________________________
void TGWin32WindowsObject::Win32CreateObject()
//*-*-*-*-*-*-*-*-*-*-*-*-*Win32CreateObject*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =================
//*-*  Open new GDI object (WIN32 ROOT window) on the screen
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
{
  int x = fPosition.x;
  int y = fPosition.y;

  fhwndRootWindow = CreateWindowEx(fDwExtStyle,
                            fWin32Mother->fROOTCLASS,           // extended window style
                            (char *)(fWin32Mother->GetTitle()), // address of window name
                            fDwStyle, // window style
                            x,y,                              // start positio of the window,
                            fSizeFull.cx, fSizeFull.cy,       // size of the window
                            NULL,                             // handle of parent of owner window
                            NULL,                             // handle of menu, or child-window identifier
                            fWin32Mother->fHInstance,         // handle of application instance
                            this);                            // address of window-creation data

  ReleaseSemaphore(fhSemaphore, 1, NULL);
// --> temporary        lpWinThr->fhwndRootWindow = CreateTextClass(WinThr);
}

//______________________________________________________________________________
TGWin32WindowsObject::~TGWin32WindowsObject(){
    if (fTypeFlag == -1) return;
    W32_Close();
}

//______________________________________________________________________________
void TGWin32WindowsObject::CreateDoubleBuffer()
{
 if (fBufferObj) return;

//*-*  Create double buffer for the first time
    int x,y,w,h;
    RECT winsize;
    GetClientRect(fhwndRootWindow,&winsize);
    x = winsize.left;
    y = winsize.top;
    w = winsize.right;
    h = winsize.bottom;

    fBufferObj = new TGWin32PixmapObject(fWin32Mother,w,h);
    ((TGWin32PixmapObject *)fBufferObj)->W32_Clear();
}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnCommand
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*     WM_COMMAND
//*-*     ==========
//*-*     wNotifyCode = HIWORD(wParam) -  notification code
//*-*                                     Specifies the notification code if the
//*-*                                     message is from a control.
//*-*                                = 1  If the message is from an accelerator.
//*-*                                  0  If the message is from a menu
//*-*     wID = LOWORD(wParam);         // item, control, or accelerator identifier
//*-*     hwndCtl = (HWND) lParam;        handle of control
//*-*                               != 0  Identifies the control sending the
//*-*                                     message if the message is from a control.
//*-*                                  0   Otherwise
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  UINT wID = LOWORD(wParam);
  if (!HIWORD(wParam))
  {
//      if (fCanvasImp && wID > 0 ) RunMenuItem(wID);
      if (wID > 0 ) RunMenuItem(wID);
//*-* If an application processes this message, it should return zero.
      return 0;
  }
  else
          return DefWindowProc(hwnd, uMsg, wParam, lParam);

//      return OnCommandForControl(hwnd,uMsg,wParam,lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnCommandForControl
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    Int_t wID = LOWORD(wParam);
    TWin32CommCtrl *ctrl = (TWin32CommCtrl *)fCommandArray[wID];
    if (!wID)
        Error("OnCommandForControl","Control ID=0");
    if (ctrl == 0)
        Error("OnCommandForControl","Wrong control ID");
//    else
//        return ctrl->OnCommand(hwnd,uMsg,wParam,lParam);

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
//______________________________________________________________________________
TCanvas  *TGWin32WindowsObject::GetCanvas(){
//    if (fCanvas && !InSendMessage())
//      fWin32Mother->read_lock();
  if (fCanvasImp)
      return gROOT->IsLineProcessing() ? 0 : fCanvasImp->Canvas();
  return 0;
};


//______________________________________________________________________________
BOOL TGWin32WindowsObject::IsMouseLeaveCanvas(Int_t x, Int_t y){

POINT hotpos = {x,y};
RECT rect;

ScreenToClient(fhwndRootWindow, &hotpos);
GetClientRect(fhwndRootWindow, &rect);

if (hotpos.x < 0 || hotpos.y < 0 ||
     hotpos.x >= rect.right || hotpos.y >= rect.bottom) return FALSE;
return TRUE;
}
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnChar
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*
//*-*  Windows Procedure to manage: WM_CHAR
//*-*
 if (fSetTextInput) return OnRootTextInput(hwnd, uMsg, wParam, lParam);
 TCanvas *canvas = 0;
 if (canvas=GetCanvas()) {
    canvas->HandleInput(kKeyPress,Int_t(wParam),Int_t(lParam & 0xff));
    return 0;
 }
 return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnKeyDown
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*
//*-*  Windows Procedure to manage: WM_KEYDOWN
//*-*
 if      (fMouseInit)    return OnRootMouse(hwnd, uMsg, wParam, lParam);
 else if (fSetTextInput) return OnRootTextInput(hwnd, uMsg, wParam, lParam);
 return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnKillFocus
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*
//*-*  Windows Procedure to manage: WM_KILLFOCUS
//*-*

//*-*
//*-*  Hide and destroy the caret when the window loses
//*-*  keyboard focus
//*-*
#if 0
 if (fSetTextInput) {
   HideCaret(hwnd);
   DestroyCaret();
   return 0;
 }
#endif
 return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnMouseButton
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//
//  Windows Procedure to manage: WM_LBUTTONDOWN WM_MBUTTONDOWN WM_RBUTTONDOWN
//                               WM_LBUTTONUP   WM_MBUTTONUP   WM_RBUTTONUP
//                               WM_MOUSEMOVE
//                               WM_CONTEXTMENU
//
//                               WM_LBUTTONDBLCLK  WM_MBUTTONDBLCLK   WM_RBUTTONDBLCLK
//

 TCanvas *canvas = 0;

 if (canvas=GetCanvas()) {

  if (fMouseInit)
     switch (uMsg) {
       case WM_LBUTTONDOWN:
       case WM_MBUTTONDOWN:
       case WM_RBUTTONDOWN:
         fMouseInit = 0;
         OnRootMouse(hwnd,uMsg,wParam,lParam);
         return 0;
       case WM_MOUSEMOVE:
//        SetCursor(fWin32Mother->fCursors[kPointer]);
       case WM_RBUTTONUP:
       case WM_MBUTTONUP:
       case WM_LBUTTONUP:
         OnRootMouse(hwnd,uMsg,wParam,lParam);
         return 0;
       default:
         break;
     };


  Int_t x = (LONG) (MAKEPOINTS(lParam).x);
  Int_t y = (LONG) (MAKEPOINTS(lParam).y);

  EEventType buttonevent = kNoEvent;
  HWND hCapWin = GetCapture();

#ifndef WIN32
//*-*   Test "mouse leave windows event"
  if (fButton && hCapWin == hwnd && IsMouseLeaveCanvas(x,y)) {


//*-*   Enforce ButtonUp events artificially

      if (fButton & 1)
          canvas->HandleInput(kButton1Up, fXMouse, fYMouse);
      if (fButton & 2)
          canvas->HandleInput(kButton2Up, fXMouse, fYMouse);
      if (fButton & 4)
          canvas->HandleInput(kButton3Up, fXMouse, fYMouse);
      fButton = 0;
      ReleaseCapture();
  }
#endif

  switch (uMsg) {
    case WM_LBUTTONDOWN:
      buttonevent = kButton1Down;
      fButton |= 1;
      if (!hCapWin) SetCapture(hwnd);
      break;
    case WM_MBUTTONDOWN:
      buttonevent = kButton2Down;
      fButton |= 2;
      if (!hCapWin) SetCapture(hwnd);
      break;
    case WM_RBUTTONDOWN:
    case WM_CONTEXTMENU:  // User clicks the right mouse button - it means (by Microsoft)
                          //                                        call ContextMenu
      buttonevent = kButton3Down;
      fButton |= 4;
      if (!hCapWin) SetCapture(hwnd);
      if (fButton & 1) {
        SendMessage(hwnd,WM_LBUTTONUP,wParam,lParam);
        SendMessage(hwnd,WM_MBUTTONDOWN,wParam,lParam);
        buttonevent = kNoEvent;
      }
      break;
    case WM_MOUSEMOVE:
      if (fButton & 1) {
        if (wParam && MK_LBUTTON)
            canvas->HandleInput(kButton1Motion, x, y);
        else
        {
            canvas->HandleInput(kButton1Up, fXMouse, fYMouse);
            fButton = 0;
        }

      }
      if (fButton == 0)
                canvas->HandleInput(kMouseMotion, x, y);
//*-*  Only Left_button mottion has beed defined yet
      if (fButton & 6 && !(fButton & 1))
          return DefWindowProc(hwnd, uMsg, wParam, lParam);
      break;
    case WM_LBUTTONUP:
      if (fButton & 1) {
        buttonevent = kButton1Up;
        fButton = 0;
      }
      else if (fButton & 2) {
        buttonevent = kButton2Up;
        fButton = 0;
      }
      break;
    case WM_MBUTTONUP:
      if (fButton & 2) {
        buttonevent = kButton2Up;
        fButton = 0;
      }
      break;
    case WM_RBUTTONUP:
      if (fButton & 4) {
        buttonevent = kButton3Up;
        fButton = 0;
      }
      else if (fButton & 2) {
        buttonevent = kButton2Up;
        fButton = 0;
      }
      break;
    case WM_LBUTTONDBLCLK:
       buttonevent = kButton1Double;
       break;
    case WM_MBUTTONDBLCLK:
       buttonevent = kButton2Double;
       break;
    case WM_RBUTTONDBLCLK:
       buttonevent = kButton3Double;
       break;
    default: break;
  }

  if (buttonevent != kNoEvent) canvas->HandleInput(buttonevent, x, y);

//*-* HandleInput can delete the 'fWin32Mother' object occasionally so we must check this.

  if (fWin32Mother) SetCursor(fWin32Mother->fCursors[fWin32Mother->fCursor]);

  if (hCapWin == hwnd && !fButton) ReleaseCapture();

  fXMouse = x;
  fYMouse = y;
  LeaveCrSection();
  return 0;
}

else
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnMouseActivate
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*   WM_MOUSEACTIVATE

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnPaint
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*  WM_PAINT  message

   PAINTSTRUCT ps;

// typedef struct tagPAINTSTRUCT { // ps
//    HDC  hdc;
//    BOOL fErase;
//    RECT rcPaint;
//    BOOL fRestore;
//    BOOL fIncUpdate;
//    BYTE rgbReserved[32];
// } PAINTSTRUCT;

   if (GetUpdateRect(hwnd,NULL, FALSE)==TRUE) {
       if (BeginPaint(hwnd,&ps)) {
           if (fBufferObj) {
               TGWin32UpdateWindow *OpCode = new TGWin32UpdateWindow();
               ExecCommand(OpCode);
           }
           EndPaint(hwnd, &ps);
       }
   }
   else if (fWin32Mother->fhdCommonPalette)
   {
       HPALETTE hPal = SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,
                                     FALSE);
       if (hPal != fWin32Mother->fhdCommonPalette) DeleteObject(hPal);
       RealizePalette(fObjectDC);
   }

return 0;

}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnPaletteChanged
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
 //*-*    Message ID: WM_PALETTECHANGED
  if (hwnd != (HWND) wParam && fWin32Mother->fhdCommonPalette)
      InvalidateRect(hwnd,NULL,FALSE);
  return DefWindowProc(hwnd, uMsg, wParam, lParam);

}
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnSetFocus
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
 //*-*    Message ID: WM_SETFOCUS

#if 0
 if (fSetTextInput) {
   CreateCaret(hwnd, (HBITMAP) 1, fdwCharX, fdwAscent);
   SetCaretPos(fiXText, fiYText);
   ShowCaret(hwnd);
   return 0;
 }
#endif
 return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnGetMinMaxInfo
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*    Message ID: WM_GETMINMAXINFO
//*-*                ================
//*-* The WM_GETMINMAXINFO message is sent to a window when the size or position
//*-* of the window is about to change. An application can use this message to
//*-* override the window's default maximized size and position, or its default
//*-* minimum or maximum tracking size.

//*-* lpmmi = (LPMINMAXINFO) lParam; // address of structure

  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnSize
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*  WM_SIZE  message
//*-*
//*-*  fwSizeType = wParam;     -  resizing flag
//*-*      nWidth = LOWORD(lParam); -  width of client area
//*-*     nHeight = HIWORD(lParam); -  height of client area
//*-*

    if (wParam == SIZE_MAXHIDE  ||
        wParam == SIZE_MAXSHOW  ||
        wParam == SIZE_MINIMIZED) return DefWindowProc(hwnd, uMsg, wParam, lParam);

    fWin32WindowSize.right  = LOWORD(lParam);
    fWin32WindowSize.bottom = HIWORD(lParam);
//*-*  Resize the status bar window if any
    if (fStatusBar && fStatusBar->IsVisible() )
    {
        SendMessage(fStatusBar->GetWindow(),uMsg,wParam,lParam);
        fStatusBar->OnSize();
        fWin32WindowSize.bottom -= fStatusBar->GetHeight();
    }

    SetWindowExtEx  (fObjectDC, fWin32WindowSize.right,
        fWin32WindowSize.bottom, NULL);
    SetViewportExtEx(fObjectDC, fWin32WindowSize.right,
        fWin32WindowSize.bottom, NULL);

    DPtoLP(fObjectDC,(POINT*) (&fWin32WindowSize),2);

//*-*   Resize the double buffer if any
    if (fBufferObj)
    {
        TGWin32Object *winobj = fBufferObj->Rescale(fWin32WindowSize.right,fWin32WindowSize.bottom);
        if (winobj && winobj != fBufferObj)
        {
          //*-*  Rescale bit map
            HBITMAP hbsrc = ((TGWin32PixmapObject *)fBufferObj)->GetBitmap();

//*-* Define the size of the source object;

            BITMAP Bitmap_buffer;
            GetObject(hbsrc, sizeof(BITMAP),&Bitmap_buffer);

            StretchBlt(
                winobj->GetWin32DC(),   // handle of destination device context
                0,                      // x-coordinate of upper-left corner of dest. rect.
                0,                      // y-coordinate of upper-left corner of dest. rect.
                fWin32WindowSize.right, // width of destination rectangle
                fWin32WindowSize.bottom,// height of destination rectangle
                fBufferObj->GetWin32DC(),   // handle of source device context
                0,                      // x-coordinate of upper-left corner of source rectangle
                0,                      // y-coordinate of upper-left corner of source rectangle
                Bitmap_buffer.bmWidth,  // width of source rectangle
                Bitmap_buffer.bmHeight, // height of source rectangle
                SRCCOPY                 // raster operation code
                );

            TGWin32Object *o = fBufferObj;
            fBufferObj = winobj;
            delete o;
        }
        if (wParam == SIZE_MAXIMIZED)
            SendMessage(hwnd,WM_EXITSIZEMOVE,0,0);
        return 0;
    } else
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnSizing
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*  WM_SIZING  message
//*-*
//*-*  fwSide = wParam;         // edge of window being sized
//*-*  lprc = (LPRECT) lParam;  // screen coordinates of drag rectangle
//*-*
//*-* The WM_SIZING message is sent to a window that the user is resizing.
//*-* By processing this message, an application can monitor the size and
//*-* position of the drag rectangle and, if needed, change its size or
//*-* position.

     return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnSysCommand
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*  WM_SYSCOMMAND  WM_DESTROY
//*-*  =============  ==========
//*-*    uCmdType = wParam;        - type of system command requested
//*-*    xPos = LOWORD(lParam);    - horizontal position, in the screen coordinates
//*-*    yPos = HIWORD(lParam);    - vertical postion, in the screen coordinates

//*-*   By unknown reason this message is supplied with the "zero" parameters
//*-*   but Windows sends a WM_CLOSE message itself to proceed

// if ((wParam & 0xFFF0) == SC_CLOSE) return OnClose(hwnd,uMsg,wParam,lParam);
    if ((wParam & 0xFFF0) == SC_CLOSE)
    {
        TCanvas *canvas = GetCanvas();
        if(canvas) { // delete canvas;
            char cmd[]= "TCanvas *c=(TCanvas *)0x1234567890123456; delete c;";
            sprintf(cmd,"TCanvas *c=(TCanvas *)%#16x; delete c;",canvas);
            printf(" OnSysCommand %s \n", cmd);
//                                      gInterpreter->ProcessLineAsynch(cmd);
            gROOT->ProcessLine(cmd);
        }
        return DefWindowProc(hwnd,uMsg, wParam, lParam);
    }

#if 0
 if (fWin32Mother->fhdCommonPalette) {
   SetSystemPaletteUse(fObjectDC,SYSPAL_STATIC);
   while(!PostMessage(HWND_BROADCAST,WM_SYSCOLORCHANGE, 0, 0)){;}
 }
#endif

    return DefWindowProc(hwnd,uMsg, wParam, lParam);
}
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnClose
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*  WM_CLOSE
//*-*  ========

TCanvas *canvas = 0;
if (canvas =  GetCanvas()) {
    Int_t cId = canvas->GetCanvasID();
    if (cId != -1) canvas->Close();
//    fCanvas = 0;
    fCanvasImp = 0;
    LeaveCrSection();
    return 0;
}
return DefWindowProc(hwnd,uMsg, wParam, lParam);

}
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnActivate
                (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
   switch (LOWORD(wParam))
    {
        case WA_CLICKACTIVE:
            fMouseActive = kTRUE;
            UpdateWindow(hwnd);
            break;
        case WA_INACTIVE:
            fMouseActive = kFALSE;
            break;
        default:
            UpdateWindow(hwnd);
            break;
    }
  return DefWindowProc(hwnd,uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnEraseBkgnd
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  return DefWindowProc(hwnd,uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnExitSizeMove
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//    Message ID: WM_EXITSIZEMOVE
//                ===============
//*-* The WM_EXITSIZEMOVE message is sent once to a window after
//*-* it has exited the moving or sizing mode.

//*-*    Resize the CurrentCanvas

  TCanvas *canvas = 0;
  if (canvas = GetCanvas()) {
      canvas->Resize();
      canvas->Update();
      LeaveCrSection();
  }
  return 0;
}


//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnCreate
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//
//  Windows Procedure to manage: WM_CREATE
//
  int iLoop;
  TEXTMETRIC *tm;
  RECT RectClient;
  HPALETTE hpallete;
  SIZE WindowSize;
  HMENU hmenu;

//*-*
//*-* Disable the Close postion in the System Menu
//*-*
  hmenu = GetSystemMenu(hwnd,FALSE);
//*-*   EnableMenuItem(hmenu,6 , MF_GRAYED | MF_BYPOSITION);


//
// Initial fill the data structure in
//
  fObjectDC           = GetDC(hwnd);
  fObjectClipRegion      = (HRGN)NULL;
  fROOTCursor          = FALSE;
  fSystemCursorVisible = TRUE;
  fMouseInit           = 0;
  fSetTextInput        = FALSE;

  fLoc.x = fLoc.y = fLocp.x = fLocp.y=0;

//*-* Create caret to character input

//   tm = new TEXTMETRIC;
  tm = (TEXTMETRIC *)malloc(sizeof(TEXTMETRIC));
    GetTextMetrics(fObjectDC,tm);
    fdwCharX = tm->tmAveCharWidth;
    fdwCharY = tm->tmHeight;
    fdwAscent= tm->tmAscent;
//    delete tm;
  ::free(tm);
//*-*
//*-*  Set and adjust a client area of the window object
//*-*
  SetMapMode    (fObjectDC,MM_ISOTROPIC);
  GetClientRect(hwnd,&RectClient);

// ?  RectClient.left   = 0;
// ? RectClient.top    = 0;
// ? RectClient.right  = 1023;
// ? RectClient.bottom = 1023;

  SetWindowExtEx(fObjectDC,
                 RectClient.right,
                 RectClient.bottom,
                 NULL);

  SetBkMode(fObjectDC,TRANSPARENT);
  SetTextAlign(fObjectDC,TA_BASELINE | TA_LEFT | TA_NOUPDATECP);

// ? GetClientRect(hwnd,&RectClient);

  SetViewportExtEx (fObjectDC,
                    RectClient.right,
                    RectClient.bottom,
                    NULL);

  GetClientRect(hwnd,&fWin32WindowSize);

  DPtoLP(fObjectDC,(POINT*)(&fWin32WindowSize),2);


  if (!fWin32Mother->flpPalette)
  {
//   if (fWin32Mother->fhdCommonPalette == NULL) {

    int iPalExist = GetDeviceCaps(fObjectDC,RASTERCAPS) & RC_PALETTE ;

    if (iPalExist)
       fWin32Mother->fMaxCol = GetDeviceCaps(fObjectDC,SIZEPALETTE);
    else {
       fWin32Mother->fMaxCol = GetDeviceCaps(fObjectDC,NUMCOLORS);
 //  9/9/96     if (fWin32Mother->fMaxCol == -1)
       fWin32Mother->fMaxCol = 256-20;
       fWin32Mother->fhdCommonPalette = 0;
    }

//*-*  At present ROOT will use 236 colors only ???

    fWin32Mother->fMaxCol = TMath::Min(256-20,fWin32Mother->fMaxCol);

//*-*  Create palette

     fWin32Mother->flpPalette = (LPLOGPALETTE) malloc((sizeof (LOGPALETTE) +
                 (sizeof (PALETTEENTRY) * ( fWin32Mother->fMaxCol))));

    if(!fWin32Mother->flpPalette){
        MessageBox(NULL, "<WM_CREATE> Not enough memory for palette.", NULL, MB_OK | MB_ICONHAND);
        PostQuitMessage (0) ;
    }

    fWin32Mother->flpPalette->palVersion    = 0x300;
    fWin32Mother->flpPalette->palNumEntries = fWin32Mother->fMaxCol;

//*-*  fill in intensities for all palette entry colors

   if (iPalExist) {
//*-*
//*-*  create a logical color palette according the information
//*-*  in the LOGPALETTE structure.
//*-*

#if 0
       GetSystemPaletteEntries(fObjectDC,  0,10,  fWin32Mother->flpPalette->palPalEntry);
       GetSystemPaletteEntries(fObjectDC,245,10,  fWin32Mother->flpPalette->palPalEntry+245);
#endif
        fWin32Mother->fhdCommonPalette = CreatePalette ((LPLOGPALETTE)  fWin32Mother->flpPalette);
   }
 }
  if (fWin32Mother->fhdCommonPalette)
  {
      HPALETTE hPal = SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,FALSE);
//      HPALETTE hPal = SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,TRUE);
      if (hPal != fWin32Mother->fhdCommonPalette) DeleteObject(hPal);
//          DeleteObject(SelectPalette(fObjectDC,fWin32Mother->fhdCommonPalette,TRUE));
      RealizePalette(fObjectDC);
  }

  return 0;
}
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::OnRootHook
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    MSG msg = {hwnd,uMsg,wParam,lParam};
    TWin32HookViaThread::ExecuteEvent(&msg,kFALSE);
    return 0;
}


#ifndef WIN32
//______________________________________________________________________________
LRESULT APIENTRY TGWin32WindowsObject::Win_ButtonDown
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
 return 0;
}
#endif

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_Clear(){ }
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_Close(){
//*-* Delete attributes if any
   if (fMenu)        { delete fMenu;        fMenu   = 0;}
   if (fContextMenu) { delete fContextMenu; fContextMenu = 0;}
   if (fStaticMenuItems) {
      delete [] fStaticMenuItems;
      fStaticMenuItems = 0;
   }
//*-* Delete then Status bar if present
   SafeDelete(fStatusBar);
   fStatusBar = 0;

//*-*  Now we can close the window
   TGWin32Command *OpCode = new TGWin32Command(IX_CLSWI,ROOT_Control);
   OpCode->SetBuffered(0);
   ExecCommand(OpCode);

//       Delete menu if presented
//       if (fMenu) delete fMenu;

   Delete();
}

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_CopyTo(TGWin32Object *obj, int xpos, int ypos){
//*-*
//*-* If this request comes from application we provide no real copy operation
//*-*
    TGWin32CopyTo *CodeOp = new TGWin32CopyTo(obj,xpos,ypos,0,0);
    CodeOp->SetBuffered(-1);  // Always with double buffer !!!
    ExecCommand(CodeOp);
}
//______________________________________________________________________________
void TGWin32WindowsObject::W32_CreateStatusBar(Int_t nparts)
{
 if (fStatusBar)
   fStatusBar->SetStatusParts(this, nparts);
 else
   fStatusBar = new TGWin32StatusBar(this, nparts);
}
//______________________________________________________________________________
void TGWin32WindowsObject::W32_CreateStatusBar(Int_t *parts, Int_t nparts)
{
 if (fStatusBar)
   fStatusBar->SetStatusParts(this, parts, nparts);
 else
   fStatusBar = new TGWin32StatusBar(this, parts, nparts);
}

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_DrawBox(int x1, int y1, int x2, int y2, TVirtualX::EBoxMode mode){ }
void  TGWin32WindowsObject::W32_DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny, int *ic){ }
void  TGWin32WindowsObject::W32_DrawFillArea(int n, TPoint *xy){ }
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_DrawLine(int x1, int y1, int x2, int y2){
   TPoint xy[2];
   xy[0].SetX(x1);
   xy[0].SetY(y1);
   xy[1].SetX(x2);
   xy[1].SetY(y2);
   W32_DrawPolyLine(2, xy);
}

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_DrawPolyLine(int n, TPoint *xy){ ; }

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_DrawPolyMarker(int n, TPoint *xy){ }
void  TGWin32WindowsObject::W32_DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode mode){ }
void  TGWin32WindowsObject::W32_GetCharacterUp(Float_t &chupx, Float_t &chupy){ }
Int_t TGWin32WindowsObject::W32_GetDoubleBuffer()
{
  TGWin32GetDoubleBuffer CodeOp;
  ExecCommand(&CodeOp);
  CodeOp.Wait();
  return CodeOp.GetBuffer();
}
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_GetGeometry(int &x, int &y, unsigned int &w, unsigned int &h)
{
  RECT winsize;
  WPARAM wParam = MAKEWPARAM(IX_GETGE,ROOT_Inquiry);
  LPARAM lParam = (LPARAM) (&winsize);

  GetClientRect(fhwndRootWindow,&winsize);

  if (fStatusBar && fStatusBar->IsVisible() )
      winsize.bottom -= fStatusBar->GetHeight();

  x = winsize.left;
  y = winsize.top;
  w = winsize.right;
  h = winsize.bottom;
}
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_GetPixel(int y, int width, Byte_t *scline){ }
void  TGWin32WindowsObject::W32_GetRGB(int index, float &r, float &g, float &b){ }
void  TGWin32WindowsObject::W32_GetTextExtent(unsigned int &w, unsigned int &h, char *mess){ }
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_Move(Int_t x, Int_t y){
    SetWindowPos(fhwndRootWindow, // handle of window
                 HWND_TOP,        // Places the window at the top of the Z order.
                 x,               // Specifies the new position of the left side of the window.
                 y,               // Specifies the new position of the top of the window
                 0,0,
                 SWP_NOSIZE);     // Retains the current size (ignores the cx and cy parameters).
    }
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_PutByte(Byte_t b){ }
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_QueryPointer(int &ix, int &iy){ }
//______________________________________________________________________________
Int_t TGWin32WindowsObject::W32_RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y){
//*-*-*-*-*-*-*-*-*-*-*Request Locatorr position*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*  x,y       : cursor position at moment of button press (output)
//*-*  ctyp      : cursor type (input)
//*-*
//*-*    ctyp=1 tracking cross
//*-*    ctyp=2 cross-hair
//*-*    ctyp=3 rubber circle
//*-*    ctyp=4 rubber band
//*-*    ctyp=5 rubber rectangle
//*-*
//*-*  mode      : input mode
//*-*
//*-*    mode=0 request
//*-*    mode=1 sample
//*-*
//*-*  Request locator:
//*-*  return button number  1 = left is pressed
//*-*                        2 = middle is pressed
//*-*                        3 = right is pressed
//*-*        in sample mode:
//*-*                       11 = left is released
//*-*                       12 = middle is released
//*-*                       13 = right is released
//*-*                       -1 = nothing is pressed or released
//*-*                       -2 = leave the window
//*-*                     else = keycode (keyboard is pressed)
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  TGWin32GetLocator  CodeOp(x,y,ctyp,mode), *code = &CodeOp;

  CodeOp.SetBuffered(0);

  SendMessage(fhwndRootWindow,
              IX11_ROOT_Input,
              (WPARAM)code->GetCOP(),
              (LPARAM)code);

  CodeOp.Wait();

  x = CodeOp.GetX();
  y = CodeOp.GetY();
  return CodeOp.GetButton();
}

//______________________________________________________________________________
Int_t TGWin32WindowsObject::W32_RequestString(int x, int y, char *text){
//*-*-*-*-*-*-*-*-*-*-*-*Request string*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==============
//*-*  x,y         : position where text is displayed
//*-*  text        : text displayed (input), edited text (output)
//*-*
//*-*  Request string:
//*-*  text is displayed and can be edited with Emacs-like keybinding
//*-*  return termination code (0 for ESC, 1 for RETURN)
//*-*
//*-*  Return value:
//*-*
//*-*    0     -  input was canceled
//*-*    1     -  inout was Ok
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  TGWin32GetString  ROOTString(x,y,text), *code = &ROOTString;

  ROOTString.SetBuffered(0);

  SendMessage(fhwndRootWindow,
              IX11_ROOT_Input,
              (WPARAM)code->GetCOP(),
              (LPARAM)code);

  ROOTString.Wait();

  return ROOTString.GetBreakKey();
}
//______________________________________________________________________________
TGWin32Object *TGWin32WindowsObject::Rescale(unsigned int w, unsigned int h){
    W32_Set(0,0,w,h);
    return this;
}
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_Resize(){ }

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_Select()
{
//*-*  Now we can close the window
#if 0
    if (fOpenGLRC)
    {
        Warning("TGWin32WindowsObject::W32_Select()"," entered");
        TGWin32GLCommand CodeOp;
        CodeOp.SetBuffered(0);
        ExecCommand(&CodeOp);
        CodeOp.Wait();
    }
    if (fOpenGLRC)
        fOpenGLRC->MakeCurrent();
#endif

}
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_Set(int x, int y, int w, int h){
    UINT  fuFlags;
    fuFlags  = x+y ? SWP_SHOWWINDOW : SWP_NOMOVE; // Retains the current position (ignores the x and y parameters).
    fuFlags |= w+h ? 0 : SWP_NOSIZE;              // Retains the current size (ignores the cx and cy parameters).
    SetWindowPos(fhwndRootWindow, // handle of window
                 HWND_TOP,        // Places the window at the top of the Z order.
                 x,               // Specifies the new position of the left side of the window.
                 y,               // Specifies the new position of the top of the window
                 w,               // Specifies the new width of the window, in pixels.
                 h,               // Specifies the new height of the window, in pixels
                 fuFlags);
}
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_SetCharacterUp(Float_t chupx, Float_t chupy){ }
void  TGWin32WindowsObject::W32_SetClipOFF(){ }
void  TGWin32WindowsObject::W32_SetClipRegion(int x, int y, int w, int h){ }
void  TGWin32WindowsObject::W32_SetCursor(ECursor cursor){ }
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_SetDoubleBuffer(int mode)
{
//*-*-*-*-*-*-*-*-*-*Set the double buffer on/off for this window-*-*-*-*-*
//*-*                ============================================
//*-* mode : 1 double buffer is on
//*-*        0 double buffer is off
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   TGWin32SetDoubleBuffer *CodeOp = new TGWin32SetDoubleBuffer(mode);
   ExecCommand(CodeOp);
}
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_SetDoubleBufferOFF(){ }
void  TGWin32WindowsObject::W32_SetDoubleBufferON(){ }
void  TGWin32WindowsObject::W32_SetDrawMode(TVirtualX::EDrawMode mode){ }
void  TGWin32WindowsObject::W32_SetFillColor(Color_t cindex){ }
void  TGWin32WindowsObject::W32_SetFillStyle(Style_t style){ }
void  TGWin32WindowsObject::W32_SetLineColor(Color_t cindex){ }
void  TGWin32WindowsObject::W32_SetLineType(int n, int *dash){ }
void  TGWin32WindowsObject::W32_SetLineStyle(Style_t linestyle){ }
void  TGWin32WindowsObject::W32_SetLineWidth(Width_t width){ }
void  TGWin32WindowsObject::W32_SetMarkerColor( Color_t cindex){ }
void  TGWin32WindowsObject::W32_SetMarkerSize(Float_t markersize){ }
void  TGWin32WindowsObject::W32_SetMarkerStyle(Style_t markerstyle){ }
void  TGWin32WindowsObject::W32_SetRGB(int cindex, float r, float g, float b){ }
void  TGWin32WindowsObject::W32_SetTextAlign(Short_t talign){ }
void  TGWin32WindowsObject::W32_SetTextColor(Color_t cindex){ }
Int_t TGWin32WindowsObject::W32_SetTextFont(char *fontname, TVirtualX::ETextSetMode mode){return 0;}
void  TGWin32WindowsObject::W32_SetTextFont(Int_t fontnumber){ }
void  TGWin32WindowsObject::W32_SetTextSize(Float_t textsize){ }
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_SetMenu(){

// This function assigns a new menu to the specified window.
// if menu=0 it remove menu from the window but doesn't destroy it.

    HMENU m=GetWindowMenu()->GetMenuHandle();
    W32_SetMenu(m);
}

//______________________________________________________________________________
void TGWin32WindowsObject::W32_SetMenu(HMENU menu)
{
  TGWin32AddMenu *OpCode = new TGWin32AddMenu(menu);
  ExecCommand(OpCode);
}

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_SetTitle(const char *title){
//  char *tr_mess = (char *)malloc(strlen(title));
//  OemToChar(title,tr_mess);
  SetWindowText(fhwndRootWindow,    // handle of window
                         title    // address of string
 );
//  free(tr_mess);
}

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_ShowMenu(){
// function redraws the menu bar of the specified window
   if (GetMenu(fhwndRootWindow)) {
      if (!DrawMenuBar(fhwndRootWindow)) {
        Int_t err = GetLastError();
        Printf("Can't redraw the menu bar. Error code=%d \n", err);
      }
   }
   else
      Error(" TGWin32WindowsObject::W32_ShowMenu()","There is no Menu");
}

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_Show()
{
   if (fhwndRootWindow) ShowWindow(fhwndRootWindow,SW_SHOWNORMAL);
}
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_ShowMenu(Int_t x, Int_t y){
//*-*
//*-*  function draws the context menu at the specified location of the window
//*-*
   HMENU m=GetWindowContextMenu()->GetMenuHandle();
   if (m)
     TrackPopupMenuEx(
                       m,     // Handle to the pop-up menu to be displayed
           TPM_LEFTALIGN          // If this flag is set, the function positions
                              // the pop-up menu so that its left side is aligned
                              // with the coordinate specified by the x parameter.

         | TPM_RIGHTBUTTON        // If this flag is set, the pop-up menu tracks the
                              // right mouse button
     , x, y,  fhwndRootWindow,
     NULL                     // Pointer to a TPMPARAMS structure that specifies
                              // an area of the screen the menu should not overlap
   );
   else
      Error("TGWin32WindowsObject::W32_ShowMenu(Int_t x, Int_t y)","There is no menu to show");
}
//______________________________________________________________________________
void TGWin32WindowsObject::W32_ShowStatusBar(Bool_t show)
{
  if (fStatusBar) {
    RECT winrect;
    GetWindowRect(fhwndRootWindow,&winrect);
    Int_t w    = winrect.right -winrect.left; // width of the current window
    Int_t h    = winrect.bottom-winrect.top;  // height of the current window
    Int_t hBar = fStatusBar->GetHeight();     // height of the the status bar control

    if (show == kFALSE ) {
        fStatusBar->Hide();
        h -= hBar;
    }
    else {
        fStatusBar->Show();
        h += hBar;
    }

   // Change the full size of the current window
   // to avoid re-computing all ROOT objects
   // (it might take a lot of time)

    W32_Set(0,0,w, h);
  }
}

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_SetStatusText(const char *text, Int_t partidx, Int_t stype)
{
    if (fStatusBar)
       fStatusBar->SetText(text, partidx,stype);
}

//______________________________________________________________________________
void  TGWin32WindowsObject::W32_Update(int mode){

// mode = 0 - synch
//        1 - Copy double buffer to "real"screen"

 InvalidateRect(fhwndRootWindow,NULL,FALSE);
 UpdateWindow(fhwndRootWindow);
}
//______________________________________________________________________________
void  TGWin32WindowsObject::W32_Warp(int ix, int iy){ }
void  TGWin32WindowsObject::W32_WriteGIF(char *name){ }
void  TGWin32WindowsObject::W32_WritePixmap(unsigned int w, unsigned int h, char *pxname){ }
//______________________________________________________________________________
Int_t TGWin32WindowsObject::ExecCommand(TGWin32Command *code){
// return GetPaint()  ?
//                  OnRootAct(fhwndRootWindow, IX11_ROOT_MSG,
//                           (WPARAM)code->GetCOP(),
//                           (LPARAM)code)
//               :
   return           SendMessage(fhwndRootWindow,
                              IX11_ROOT_MSG,
                              (WPARAM)code->GetCOP(),
                              (LPARAM)code);

};

//______________________________________________________________________________
void TGWin32WindowsObject::RunMenuItem(Int_t index){
    TVirtualMenuItem *item = (TWin32MenuItem *)fCommandArray[index];
    if (!index)
        Error("RunMenuItem","Menu ID=0");
    if (item == 0)
        Error("RunMenuItem","Wrong menu ID");
    else
        item->ExecuteEvent(fCanvasImp);
}

//______________________________________________________________________________
void TGWin32WindowsObject::UnRegisterMenuItem(TVirtualMenuItem *item){
   Int_t id =  item->GetCommandId();
   if (id && id != -1) fCommandArray.Remove(item);
}

//______________________________________________________________________________
void TGWin32WindowsObject::UnRegisterControlItem(TWin32CommCtrl *ctrl){
   if (!ctrl) return;
   Int_t id =  ctrl->GetCommandId();
   if (id && id != -1) fCommandArray.Remove(ctrl);
}
