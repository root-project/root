// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   01/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWin32CommCtrl.h"
#include "TWin32BrowserImp.h"

///////////////////////////////////////////////////////////////
//                                                           //
// TWin32Control it is ABC class to use WIN32 common controls//
//           (ListView and TreeView)                         //
//                                                           //
//                                                           //
///////////////////////////////////////////////////////////////

// ClassImp(TWin32CommCtrl)

//______________________________________________________________________________
LRESULT APIENTRY TWin32CommCtrl::HookClass(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Windows procedure to subclass control                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
 TWin32CommCtrl *obj = 0;
 obj = (TWin32CommCtrl *)GetWindowLong(hwnd,GWL_USERDATA);

// following comments from Valery
//   The STRICT macro symbol is defined by VC++ v.6.0 It is only correction I
//   did to compile ROOT with VC++ v.6.0
 if (obj) {
     obj->OnSubClassCtrl(hwnd,uMsg,wParam,lParam);
#ifdef STRICT
     return ::CallWindowProc(obj->fPrevWndFunc, hwnd, uMsg, wParam, lParam);
#else
     return ::CallWindowProc((FARPROC)(obj->fPrevWndFunc), hwnd, uMsg, wParam, lParam);
#endif
 }
 return 0;

}

//______________________________________________________________________________
TWin32CommCtrl::TWin32CommCtrl(TGWin32WindowsObject *winobj, const char *title, Float_t x,Float_t y, Float_t width, Float_t height, const char *type, UInt_t style)
{
//  Create a control object for the winobj object.
        // The postion and the size of the control are defined in the relative units in the 0-1 range
        // For example width =1 means the the size of the control is equal of the width
        // of the client area of the master windows object.

//*-*  Ensure that the common control DLL is loaded
  InitCommonControls();

  fhwndWindow  = 0;
  fWindowType  = type;
  fWindowStyle = style;
  fMasterWindow= winobj;


  if (fMasterWindow) {
//*-* define the size of the control
          SetXCtrl(x);
          SetYCtrl(y);
          SetWCtrl(width);
          SetHCtrl(height);

      Int_t xpos, ypos, h, w;
      MapToMasterWindow(&xpos, &ypos, &h, &w);

      fWindowStyle |=  WS_CHILD;

      if (!(fhwndWindow = CreateWindowEx(WS_EX_CLIENTEDGE, (LPCTSTR) fWindowType, title, fWindowStyle,
                xpos, ypos, w, h,
                fMasterWindow->GetWindow(), (HMENU) (strstr(fWindowType,WC_TREEVIEW) ? kID_TREEVIEW:kID_LISTVIEW) ,
                (fMasterWindow->GetWin32ObjectMother())->GetWin32Instance(),
                0)))
          {
                  int err = GetLastError();
                  printf(" Last error was %d \n", err);
          Error("OnCreate","Can't create the WIN32 common control object");
          }
          else
                  SetSubClass();

  }
}

//______________________________________________________________________________
TWin32CommCtrl::~TWin32CommCtrl()
{
    if (fhwndWindow) {
       SetWindowLong(fhwndWindow,GWL_WNDPROC,(LONG) fPrevWndFunc);
       SetWindowLong(fhwndWindow,GWL_USERDATA,(LONG)0);
    }
}

//______________________________________________________________________________
UINT TWin32CommCtrl::SetCommandId(UINT id){
//*-*  Check whether the Command ID for this item had been assigned
//*-*  If this is for the first time just assign the new value
//*-*  return the ID for this Item

    if (fItemID == -1) fItemID = id;
    else
       Error("SetCommandId", "Impossible to set the Control ID twice");

    return GetCommandId();
}

//______________________________________________________________________________
void TWin32CommCtrl::SetSubClass()
{
// This function SubClass this control to cautch some Events
 fPrevWndFunc  = (WNDPROC) SetWindowLong(fhwndWindow,GWL_WNDPROC,(LONG) HookClass);
 SetWindowLong(fhwndWindow,GWL_USERDATA,(LONG)this);
}

//______________________________________________________________________________
void TWin32CommCtrl::SetXCtrl(Float_t x)
{
//*-*  Set the horisontal position of the control in the relative units
//*-*   (=1.0 means the right border of the client area of the the parent window)
    fXCtrl = 0;
    if (x >=0 && x < 1)
                 fXCtrl = x;
        else if (x >= 1)
                 fXCtrl = 1.0;
}

//______________________________________________________________________________
void TWin32CommCtrl::SetYCtrl(Float_t y)
{
//*-*  Set the vertical position of the control in the relative units
//*-*   (=1.0 means bottom of the client area of the the parent window)
    fYCtrl = 0;
    if (y >=0 && y < 1)
                 fYCtrl = y;
        else if (y >= 1)
                 fYCtrl = 1.0;
}

//______________________________________________________________________________
void TWin32CommCtrl::SetWCtrl(Float_t w)
{
//*-*  Set the width of the control in the relative units
//*-*   (=1.0 means as large as the parent window)
      fWidth = 1 ;
          if (w >=0 && w < 1)
             fWidth =   w;
          else if (w < 0)
             fWidth = 0;
}

//______________________________________________________________________________
void TWin32CommCtrl::SetHCtrl(Float_t h)
{
//*-*  Set the height of the control in the relative units
//*-*   (=1.0 means as large as the parent window)
      fHeight = 1 ;
          if (h >=0 && h < 1)
             fHeight =   h;
          else if (h < 0)
             fHeight = 0;
}

//______________________________________________________________________________
void TWin32CommCtrl::MapToMasterWindow(Int_t *xpos, Int_t *ypos, Int_t *h, Int_t *w)
{
// define the size of the Master window and pixel size of the control
  HWND hwndm = fMasterWindow->GetWindow();
  RECT size;
  const Int_t pxgap = 2; // We allow "pxgap" misvalue because of rounding
  GetClientRect(hwndm, &size);
  *xpos = fXCtrl*size.right+1;
    if (size.right-(*xpos) <= pxgap)  *xpos = size.right;
        else if (*xpos <= pxgap) *xpos = 0;
  *ypos = fYCtrl*size.bottom+1;
    if (size.bottom-(*ypos) <= pxgap) *ypos = size.bottom;
        else if (*ypos <= pxgap) *ypos = 0;
  *w = fWidth*size.right+1;
    if (size.right-(*w) <= pxgap)     *w = size.right;
        else if (*w <= pxgap) *w = 0;
  *h = fHeight*size.bottom+1;
    if (size.bottom-(*h) <= pxgap)    *h = size.bottom;
        else if (*h <= pxgap) *h = 0;
}

//______________________________________________________________________________
void   TWin32CommCtrl::MoveControl()
{
//*-*  Set the control to the new position
        Int_t xpos,ypos, h, w;
        MapToMasterWindow(&xpos, &ypos, &h, &w);
        MoveWindow(fhwndWindow, // handle of window
    xpos,             // horizontal position
    ypos,             // vertical position
    w,             // width
    h,             // height
    TRUE           // repaint flag
   );
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32CommCtrl::OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*  OnCommand(hwnd,uMsg,wParam,lParam)
//*-*
//*-*  Accepts the WM_COMMAND message sent to parent windows (hwnd)
//*-*
//*-*  hwnd  - handle of the parent window  (MUST be fMasterWindow->GetWindow());
//*-*
//*-*  uMsg  - MUST be WM_COMMAND constant
//*-*
//*-*  wParam-
//*-*     wNotifyCode = HIWORD(wParam) -  notification code
//*-*     wID = LOWORD(wParam);        -  control identifier
//*-*
//*-*  lParam- handle of the control (must be equal fhwndWindow)
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
//______________________________________________________________________________
LRESULT APIENTRY TWin32CommCtrl::OnSubClassCtrl(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
        Bool_t ncevent = kFALSE;
        switch (uMsg)
        {
        case WM_NCLBUTTONDOWN:
        case WM_NCLBUTTONUP:
        case WM_NCMOUSEMOVE:
                ncevent = kTRUE;  // coordinates are relative to the upper-left corner of the screen.
        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_MOUSEMOVE:
            {
                POINT pt;
                pt.x = LOWORD(lParam);  // horizontal position of cursor
                pt.y = HIWORD(lParam);  // vertical position of cursor
                if (ncevent) ScreenToClient(fhwndWindow,&pt);

                MapWindowPoints(fhwndWindow,fMasterWindow->GetWindow(),&pt,1);
                ((TWin32BrowserImp *)fMasterWindow)->OnSizeCtrls(uMsg,pt.x,pt.y);
                break;
            }
        default:
                break;
        }
        return 0;
}


//______________________________________________________________________________
void  TWin32CommCtrl::SetStyle(DWORD style)
{
// Specifies the style of the control window being created.
    if (SetWindowLong(fhwndWindow,GWL_STYLE,style)) fWindowStyle = style;
    else
    {
        Int_t err = GetLastError();
        char buffer[100];
        sprintf(buffer,"Last Error Code = %d", err);
        Error("TWin32CommCtrl::SetStyle", buffer);
    }
}


#ifdef uuu
//______________________________________________________________________________
LRESULT APIENTRY TWin32CommCtrl::OnNotify(LPARAM lParam)
{
 // CallBack function to manage the notify messages
//*-*  We have to create a copy of the notification message to keep it
 TWin32SendClass *CodeOp = new TWin32SendClass(this,
                                   (UInt_t)(((LPNMHDR) lParam)->code),(UInt_t)lParam,0,0);
 ExecCommandThread(CodeOp);
 return 0;
}


#endif
