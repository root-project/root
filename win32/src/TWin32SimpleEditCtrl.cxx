// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   01/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWin32SimpleEditCtrl.h"

///////////////////////////////////////////////////////////////
//                                                           //
// TWin32SimpleEditCtrl                                      //
//                                                           //
// Performs TVirtualX::RequestString  function under Win32        //
//                                                           //
//                                                           //
///////////////////////////////////////////////////////////////


// ClassImp(TWin32SimpleEditCtrl)


//______________________________________________________________________________
TWin32SimpleEditCtrl::TWin32SimpleEditCtrl(TGWin32WindowsObject *winobj, const char *text,Int_t lTitle, Float_t x,Float_t y, Float_t width, Float_t height, const char *type, UInt_t style)
{
//  Create a control object for the winobj object.
        // The postion and the size of the control are defined in the relative units in the 0-1 range
        // For example width =1 means the the size of the control is equal of the width
        // of the client area of the master windows object.

//*-*  Ensure that the common control DLL is loaded
  InitCommonControls();
  fEditBufferLength = lTitle;
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
//*-*  Register this control with parent
      winobj->RegisterControlItem(this);
      if (!(fhwndWindow =
          CreateWindowEx(WS_EX_LEFT,
                       (LPCTSTR) fWindowType, // predefined class
                                        NULL, // no window title
                                fWindowStyle,
                            xpos, ypos, w, h,
                  fMasterWindow->GetWindow(), // parent window
                      (HMENU) GetCommandId(), // edit control ID
                (fMasterWindow->GetWin32ObjectMother())->GetWin32Instance(),
                NULL)))
          {
                int err = GetLastError();
                printf(" Last error was %d \n", err);
                Error("OnCreate","Can't create the WIN32 common control object");
                winobj->UnRegisterControlItem(this);
          }
          else
                  SetSubClass();

  }
}

//______________________________________________________________________________
TWin32SimpleEditCtrl::~TWin32SimpleEditCtrl()
{
  if (fhwndWindow) {DestroyWindow(fhwndWindow); fhwndWindow = 0;}
}

//______________________________________________________________________________
void   TWin32SimpleEditCtrl::MoveControl()
{
//*-*  Set the control to the new position
        Int_t xpos,ypos, h, w;
        MapToMasterWindow(&xpos, &ypos, &h, &w);
        MoveWindow(fhwndWindow,  // handle of window
                   xpos,         // horizontal position
                   ypos,         // vertical position
                   w,            // width
                   h,            // height
                   TRUE          // repaint flag
        );
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32SimpleEditCtrl::OnSubClassCtrl(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
        switch (uMsg)
        {
         case WM_CHAR:
             switch(wParam)
             {
                case 0x0A: // line feed
                case 0x0D: // carriage return
                case 0x1B: // escape
//*-*  Notify the parent window about these events too
                    SendMessage(fMasterWindow->GetWindow(),uMsg,wParam,lParam);
                    return 0;
                default:
                    break;
             }
        default:
                break;
        }
        return 0;
}

//______________________________________________________________________________
Char_t  *TWin32SimpleEditCtrl::GetText(char *receiveBuffer)
{
        if (!(fEditBufferLength && receiveBuffer) ) return 0;
        SendMessage(GetWindow(),WM_GETTEXT,(WPARAM)fEditBufferLength,
                                    (LPARAM)receiveBuffer);
        return receiveBuffer;
}
