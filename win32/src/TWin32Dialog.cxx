// @(#)root/win32:$Name:  $:$Id: TWin32Dialog.cxx,v 1.1.1.1 2000/11/27 22:57:29 fisyak Exp $
// Author: Valery Fine   19/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32Dialog
#include "TWin32Dialog.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#ifndef ROOT_TObjString
#include "TObjString.h"
#endif

//______________________________________________________________________________
BOOL CALLBACK OnInitDialog
                 (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//
//  Windows Procedure to manage: WM_INITDIALOG
//

  TWin32Dialog *lpTWin32Dialog = (TWin32Dialog *)lParam;
  SetWindowLong(hwnd,GWL_USERDATA,(LONG)lpTWin32Dialog);   // tie the dialog and ROOT object
  return lpTWin32Dialog->OnInitDialog(hwnd,uMsg,wParam,lParam);
}

//______________________________________________________________________________
BOOL CALLBACK DlgROOT(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Main Universal Dialog procedure to manage all modal Win32 dialogs    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//*-* Except in response to the WM_INITDIALOG message, the dialog box procedure
//*-* should
//*-*
//*-*    return nonzero
//*-*           =======
//*-* if it processes the message, and
//*-*
//*-*    return zero
//*-*           ====
//*-* if it does not.
//*-* In response to a WM_INITDIALOG message, the dialog box procedure should
//*-*
//*-*    return zero
//*-*           ====
//*-* if it calls the SetFocus function to set the focus to one of the controls
//*-* in the dialog box. Otherwise, it should
//*-*
//*-*    return nonzero,
//*-*           =======
//*-* in which case the system sets the focus to the first control in the dialog
//*-* box that can be given the focus.

 if (uMsg == WM_INITDIALOG)
     return ::OnInitDialog(hwnd, uMsg, wParam, lParam);
 else {
#ifndef WIN32
     if (lpWin32Dialog) {
               BOOL ret_value = lpWin32Dialog->CallCallback(hwnd, uMsg, wParam, lParam);
        return ret_value;
#else
     if (uMsg == WM_COMMAND) {
        TWin32Dialog *lpWin32Dialog = ((TWin32Dialog *)GetWindowLong(hwnd,GWL_USERDATA));
        return lpWin32Dialog->OnCommand(hwnd, uMsg, wParam, lParam);
     }
#endif
     else
        return 0;
 }
}

// ClassImp(TWin32Dialog)

//______________________________________________________________________________
TWin32Dialog::TWin32Dialog()
{
//*-*   Default ctor

   fDialogWindows = 0;
   fDialogResult = -1;
}

//______________________________________________________________________________
TWin32Dialog::TWin32Dialog(HWND hwnd, char *name, const char *title, Int_t itemID) : TNamed(name,title)
{
    fDialogWindows = 0;
    fWindow      = hwnd;
    fFirstItemID = itemID;
    fDialogResult=0;

//*-*  Define a dialog box

   fHgbl =  (LPWORD)(GlobalAlloc(GMEM_ZEROINIT, 4096));

   fDlgPointers.lpdt = (LPDLGTEMPLATE) GlobalLock(fHgbl);


/* Some common values for all controls */
   Int_t   x0 = 1;
   Int_t   y0 = 2*x0;

   Int_t wId = fFirstItemID;

   fVertStep   = 12;     // one character
   fyDefSize   = fVertStep;
   fLeftMargin = x0+4;  // one character;


   fDlgPointers.lpdt->style =  DS_MODALFRAME | WS_CAPTION | WS_SYSMENU | WS_VISIBLE |
                               DS_SYSMODAL | DS_ABSALIGN | DS_3DLOOK | DS_CONTEXTHELP ;
   fDlgPointers.lpdt->dwExtendedStyle = 0;

//*-*  number of controls

   fDlgPointers.lpdt->cdit  = 3;

//*-*  Define the dialog poistion on the screen

   fDlgPointers.lpdt->x  = CW_USEDEFAULT;
   fDlgPointers.lpdt->y  = CW_USEDEFAULT;

//*-*  Define the dialog size

   fDlgPointers.lpdt->cx = 2*fLeftMargin;
   fDlgPointers.lpdt->cy = 10;

//*-*  mark the end of the Dialog template and start the item templates

    fDlgPointers.lpw = (LPWORD) (fDlgPointers.lpdt + 1);

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*  In a dialog box template, the DLGTEMPLATE structure is always immediately
//*-*  followed by three variable-length arrays that specify the menu, class, and
//*-*  title for the dialog box. When the DS_SETFONT style is given, these arrays
//*-*  are also followed by a 16-bit value specifying point size and another
//*-*  variable-length array specifying a typeface name. Each array consists of
//*-*  one or more 16-bit elements.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Menu array*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*

   *fDlgPointers.lpw++ = 0;   /* no menu */

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Class array *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
   *fDlgPointers.lpw++ = 0;   // predefined dialog box class (by default)

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Title array *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*
//*-*  Set the    Title of the dialog box
//*-*
   Int_t nchar = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, (LPCSTR)title, -1,
                                       (LPWSTR)(fDlgPointers.lpw), 1024);
   if (!nchar) {
            Int_t Err = GetLastError();
            Error("TWin32Dialog::TWin32Dialog","Wrong char conversion");
            Printf("Error number =%d \n", Err);
   }
   fDlgPointers.lpw += nchar;
}
//______________________________________________________________________________
TWin32Dialog::~TWin32Dialog()
{
    if ( fDialogResult == -1) return;
    if (fDialogResult && fDialogWindows) 
       ::EndDialog(fDialogWindows,int(fDialogResult));
    GlobalFree(fHgbl);
}

//______________________________________________________________________________
void TWin32Dialog::AttachControlItem(LPPOINT lpPoint,LPSIZE lpSize,
                                       DWORD lStyle,DWORD lExtStyle,
                                       char *lpszTitle,
                                       WORD IdControl,EWinDialogControls wType)
 {
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*AttachControlItem-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*   AttachControlItem
//*-*        creates a template of the dialog control item within dialog template
//*-*
//*-*
//*-*   LPPOINT lpPoint   - Specifies the xy-coordinates in dialog box units,
//*-*                       of the upper-left corner of the control.
//*-*   LPSIZE  lpSize    - Specifies the size, in dialog box units, of the control.
//*-*   DWORD   lStyle    - Specifies the style of the control.
//*-*                       This member can be a combination of window style values
//*-*                       (such as WS_BORDER) and one or more of the control style
//*-*                       values (such as BS_PUSHBUTTON and ES_LEFT).
//*-*   DWORD   lExtStyle - Specifies extended styles for a window.
//*-*                       This member is not used to create controls in dialog boxes,
//*-*                       but applications that use dialog box templates can use it
//*-*                       to create other types of windows.
//*-*   char   *lpszTitle -
//*-*
//*-*
//*-*
//*-*
//*-*
//*-*
//*-*
   int nchar;
/*  ALIGNITEM */
  {ULONG l;
     l = (ULONG) fDlgPointers.lpw;
     l +=3;
     l >>=2;
     l <<=2;
     fDlgPointers.lpw = (PWORD) l;
   }

  *fDlgPointers.lpw++ = LOWORD (lStyle);
  *fDlgPointers.lpw++ = HIWORD (lStyle);
  *fDlgPointers.lpw++ = LOWORD (lExtStyle);
  *fDlgPointers.lpw++ = HIWORD (lExtStyle);

  *fDlgPointers.lpw++ = (WORD) lpPoint->x;
  *fDlgPointers.lpw++ = (WORD) lpPoint->y ;
  *fDlgPointers.lpw++ = (WORD) lpSize->cx;
  *fDlgPointers.lpw++ = (WORD) lpSize->cy;
  *fDlgPointers.lpw++ =        IdControl;
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-* In a dialog box template, the DLGITEMTEMPLATE structure is always immediately
//*-* followed by three variable-length arrays specifying the class, title, and
//*-* creation data for the control. Each array consists of one or more 16-bit elements.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Class array *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*  fill in class i.d. Button in this case

  *fDlgPointers.lpw++ = (WORD)0xffff;
  *fDlgPointers.lpw++ = (WORD)wType;

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Title array *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
  /* copy the text of the first item, null terminate the string. */

   if (lpszTitle != NULL) {
     nchar = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED,(LPCSTR)lpszTitle, -1,
                                 (LPWSTR)(fDlgPointers.lpw), 2*strlen(lpszTitle));
     if (!nchar) {
              Int_t Err = GetLastError();
              Error("TWin32Dialog::TWin32Dialog","Wrong char conversion");
              Printf("Error number =%d \n", Err);
     }

     fDlgPointers.lpw += nchar;
   }
   else *fDlgPointers.lpw++ = 0;

   *fDlgPointers.lpw++ = 0;  // advance pointer over nExtraStuff WORD

}

//______________________________________________________________________________
BOOL CALLBACK TWin32Dialog::OnInitDialog(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    fDialogWindows = hwnd;
    return TRUE;
}

//______________________________________________________________________________
BOOL CALLBACK TWin32Dialog::OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
 int   lText = 0;
 int   loldText = 0;
 int   lParBlock = 0;
 char *chBuffer = 0;
 const int lBuf = 1024;
 char  chResp[lBuf];
 TObjArray *ParList=0;

  switch (LOWORD(wParam)) {
      case IDOK: {
          Int_t numChildren = fDlgPointers.lpdt->cdit;
// Create list of parameters
          ParList = new TObjArray(numChildren/2 - 1);
          Int_t index = 0;
          for( Int_t i = 0; i < ( numChildren - 2 ); i += 2 ) {

            lText = GetDlgItemText(hwnd,fFirstItemID+i+1,chResp,lBuf);
            if (lText > lBuf) {
               Warning("OnCommand","Too long parameter");
               lText = lBuf;
            }
            chResp[lText] = '\0';
//            ParList[index] = (TObject *)(new TObjString(chResp));
            ParList->AddAt((TObject *)(new TObjString(chResp)),index);
            index++;
          }
       }
                    /* Fall through. */
      case IDCANCEL:

            EndDialog(hwnd, (Int_t)ParList);
            return TRUE;
      default:
            return FALSE;

     }
}

//______________________________________________________________________________
void TWin32Dialog::Win32CreateCallbacks()
{
#ifndef WIN32
 fDialogCallBackList.AddCallBack(WM_COMMAND,(CallBack_t)TWin32Dialog::OnCommand,this);
 fDialogCallBackList.AddCallBack(WM_MBUTTONDOWN,(CallBack_t)TGWin32WindowsObject::OnMouseButton,this);
#endif

}

//______________________________________________________________________________
void  TWin32Dialog::Draw()
{

//*-*  Display this dialog

  GlobalUnlock(fHgbl);

  fDialogResult =
     DialogBoxIndirectParam((HINSTANCE)GetWindowLong(fWindow, GWL_HINSTANCE),
                            (LPDLGTEMPLATE) fHgbl,
                            fWindow,
                            (DLGPROC)::DlgROOT,(LPARAM)this);
 if ((Int_t) fDialogResult == -1) {
        Int_t err = GetLastError();
        Printf("*** Error id= %d \n", err);
        Error("TWin32Dialog::Draw()","Can;t create dialog box");
 }
}

//______________________________________________________________________________
BOOL  TWin32Dialog::CallCallback(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
#ifndef WIN32
    return fDialogCallBackList(hwnd, uMsg, wParam, lParam);
#else
   return TRUE;
#endif
}
