// @(#)root/win32:$Name:  $:$Id: TWin32Dialog.h,v 1.1.1.1 2000/11/27 22:57:28 fisyak Exp $
// Author: Valery Fine   19/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32Dialog
#define ROOT_TWin32Dialog

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#include "TWin32CallBackList.h"

typedef struct {
       LPDLGTEMPLATE lpdt;
       LPWORD  lpw;
       LPDLGITEMTEMPLATE lpdit;
 } Dialog_t;

//*-* Following are the class atom values for predefined controls:

typedef enum {
    kWButton = 0x0080, kWEdit, kWStatic, kWList_box, kWScroll_bar, kWCombo_box
} EWinDialogControls;

BOOL CALLBACK OnInitDialog(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
BOOL CALLBACK DlgROOT(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

class TWin32Dialog : public TNamed {

 private:

//*-*   Variable to establish the Dialog Box */

    Int_t   fLeftMargin;
    Int_t   fVertStep;
    Int_t   fyDefSize;

    UINT    fDialogResult;
    HWND    fDialogWindows;  //  dialog box handle

//*-*  Working variables to create a dialog template and dialog items

    LPWORD   fHgbl;
    Dialog_t fDlgPointers;

//*-*  The mother window

    HWND     fWindow;
    Int_t    fFirstItemID;


 TWin32CallBackList  fDialogCallBackList;



 public:

  TWin32Dialog();
  TWin32Dialog(HWND hwnd, char *name="ContextDialog", const char *title="Dialog", Int_t itemID=1000);
 ~TWin32Dialog();
  void AttachControlItem(LPPOINT lpPoint,LPSIZE lpSize,
                           DWORD lStyle,  DWORD lExtStyle,
                           char *lpszTitle,
                           WORD IdControl, EWinDialogControls wType);

  BOOL   CallCallback(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  void   Draw();
  Int_t  GetLeftMargin(){ return fLeftMargin;}
  Int_t  GetVertStep(){ return fVertStep;}
  Int_t  GetyDefSize(){ return fyDefSize;}
  Int_t  GetHeight(){return fDlgPointers.lpdt->cy;}
  Int_t  GetFirstID(){ return fFirstItemID;}
  Int_t  GetWidth(){return fDlgPointers.lpdt->cx;}
  HGLOBAL GetDlgTemplate(){ return fHgbl; }

  BOOL CALLBACK OnInitDialog(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  BOOL CALLBACK OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  void   SetNumberOfControls(Int_t n){fDlgPointers.lpdt->cdit = n;}
  void   SetHeight(Int_t y){fDlgPointers.lpdt->cy = y;}
  void   SetWidth(Int_t x){fDlgPointers.lpdt->cx = x;}
  void   Win32CreateCallbacks();
  UINT   GetDialogResult(){return fDialogResult;}

  // ClassDef(TWin32Dialog,0)
};
#endif
