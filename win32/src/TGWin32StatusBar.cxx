// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   26/08/96


#ifndef ROOT_TGWin32StatusBar
#include "TGWin32StatusBar.h"
#endif

#ifndef ROOT_TGWin32WindowsObject
#include "TGWin32WindowsObject.h"
#endif

#include "Win32Constants.h"

// private:
//        TGWin32WindowsObject *fParent;       //pointer to the parent object
//        HWND                  fhwndWindow;   // The handle of the status bar window
//        HMENU                 fStatusBarID;  // The status bar ID
//        Int _t                fnParts;       // The number of the status bar parts
//        Int_t                 fParts[];      // Array to hold the size of the
                                             // status bar parts



//______________________________________________________________________________
TGWin32StatusBar::TGWin32StatusBar(TGWin32WindowsObject *win, Int_t nParts)
{
// This ctor creates the StatusBar object with <nParts> "single-size" parts
   if (!win) return;
   fParent     = win;
   fhwndWindow = 0;
   fParts      = 0;
   fnParts     = 0;
   DoCreateStatusWindow();
   SetStatusParts(win,nParts);
   OnSize();
}
//______________________________________________________________________________
TGWin32StatusBar::TGWin32StatusBar(TGWin32WindowsObject *win, Int_t *parts, Int_t nParts)
{
// parts  - an interger array of the relative sizes of each parts (in percents) //
// nParts - number of parts                                                     //

   if (!win) return;
   fParent     = win;
   fhwndWindow = 0;
   fParts      = 0;
   fnParts     = 0;
   DoCreateStatusWindow();
   SetStatusParts(win,parts,nParts);
   OnSize();
}

//______________________________________________________________________________
TGWin32StatusBar::~TGWin32StatusBar()
{
 if (fnParts <= 0) return;
 if (fParts) {
       if (fhwndWindow) DestroyWindow(fhwndWindow);
       delete [] fParts;
       fParts  = 0;
       fnParts = 0;
 }
}
//______________________________________________________________________________
void TGWin32StatusBar::DoCreateStatusWindow()
{
// DoCreateStatusWindow - creates a status window and divides it into
//     the specified number of parts.
// Returns the handle to the status window.
// hwndParent - parent window for the status window
// nStatusID - child window identifier
// hinst - handle to the application instance
// nParts - number of parts into which to divide the status window
// HWND DoCreateStatusWindow(HWND hwndParent, int nStatusID,
// HINSTANCE hinst, int nParts)

  TGWin32CreateStatusBar *CodeOp = new TGWin32CreateStatusBar();
//  fParent->ExecCommand(CodeOp);
   SendMessage(fParent->GetWindow(),
               IX11_ROOT_MSG,
               (WPARAM)CodeOp->GetCOP(),
               (LPARAM)CodeOp);

  CodeOp->Wait();
  fhwndWindow = CodeOp->GetWindow();
  delete CodeOp;

}
//______________________________________________________________________________
void TGWin32StatusBar::Draw(){;}

//______________________________________________________________________________
Int_t TGWin32StatusBar::GetHeight()
{
    RECT effectiveSize;
    GetWindowRect(GetWindow(),&effectiveSize);
    return (effectiveSize.bottom - effectiveSize.top+1);
}

//______________________________________________________________________________
void TGWin32StatusBar::OnSize()
{
// Calculate the absolute size of the all parts
  if (fnParts <= 0) return;

  RECT rcClient;

//*-*  Get the coordinates of the parent window's client area.
     GetClientRect(fParent->GetWindow(), &rcClient);

//*-*  Calculate the right edge coordinate for each part, and
//*-*  copy the coordinates to the array.
//*-*
    Int_t nWidth = rcClient.right * 0.01;
    Int_t *lpparts = new Int_t[fnParts];
    for (int i=0; i<fnParts; i++)
      lpparts[i] = (i ? lpparts[i-1] : 0) + fParts[i] * nWidth;

//*-* Tell the status window to create the window parts.
    SendMessage(fhwndWindow, SB_SETPARTS, (WPARAM) fnParts,
        (LPARAM) lpparts);

    delete [] lpparts;
}
//______________________________________________________________________________
void TGWin32StatusBar::SetFont(){;}
//______________________________________________________________________________
void TGWin32StatusBar::SetHeight(Int_t h){;}                // Set the height of the status bar

//______________________________________________________________________________
void TGWin32StatusBar::SetStatusParts(TGWin32WindowsObject *win, Int_t nParts)
{
// This creates the StatusBar object with <nParts> "single-size" parts
 if (nParts <= 0) return;
 fnParts = nParts;
 fParts = new Int_t[fnParts];
 for (int i=0;i<fnParts; fParts[i++] = 100/fnParts){;}
}

//______________________________________________________________________________
void TGWin32StatusBar::SetStatusParts(TGWin32WindowsObject *win, Int_t *parts, Int_t nParts)
{
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
// It creates the parts of the status bar                                       //
// win  - pointer to the 'parent' window                                        //
// parts  - an interger array of the relative sizes of each parts (in percents) //
// nParts - number of parts                                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
 float factor = 1.0;
 int sum = 0;
 if (nParts <= 0) return;
 fnParts = nParts;
 fParts = new Int_t[fnParts];

 for (int i=0;i<fnParts; i++)
 {
   if (parts[i] > 0)
    fParts[i] = parts[i];
   else
    fParts[i] = 0;
    sum += fParts[i];
 }

// The total size of all parts can not exceed the 100 % limit;

  factor = 100/sum;
  if (factor < 1 ) { // Scale all parts to fit 100% limit
    for (int i=0; i<nParts; fParts[i++] *= factor){;}
  }
}

//______________________________________________________________________________
void TGWin32StatusBar::SetText(const char *text, Int_t npart, Int_t stype)
{   // Draw the text into the 'npart' part of the status bar

int iPart = npart;

if (iPart < 0 || fnParts <= 0) return;

//*-* Tell the status window to set the text.
    SendMessage(fhwndWindow, SB_SETTEXT, (WPARAM) iPart  | stype,
        (LPARAM) text);

// 0    The text is drawn with a border to appear lower than the plane of the window.
// SBT_NOBORDERS        The text is drawn without borders.
// SBT_OWNERDRAW        The text is drawn by the parent window.
// SBT_POPOUT   The text is drawn with a border to appear higher than the plane of the window.

}
