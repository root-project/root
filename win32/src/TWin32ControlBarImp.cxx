// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   26/02/96


#include "TWin32ControlBarImp.h"
#include "TControlBar.h"
#include "TControlBarButton.h"
#include "TGWin32Object.h"
#include "TApplication.h"
#include "TWin32Application.h"
#include "TROOT.h"

#include <commctrl.h>

// ClassImp(TWin32ControlBarImp)

//______________________________________________________________________________
/////
//
// WidestBtn Function
//     Calculates the width, in pixels of the widest button in the
//     toolbar. Since toolbar controls use the same with for all
//     buttons, the return value is used when sending the
//     TB_SETBUTTONWIDTH message;
//
// Accepts:
//    LPTSTR *: Address of the array of button text strings.
//    HWND: The handle to the parent window.
//
// Returns:
//    An INT value representing the minimum width, in pixels,
//    a button must be to allow room for the button text.
/////

static INT WINAPI HeightBtn(HWND hwndOwner,LPTSTR pszStrArray, Int_t &height)
{
// The toolbar reserves pixels for space between buttons,
// text, etc. This value is added to the return value
// to compensate.
#define EXTRA_PIXELS 20
   const Int_t yExtraPixel = 20;
   const Int_t xExtraPixel = 10;

   INT      i;
   SIZE     sz;
   LOGFONT  lf;
   HFONT    hFont;
   HDC      hdc;

   // Get the font used to display button text, the select it into
   // a device context to be passed to GetTextExtentPoint32.
   SystemParametersInfo(SPI_GETICONTITLELOGFONT,sizeof(LOGFONT),&lf,0);

   hdc = GetDC(hwndOwner);
   hFont = CreateFontIndirect(&lf);
   SelectObject(hdc,hFont);

   GetTextExtentPoint32(hdc, pszStrArray,
                           strlen(pszStrArray), &sz);
   // Release the DC and font now that we're done.
   ReleaseDC(hwndOwner, hdc);
   DeleteObject(hFont);

   // Return the sum of the string width, the border and extra pixels.
   height = sz.cy + GetSystemMetrics(SM_CYBORDER) + yExtraPixel;
   return   sz.cx + GetSystemMetrics(SM_CXBORDER) + xExtraPixel;
}
#undef EXTRA_PIXELS



//______________________________________________________________________________
TWin32ControlBarImp::TWin32ControlBarImp(TControlBar *c, Int_t x, Int_t y) :
 TControlBarImp( c ){
//    fDialogFrame = 0;
   fXpos    = x;
   fYpos    = y;
   fHwndTB  = 0;
   fCreated = 0;

   fLastWidth  = 0;
   fLastHeight = 0;
}

//______________________________________________________________________________
TWin32ControlBarImp::~TWin32ControlBarImp() {;}

//______________________________________________________________________________
void TWin32ControlBarImp::Create(){

//*-*  Launch DialogFrame thread

//  if (!fDialogFrame)
//        fDialogFrame = new TDialogFrame();

//*-* Create the parent window;
//    CreateWindowsObject((TGWin32 *)gVirtualX,x,y,w,h);
    if (fXpos == -999)
       CreateWindowsObject((TGWin32 *)gVirtualX,0,0,400,500);
    else
       CreateWindowsObject((TGWin32 *)gVirtualX,fXpos,fYpos,400,500);
    fCreated = 1;

}


//______________________________________________________________________________
void TWin32ControlBarImp::ExecThreadCB(TWin32SendClass *command){
   HWND msghwnd     = (HWND)  (command->GetData(0));
   UINT msguMsg     = (UINT)  (command->GetData(1));
   WPARAM msgwParam = (WPARAM)(command->GetData(2));
   LPARAM msglParam = (LPARAM)(command->GetData(3));
   delete (TWin32SendClass *)command;

//*-*  define the button instance
   Int_t wID = LOWORD(msgwParam);
   TControlBarButton *button = GetButton(wID);

//*-*  Define the current tiltle of the parent window
//*-*             and save  it
   HWND parentwindow =  GetWindow();
   Int_t buflen = GetWindowTextLength(parentwindow);
   LPTSTR lbuf = 0;
   if (buflen) {
      lbuf = (LPTSTR) malloc((buflen+2)*sizeof(PTSTR));
      GetWindowText(parentwindow,lbuf,buflen+1);
   }

//*-*  Set a new title according the new action
   SetWindowText(parentwindow,button->GetAction());

//*-*  Implement the desired action
   button->Action();

//*-*  Change the toolbar buttom state to show "Action has been done"
   if (SendMessage((HWND)msglParam, TB_ISBUTTONCHECKED, (WPARAM) wID, 0))
                PostMessage((HWND) msglParam, TB_CHECKBUTTON, (WPARAM) wID,
                (LPARAM) MAKELONG(FALSE, 0));
   if (buflen) {
       SetWindowText(parentwindow,lbuf);
       free(lbuf);
   }
   MessageBeep(MB_OK);
  // FlashWindow(parentwindow,TRUE):

}

//______________________________________________________________________________
TControlBarButton *TWin32ControlBarImp::GetButton(Int_t id){
//*-*-*-*-*-*-*-*-*-*-*-*-* GetButton *-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*  Looking for the button with the present id
//*-*  returns the pointer to the found TControlBarButton
//*-*    otherwise
//*-*  returns 0
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*  if (id <= 0) return 0;
  TIter next(fControlBar->GetListOfButtons());
  TControlBarButton *button =0;
  Int_t buttoncounter = 0;
  //*-* Looking for the button with the present "Id"
   while(button = (TControlBarButton *)next()) {
     if (button->GetType() == TControlBarButton::kButton)
        if (++buttoncounter == id)
              return button;
   }
   return 0;
}

//______________________________________________________________________________
void TWin32ControlBarImp::Hide(){
   ShowWindow(GetWindow(),SW_HIDE);
}

//______________________________________________________________________________
void TWin32ControlBarImp::Show(){
   if (!fCreated) Create();
   ShowWindow(GetWindow(),SW_SHOW);
   if(fHwndTB) ShowWindow(fHwndTB, SW_SHOW); }

//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnClose(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //*-*    Message ID: WM_CLOSE
  //                =============
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_COMMAND
  //                =============
//    cout <<" TWin32ControlBarImp::OnCommand" << endl;
    Int_t wID = LOWORD(wParam);         // item, control, or accelerator identifier

//*-*  Check whether this button is still pressed
//*-*  The TB_GETSTATE message retrieves information about the state
//*-*  of the specified button in a toolbar
//*-*
//*-* Returns the button state information if successful or  - 1 otherwise

    if (SendMessage((HWND) lParam, TB_ISBUTTONCHECKED, (WPARAM) wID, 0))
    {

      TWin32SendClass *CodeOp = new TWin32SendClass(this,
              (UInt_t)hwnd,(UInt_t)uMsg,(UInt_t)wParam,(UInt_t)lParam);
      ExecCommandThread(CodeOp);
//      CodeOp.WaitClass();
//*-*  Change the toolbar buttom state
//      if (SendMessage((HWND) lParam, TB_ISBUTTONCHECKED, (WPARAM) wID, 0))
//                    PostMessage((HWND) lParam, TB_PRESSBUTTON, (WPARAM) wID, 0);
      return 0;
    }
    else
      return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnMouseButton(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_LBUTTONDOWN(UP) WM_MBUTTONDOWN(UP) WM_RBUTTONDOWN(UP)
  //                ================== ================== ==================
  //                WM_MOUSEMOVE
  //                ============
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-* Create ToolBar control child window;

//*-*  Ensure that the common control DLL is loaed
    InitCommonControls();

//*-*  Create a toolbar that the user can customize and that has a tooltip
//*-*  associated with it.

    UInt_t ID_TOOLBAR = 1;
    if (fHwndTB =
              CreateWindowEx(0,TOOLBARCLASSNAME, (LPSTR) NULL,
              CCS_NODIVIDER |
              WS_CHILD | TBSTYLE_TOOLTIPS | TBSTYLE_WRAPABLE | CCS_ADJUSTABLE ,
              CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, hwnd, (HMENU) ID_TOOLBAR,
              fWin32Mother->GetWin32Instance(),
              this))
    {

      SendMessage(fHwndTB, TB_BUTTONSTRUCTSIZE, (WPARAM) sizeof(TBBUTTON), 0);

//*-*  Add the bitmap containing button image to the toolbar.

      TBADDBITMAP tbab;
      tbab.hInst = HINST_COMMCTRL;
//      tbab.nID   = IDB_STD_LARGE_COLOR;
      tbab.nID   = IDB_STD_SMALL_COLOR;
      Int_t NUM_BUTTON_BITMAPS = 1;
      Int_t Bmp_Image_Idx = SendMessage(fHwndTB, TB_ADDBITMAP, (WPARAM) NUM_BUTTON_BITMAPS,
                                       (LPARAM) &tbab);
//*-*   Filling Windos NT toolbar datastructure

      TIter next(fControlBar->GetListOfButtons());
      TControlBarButton *button = 0;
      Int_t button_counter = 0;
      TBBUTTON tbb;
      Int_t NUM_BUTTONS = 1;

//      Int_t groupflag = TBSTYLE_CHECKGROUP;
      Int_t dxButton  = -1;
      Int_t nextWidth = -1;
      Int_t dyButton  =  0;
      while(button = (TControlBarButton *)next()) {
        Int_t istridx;  // Index of the button string

    //*-* Add button string to the toolbar

        istridx = SendMessage(fHwndTB,TB_ADDSTRING, (WPARAM) 0, (LPARAM) (LPSTR) button->GetName());

    //*-*  Fill the TBBUTTON array with button information, and add the buttons to the
    //*-*  toolbar

        tbb.iBitmap    = STD_PROPERTIES; // Bmp_Image_Idx;
        tbb.idCommand  = ++button_counter;
        tbb.fsState = TBSTATE_ENABLED | TBSTATE_WRAP;
        tbb.fsStyle = TBSTYLE_CHECK;
        tbb.dwData = 0;
        tbb.iString = istridx;

//        groupflag = 0;

        SendMessage(fHwndTB,TB_ADDBUTTONS, (WPARAM) NUM_BUTTONS,
                                           (LPARAM) (LPTBBUTTON) &tbb);
        nextWidth = HeightBtn(fHwndTB,(LPSTR) button->GetName(),dyButton);
        if (nextWidth > dxButton) dxButton =  nextWidth;

#ifdef draft
   //*-*  Create separator. Separator allows adjust the size of the ToolBar

        tbb.iBitmap   = 0;
        tbb.idCommand = 0;
        tbb.fsState   = 0;
        tbb.fsStyle   = TBSTYLE_SEP;
        tbb.dwData    = 0;
        tbb.iString   = 0;

        SendMessage(fHwndTB,TB_ADDBUTTONS, (WPARAM) NUM_BUTTONS,
                                           (LPARAM) (LPTBBUTTON) &tbb);
#endif
     }

//*-*   Now we can determinate the initial size of the parent window
//*-*   The TB_SETBUTTONSIZE sets the size of the buttons to be added to a toolbar.
         SendMessage(fHwndTB,TB_SETBUTTONSIZE, 0, (LPARAM) MAKELONG(dxButton, dyButton));

//*-*   Take number of buttons

     if (button_counter != fControlBar->GetListOfButtons()->GetSize())
            Error(" TWin32ControlBarImp::OnCreate","The number of the buttons is wrong");

//*-*   Take the number of columns


//*-*   Define the size of a button in the ToolBar

     SendMessage(fHwndTB,TB_GETITEMRECT,0,(WPARAM)&fButtonSize);

//*-*  Set the button size (It is supposed all buttons have the same size)

     fButtonSize.right  -= fButtonSize.left;
     fButtonSize.left    = 0;
     fButtonSize.bottom -= fButtonSize.top;
     fButtonSize.top     = 0;


#ifdef draft
 //*-*   Define the size of a button-separator in the ToolBar

     RECT separatorsize;
     SendMessage(fHwndTB,TB_GETITEMRECT,0,(WPARAM)&separatorsize);
#endif

  //*-*   Calculate the size of the window
     if (fControlBar->GetOrientation() == 1){
      fNumCol = fControlBar->GetNumberOfColumns();
      fNumRow = TMath::Max(1,TMath::Min(button_counter,(button_counter + fNumCol-1)/fNumCol));
     }
     else {
      fNumRow = fControlBar->GetNumberOfRows();
      fNumCol = TMath::Max(1,TMath::Min(button_counter,(button_counter + fNumRow-1)/fNumRow));
     }
//*-*  Re-size of the parent window
     fWin32WindowSize.left  = 0;
     fWin32WindowSize.right   = fNumCol*fButtonSize.right;
     fWin32WindowSize.bottom  = fNumRow*fButtonSize.bottom;
     fResizeFlag = kTRUE;


     OnSize(hwnd,WM_SIZE,0,MAKELPARAM(fWin32WindowSize.right,fWin32WindowSize.bottom));

//*-*
//*-* Define the total size of the non-client area of the parent window
//*-*
     RECT windowsize = fWin32WindowSize;

     fTotalXNCCDiff = windowsize.right  - windowsize.left;
     fTotalYNCCDiff = windowsize.bottom - windowsize.top;

     AdjustWindowRectEx(&windowsize,
                        GetWindowLong(hwnd,GWL_STYLE),
                        FALSE,
                        GetWindowLong(hwnd,GWL_EXSTYLE));


     fTotalXNCCDiff -= windowsize.right  - windowsize.left; // total size of the non-client area of the window
     fTotalXNCCDiff = -fTotalXNCCDiff;

     fTotalYNCCDiff -= windowsize.bottom - windowsize.top;
     fTotalYNCCDiff = -fTotalYNCCDiff;

//*-*  set window it its place

     OnExitSizeMove(hwnd,WM_EXITSIZEMOVE,0,0);  // Immitate ExitSizeMove event to set the right number of rows

     return 0;
    }   //  Create toolbar loop
    else {
      int err = GetLastError() ;
      Error(" TWin32ControlBarImp::OnCreate()", "Failed to Create Toolbar");
//      cout << "*** Last error code: " << err << endl;
     return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

}
//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnGetMinMaxInfo(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*    Message ID: WM_GETMINMAXINFO
//*-*                ================
//*-* The WM_GETMINMAXINFO message is sent to a window when the size or position
//*-* of the window is about to change. An application can use this message to
//*-* override the window's default maximized size and position, or its default
//*-* minimum or maximum tracking size.
  LPMINMAXINFO lpmmi = (LPMINMAXINFO) lParam; // address of structure

//*-* Set minimum size of tne window to be equal to the single buttom size
//  lpmmi->ptMinTrackSize.x = fButtonSize.right + fTotalXNCCDiff-1;
 //  lpmmi->ptMinTrackSize.x = fButtonSize.right + fTotalXNCCDiff;
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
  return 0;
}
//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnNotify(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_NOTIFY
  //                =========
    switch (((LPNMHDR) lParam)->code){
        case TTN_NEEDTEXT:
          {
            LPTOOLTIPTEXT lpttt = (LPTOOLTIPTEXT) lParam;
            lpttt->hinst     = NULL;
            Int_t idButton  = lpttt->hdr.idFrom;
            lpttt->lpszText = "This is a hint";


            TIter next(fControlBar->GetListOfButtons());
            TControlBarButton *button = GetButton(idButton);
            if (button)
                   lpttt->lpszText = (char *) button->GetTitle();
            break;
          }
        default:
          return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
        return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnPaint      (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_PAINT
  //                =======
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnSize     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
//    Message ID: WM_SIZE
//                =======
//    cout <<" TWin32ControlBarImp::OnSize" << endl;

    if (!fHwndTB) return DefWindowProc(hwnd, uMsg, wParam, lParam);

//*-* remember the last size the size of the Control bar

    fLastWidth  = LOWORD(lParam);  // width of client area
    fLastHeight = HIWORD(lParam);  // height of client area

    return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnExitSizeMove(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//    Message ID: WM_EXITSIZEMOVE
    Int_t nWidth = fLastWidth - fWin32WindowSize.right;  // user delta
    Int_t ixdir = 0; // flag whether windows is being changing along x-direction
    if      (nWidth > 0)   ixdir = +1;
    else if (nWidth < 0) { ixdir = -1; nWidth = -nWidth; }

    Int_t nHeight = fLastHeight - fWin32WindowSize.bottom;
    Int_t iydir = 0; // flag whether windows is being changing along y-direction
    if      (nHeight > 0)  iydir = +1;
    else if (nHeight < 0) {iydir = -1; nHeight = -nHeight; }

//*-*  If both the size is being changed both directions
    if (ixdir*iydir)  { //*-*  follows the "controlbar orientation"
      if (fControlBar->GetOrientation() == 1) ixdir = 0;
      else iydir = 0;
    }
//*-* Define direction
    Int_t button_counter = fControlBar->GetListOfButtons()->GetSize();

    Int_t addButton = 0;
//*-* Change windows size in this direction
    if (ixdir) {
       addButton = ixdir*(nWidth + fButtonSize.right - 1)/fButtonSize.right;
       fNumCol += addButton;
       if (fNumCol < 1)                   { fNumCol = 1; fNumRow =  button_counter;        }
       else if (fNumCol > button_counter) { fNumCol = button_counter; fNumRow = 1;         }
       else                               { fNumRow = (button_counter + fNumCol-1)/fNumCol;}
    }

    if (iydir) {
       addButton = iydir*(nHeight + fButtonSize.bottom - 1)/fButtonSize.bottom;
       fNumRow  += addButton;
       if (fNumRow < 1)                   { fNumRow  = 1; fNumCol =  button_counter;       }
       else if (fNumRow > button_counter) { fNumRow  = button_counter; fNumCol = 1;        }
       else                               { fNumCol = (button_counter + fNumRow-1)/fNumRow;}
    }

    fResizeFlag = kTRUE;

    fWin32WindowSize.right  = fNumCol*fButtonSize.right  + 1;
    fWin32WindowSize.bottom = fNumRow*fButtonSize.bottom + 2;

    RECT rect;
    GetWindowRect(hwnd,&rect);
    MoveWindow(hwnd,rect.left,rect.top,
               fWin32WindowSize.right  + fTotalXNCCDiff,
               fWin32WindowSize.bottom + fTotalYNCCDiff,
               TRUE);

    SendMessage(fHwndTB,TB_SETROWS,MAKEWPARAM(fNumRow+1, TRUE),(LPARAM)&rect);
// second approach
    fNumRow = SendMessage(fHwndTB,TB_GETROWS,0,0);
    fNumCol = (button_counter + fNumRow-1)/fNumRow;
    fWin32WindowSize.right  = fNumCol*fButtonSize.right  + 1;
    fWin32WindowSize.bottom = fNumRow*fButtonSize.bottom + 2;

    GetWindowRect(hwnd,&rect);
    MoveWindow(hwnd,rect.left,rect.top,
               fWin32WindowSize.right  + fTotalXNCCDiff,
               fWin32WindowSize.bottom + fTotalYNCCDiff,
               TRUE);
   return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnEraseBkgnd(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-*    Message ID: WM_ERASEBKGND
//*-*                =============
    HDC hdc = (HDC) wParam; // handle of device context
    RECT rect;
    GetClientRect(hwnd,&rect);
    FillRect(hdc, &rect, (HBRUSH) (COLOR_BACKGROUND+1));
    return 1;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnSizing     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
//*-*    Message ID: WM_SIZING
//*-*                =========
//*-* The WM_SIZING message is sent to a window that the user is resizing.
//*-* By processing this message, an application can monitor the size and
//*-* position of the drag rectangle and, if needed, change its size or
//*-* position.
//*-*
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnSysCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_SYSCOMMAND
  //                =============
//    printf(" TWin32ControlBarImp::OnSysCommand \n");
    delete this;
    return 0;
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//__________ ____________________________________________________________________
LRESULT APIENTRY TWin32ControlBarImp::OnRootInput(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_USER+10 OnRootInput
  //                ==========
    printf(" TWin32ControlBarImp::OnRootInput \n");
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

