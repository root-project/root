// @(#)root/win32:$Name:  $:$Id: TWin32BrowserImp.cxx,v 1.2 2001/05/23 16:41:25 brun Exp $
// Author: Valery Fine(fine@vxcern.cern.ch)   21/10/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32BrowserImp
#include "TWin32BrowserImp.h"
#endif

#include "TControlBar.h"
#include "TBrowser.h"
#include "TControlBarButton.h"
#include "TGWin32Object.h"
#include "TApplication.h"
#include "TWin32Application.h"
#include "TROOT.h"
#include "TInterpreter.h"

#include "TWinNTSystem.h"

#include <commctrl.h>


// ClassImp(TWin32BrowserImp)


//______________________________________________________________________________
TWin32BrowserImp::TWin32BrowserImp()
{
//   Deafult ctor for Dictionary

//   fHwndTB = 0;
   fCreated = 0;
   fTreeListFlag = kTreeOnly;
}

//______________________________________________________________________________
TWin32BrowserImp::TWin32BrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height)
{
    CreateBrowser(b);
    CreateWindowsObject((TGWin32 *)gVirtualX,0,0,width,height);
    W32_SetTitle(title && strlen(title) ? title : b->GetName());

    fMenu = new TWin32Menu("BrowserMenu",title);

    MakeMenu();

}
//______________________________________________________________________________
TWin32BrowserImp::TWin32BrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
    CreateBrowser(b);
    CreateWindowsObject((TGWin32 *)gVirtualX,x,y,width,height);
    W32_SetTitle(title && strlen(title) ? title : b->GetName());

    fMenu = new TWin32Menu("BrowserMenu",title);

    MakeMenu();
}

//______________________________________________________________________________
void TWin32BrowserImp::CreateBrowser(TBrowser *b)
{
  fBrowser = b;
  fCreated = 1;
  fListBlocked = kFALSE;
//    fTreeListFlag  = kTreeOnly;
  fTreeListFlag  = kBoth;

  //*-*  Since Microsoft destroys this list with WM_DESTROY message we had to re-create
  //*-*  the Image list of Icons each time and creates a local copy of that.
  //*-* The right place to make icon list is a TWin32Application !!!

  CreateIcons();
  fhCursor = LoadCursor(NULL,IDC_SIZEWE);

}
//______________________________________________________________________________
TWin32BrowserImp::~TWin32BrowserImp()
{
  if (fCreated){
      fCreated = 0;
      if (GetWindow())
          DestroyWindow(GetWindow());

      TWin32CommCtrl *ctrl = GetCommCtrl(kID_TREEVIEW);
      if (ctrl)
               delete ctrl;

      ctrl = GetCommCtrl(kID_LISTVIEW);
      if (ctrl)
               delete ctrl;
  }
}

//______________________________________________________________________________
void TWin32BrowserImp::Add(TObject *obj, const char *name)
{
 const char *n = name;
 if (!n) n = obj->GetName();
 if (obj->IsFolder()) AddToList(obj,n);

 if (fListBlocked) return;

 if (fTreeListFlag & kListViewOnly)
     { TWin32CommCtrl *ctrl = GetCommCtrl(kID_LISTVIEW);
       if (ctrl)
           ctrl->Add(obj,n);
     }
}
//______________________________________________________________________________
void TWin32BrowserImp::AddToList(TObject *obj, const char *name)
{

   if (fTreeListFlag & kTreeOnly)
       {
          TWin32CommCtrl *ctrl = GetCommCtrl(kID_TREEVIEW);
          if (ctrl)
               ctrl->Add(obj,name);
       }

}

//______________________________________________________________________________
void TWin32BrowserImp::CreateIcons()
{

   fhSmallIconList  = 0;
   fhNormalIconList = 0;

   fhSmallIconList = ImageList_Create(GetSystemMetrics(SM_CXSMICON),
                                      GetSystemMetrics(SM_CYSMICON),
                                      ILC_MASK,kTotalNumOfICons,1);

   fhNormalIconList = ImageList_Create(GetSystemMetrics(SM_CXICON),
                                       GetSystemMetrics(SM_CYICON),
                                       ILC_MASK,kTotalNumOfICons,1);

   HICON hicon;
   Int_t i;
   for (i=0;i<kTotalNumOfICons; i++)
   {
     ImageList_AddIcon(fhSmallIconList,((TWinNTSystem *)gSystem)->GetSmallIcon(i));
     ImageList_AddIcon(fhNormalIconList,((TWinNTSystem *)gSystem)->GetNormalIcon(i));
   }

}

//______________________________________________________________________________
void TWin32BrowserImp::Shift(UInt_t lParam)
{
// expand branch
//  fhSelectedItem = fhLastCreated;
  if (lParam && GetCommCtrl(kID_TREEVIEW))
                      GetCommCtrl(kID_TREEVIEW)->Shift(lParam);
  else if (!fListBlocked && GetCommCtrl(kID_LISTVIEW))
                      GetCommCtrl(kID_LISTVIEW)->Shift(lParam);
  return;
}

//______________________________________________________________________________
void TWin32BrowserImp::ExecThreadCB(TWin32SendClass *command){
   HWND msghwnd     = (HWND)  (command->GetData(0));
   UINT msguMsg     = (UINT)  (command->GetData(1));
   WPARAM msgwParam = (WPARAM)(command->GetData(2));
   LPARAM msglParam = (LPARAM)(command->GetData(3));
   delete (TWin32SendClass *)command;

#ifdef uuu
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
#endif
}

//______________________________________________________________________________
void TWin32BrowserImp::BrowseObj(TObject *obj)
{
  if (obj) {
    TBrowser *b = Browser();
    if (b) obj->Browse(b);
  }
}

//______________________________________________________________________________
void TWin32BrowserImp::Iconify(){ ; }

//______________________________________________________________________________
void TWin32BrowserImp::Hide(){ ; }


//______________________________________________________________________________
void TWin32BrowserImp::MakeMenu(){

Int_t iMenuLength = sizeof(fStaticMenuItems) / sizeof(fStaticMenuItems[0]);
Int_t i = 0;
TWin32Menu *PopUpMenu;

//*-*   Static data member to create menus for all canvases

 fStaticMenuItems  =  new TWin32MenuItem *[kEndOfMenu+2];

 //*-*  simple  separator
 fStaticMenuItems[i++] = new                     TWin32MenuItem(kSeparator);
 //*-*  Some other type of separators
 fStaticMenuItems[i++] = new                     TWin32MenuItem(kMenuBreak);
 fStaticMenuItems[i++] = new                     TWin32MenuItem(kMenuBarBreak);


//*-*  Main Canvas menu items
 Int_t iMainMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("File","&File",kSubMenu);
 fStaticMenuItems[i++] = new TWin32MenuItem("Edit","&Edit",kSubMenu);
 fStaticMenuItems[i++] = new TWin32MenuItem("View","&View",kSubMenu);
 Int_t iMainMenuEnd = i-1;


//*-*   Items for the File Menu

 Int_t iFileMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("New","&New",NewCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Open","&Open",OpenCB);
 fStaticMenuItems[i++] = new TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Save","&Save",SaveCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("SaveAs","Save &As",SaveAsCB);
 fStaticMenuItems[i++] = new TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Print","&Print",PrintCB);
 fStaticMenuItems[i++] = new TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Close","&Close",CloseCB);
 Int_t iFileMenuEnd = i-1;


//*-*   Items for the Edit Menu

 Int_t iEditMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("Cut","Cu&t",CutCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Copy","&Copy",CopyCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Paste","&Paste",PasteCB);
 fStaticMenuItems[i++] = new TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("SelectAll","Select &All",SelectAllCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("InvertSelection","&Invert Selection",InvertSelectionCB);
 Int_t iEditMenuEnd = i-1;

//*-*   Items for the View

 Int_t iViewMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("ToolBar","&Tool Bar",ToolBarCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("StatusBar","&Status Bar", StatusBarCB);
 fStaticMenuItems[i++] = new TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("LargeIcons","&Large Icons",LargeIconsCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("SmallIcons","&Small Icons",SmallIconsCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Details","&Details",DetailsCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("ArrangeIcons","&Arrange Icons",kSubMenu);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Refresh","&Refresh",RefreshCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Options","&Options",kSubMenu);
 Int_t iViewMenuEnd = i-1;

 Int_t iEndOfMenu =  i-1;

 iMenuLength = i;
//*-* Create full list of the items

 for (i=0;i<=iEndOfMenu;i++)
    RegisterMenuItem(fStaticMenuItems[i]);

//*-*  Create static menues (one times for all Canvas ctor)


//*-* File
   PopUpMenu = fStaticMenuItems[kMFile]->GetPopUpItem();
      for (i=iFileMenuStart;i<=iFileMenuEnd; i++)
        PopUpMenu->Add(fStaticMenuItems[i]);

//*-* Edit
   PopUpMenu = fStaticMenuItems[kMEdit]->GetPopUpItem();
     for (i=iEditMenuStart;i<=iEditMenuEnd; i++)
       PopUpMenu->Add(fStaticMenuItems[i]);

//*-* View
   PopUpMenu = fStaticMenuItems[kMView]->GetPopUpItem();
     for (i=iViewMenuStart;i<=iViewMenuEnd; i++)
       PopUpMenu->Add(fStaticMenuItems[i]);

//*-*  Create main menu
     for (i=iMainMenuStart;i<=iMainMenuEnd; i++)
       fMenu->Add(fStaticMenuItems[i]);

//*-*  Glue this menu onto the canvas window

   // W32_SetMenu(fMenu->GetMenuHandle());

}

//______________________________________________________________________________
void TWin32BrowserImp::MakeStatusBar()
{
  // fStatusBar = new TGWin32StatusBar(this);
}

//______________________________________________________________________________
void TWin32BrowserImp::MakeToolBar()
{
}

//______________________________________________________________________________
void TWin32BrowserImp::RecursiveRemove(TObject *obj)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::Show(){
   if (!fCreated) return;

   HWND win = GetCtrlHandle(kID_TREEVIEW);
   if(win) ShowWindow(win, SW_SHOW);

   win = GetCtrlHandle(kID_LISTVIEW);
   if(win) ShowWindow(win, SW_SHOW); }

//______________________________________________________________________________
LRESULT APIENTRY TWin32BrowserImp::OnClose(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //*-*    Message ID: WM_CLOSE
  //                =============
    CloseCB(this,NULL);
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32BrowserImp::OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  //    Message ID: WM_COMMAND
  //                =============
    return TGWin32WindowsObject::OnCommand(hwnd,uMsg,wParam,lParam);

//    cout <<" TWin32BrowserImp::OnCommand" << endl;
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
Bool_t TWin32BrowserImp::OnSizeCtrls(UINT uMsg,LONG x, LONG y)
{
  Int_t xPos = x;
  Int_t xAbs = fDelBorder*fWidth;

//  printf(" Cursor pos %d border pos = %d \n", xPos, xAbs);

  if (!fStartMoving && TMath::Abs(xAbs-xPos) > 5) return kFALSE;

  SetCursor(fhCursor);

  switch (uMsg)
  {
    case WM_NCLBUTTONDOWN:
    case WM_LBUTTONDOWN:
                              fStartMoving = kTRUE; break;
    case WM_NCLBUTTONUP:
    case WM_LBUTTONUP:
                              fStartMoving = kFALSE; break;
  default: break;
  }

  if (!fStartMoving)              return kTRUE;

  fDelBorder = ((Float_t)xPos)/fWidth;
  TWin32CommCtrl *ctrl = GetCommCtrl(kID_TREEVIEW);
  if (ctrl)
  {
      ctrl->SetWCtrl(fDelBorder);
      ctrl->MoveControl();
   }
   ctrl = GetCommCtrl(kID_LISTVIEW);
   if (ctrl)
   {
      ctrl->SetXCtrl(fDelBorder);
      ctrl->SetWCtrl(1.0-fDelBorder);
      ctrl->MoveControl();
   }
   return kTRUE;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32BrowserImp::OnMouseButton(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  //    Message ID: WM_LBUTTONDOWN(UP) WM_MBUTTONDOWN(UP) WM_RBUTTONDOWN(UP)
  //                ================== ================== ==================
  //                WM_MOUSEMOVE
  //                ============
    if (uMsg != WM_MOUSEMOVE){} // printf(" TWin32BrowserImp::OnMouseButton \n");
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32BrowserImp::OnCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-* Create TreeView control child window;

//*-*  Create a toolbar that the user can customize and that has a tooltip
//*-*  associated with it.
//*-*  Check system error

    Float_t w = 1;
    SetWindow(hwnd);
    if (!((fTreeListFlag & kBoth) ^ kBoth)) w = 0.33;
        fDelBorder = w;
    if (fTreeListFlag & kTreeOnly)
              fWin32CommCtrls[kID_TREEVIEW] =
                                       new TWin32TreeViewCtrl(this,"Root Tree",0.0,0.0,w,1.0);

//*-*  Create ListView control

    if ((fTreeListFlag & kListViewOnly) || fTreeListFlag == kMultListView)
              fWin32CommCtrls[kID_LISTVIEW] =
                                       new TWin32ListViewCtrl(this,"Root Objects",w,0.0,1-w,1.0);
    return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32BrowserImp::OnNotify(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
//    Message ID: WM_NOTIFY
//                =========
 TWin32CommCtrl *ctrl = GetCommCtrl((Int_t) wParam);

 if (ctrl)
 {
   LRESULT res = ctrl->OnNotify(lParam);
   if (!res) return res;

   RECT prc;
//*-*  Get ctrl coordinat
   TObject *obj = (TObject *)ctrl->GetItemObject(&prc);
   fBrowser->SetSelected(obj);
//*-* Convert the ctrl coordinat to the Browser windows coordinats
   MapWindowPoints(ctrl->GetWindow(),hwnd,(POINT *)(&prc),2);
//*-*     The TObject has been detected, create PopUp menu
   TContextMenu *menu = fBrowser->GetContextMenu();
   if (obj && menu)
               menu->Popup(prc.left,prc.bottom, obj,fBrowser);
   return 0;
 }
 else
     return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32BrowserImp::OnPaint      (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_PAINT
  //                =======
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32BrowserImp::OnSize     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
//    Message ID: WM_SIZE
//                =======
//    cout <<" TWin32BrowserImp::OnSize" << endl;

//*-* Adjust the size of the Control bar

        fWidth  = LOWORD(lParam);
    TWin32CommCtrl *ctrl = GetCommCtrl(kID_TREEVIEW);
    if (ctrl)
                ctrl->MoveControl();
        ctrl = GetCommCtrl(kID_LISTVIEW);
    if (ctrl)
                ctrl->MoveControl();
    return 0;
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32BrowserImp::OnSysCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_DESTROY
  //                =============
 //   printf(" TWin32BrowserImp::OnSysCommand \n");
   return DefWindowProc(hwnd, uMsg, wParam, lParam);
}


//*-*   CallBack functions

//______________________________________________________________________________
void TWin32BrowserImp::NewCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::OpenCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::SaveCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::SaveAsCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::PrintCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::CloseCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
//    Printf("TWin32BrowserImp::CloseCB \n");
    if (obj) {
      gInterpreter->DeleteGlobal(obj->Browser());
      delete obj->Browser();
    }
}

//______________________________________________________________________________
void TWin32BrowserImp::CutCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::CopyCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::PasteCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::SelectAllCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::InvertSelectionCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::ToolBarCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::StatusBarCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::LargeIconsCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::SmallIconsCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::DetailsCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32BrowserImp::RefreshCB(TWin32BrowserImp *obj, TVirtualMenuItem *item)
{
}
    
// Default actions
 
#define defAction { return DefWindowProc(hwnd,uMsg, wParam, lParam); }
 
LRESULT APIENTRY TWin32BrowserImp::OnActivate         (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnCommandForControl(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnEraseBkgnd       (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnGetMinMaxInfo    (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnMouseActivate    (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnPaletteChanged   (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnSetFocus         (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnKillFocus        (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnSizing           (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnExitSizeMove     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
 
LRESULT APIENTRY TWin32BrowserImp::OnChar             (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
LRESULT APIENTRY TWin32BrowserImp::OnKeyDown          (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) defAction
  


