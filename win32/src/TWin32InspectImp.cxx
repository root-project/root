// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   09/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWin32InspectImp.h"
#include "TWin32ListViewCtrl.h"
#include "TClass.h"
#include "TRealData.h"
#include "TDataType.h"
#include "TDataMember.h"
#include "TLink.h"
#include "TDatime.h"
#include "TWinNTSystem.h"

#include <commctrl.h>


// ClassImp(TWin32InspectImp)

///////////////////////////////////////////////////////////////
//                                                           //
//   TWin32InspectImp is a special WIN32 object to implement //
//   TObject::Inspect  member funection                      //
//                                                           //
///////////////////////////////////////////////////////////////


//______________________________________________________________________________
TWin32InspectImp::TWin32InspectImp()
{
//   Deafult ctor for Dictionary

//   fCreated = 0;
}

//______________________________________________________________________________
TWin32InspectImp::TWin32InspectImp(const TObject *obj, const char *title,
                                   UInt_t width, UInt_t height)
{
    CreateWindowsObject((TGWin32 *)gVirtualX,0,0,width,height);
    CreateInspector(obj);

//    fMenu = new TWin32Menu("InspectMenu",title);

//    MakeMenu();

}
//______________________________________________________________________________
TWin32InspectImp::TWin32InspectImp(const TObject *obj, const char *title,
                                  Int_t x, Int_t y, UInt_t width, UInt_t height)
{
    CreateWindowsObject((TGWin32 *)gVirtualX,x,y,width,height);
    CreateInspector(obj);

//    fMenu = new TWin32Menu("BrowserMenu",title);

//    MakeMenu();
}

//______________________________________________________________________________
void TWin32InspectImp::CreateInspector(const TObject *obj)
{
  fObject = obj;
//  fCreated = 1;

  //*-*  Since Microsoft destroys this list with WM_DESTROY message we had to re-create
  //*-*  the Image list of Icons each time and creates a local copy of that.
  //*-* The right place to make icon list is a TWin32Application !!!

  CreateIcons();
//  fhCursor = LoadCursor(NULL,IDC_SIZEWE);
  MakeTitle();
  MakeHeaders();
  AddValues();
}

//______________________________________________________________________________
void TWin32InspectImp::MakeHeaders()
{

  char *headers[3];
  headers[0] = "Member Name";
  headers[1] = "Value";
  headers[2] = "Title";

  Int_t widths[]  = {96,120,320};
  TWin32ListViewCtrl *ctrl = ( TWin32ListViewCtrl *)GetCommCtrl();
  if (!ctrl) return;
  ctrl->AddColumns((const char **)headers,3,widths);
}

//______________________________________________________________________________
void TWin32InspectImp::MakeTitle()
{
    TClass *cl = fObject->IsA();
    if (cl == 0) return;

    char buffer[1024];
    sprintf(buffer, "%s   :   %s:%d   -   \"%s\" ", cl->GetName(),
                                           fObject->GetName(),
                                           fObject->GetUniqueID(),
                                           fObject->GetTitle());
    W32_SetTitle(buffer);
}

//______________________________________________________________________________
void TWin32InspectImp::AddValues()
{
    Int_t cdate = 0;
    Int_t ctime = 0;
    UInt_t *cdatime = 0;
    Bool_t isdate = kFALSE;
    enum {kname, kvalue, ktitle};
    TWin32ListViewCtrl *ctrl = ( TWin32ListViewCtrl *)GetCommCtrl();
    if (!ctrl) return;

    char *line[3];

    line[kname] = " ";
    line[kvalue] = new char[255];
    line[ktitle] = " ";

    TClass *cl = fObject->IsA();
    if (cl == 0) return;
    if (!cl->GetListOfRealData()) cl->BuildRealData();

//*-*- count number of data members in order to resize the canvas
    TRealData *rd;
    TIter      next(cl->GetListOfRealData());
    Int_t nreal = cl->GetListOfRealData()->GetSize();
    if (nreal == 0)  return;

//*-*  Prepare a list view control for adding a large number of items
//*-*
    ctrl->SetItemCount(nreal);

    while (rd = (TRealData*) next()) {
       TDataMember *member = rd->GetDataMember();
       TDataType *membertype = member->GetDataType();
       isdate = kFALSE;
       if (strcmp(member->GetName(),"fDatime") == 0 && strcmp(member->GetTypeName(),"UInt_t") == 0)
       {
          isdate = kTRUE;
       }
//*-*- Encode data member name
       line[kname] = (char *)(rd->GetName());

//*-*- Encode data value or pointer value
       Int_t offset = rd->GetThisOffset();
       char *pointer = (char*)fObject + offset;
       char **ppointer = (char**)(pointer);
       TLink *tlink = 0;
//       TObject *tlink = 0;
       // tlink->SetName((char *)member->GetTypeName());

       if (member->IsaPointer()) {
          char **p3pointer = (char**)(*ppointer);
          if (!p3pointer) {
             sprintf(line[kvalue],"->0");
          } else if (!member->IsBasic()) {
             sprintf(line[kvalue],"->%x ", p3pointer);
             tlink = new TLink(0, 0, p3pointer);
//             tlink = (TObject *)p3pointer;
          } else if (membertype){
               if (!strcmp(membertype->GetTypeName(), "char"))
                  sprintf(line[kvalue], "%s", *ppointer);
               else
                  strcpy(line[kvalue], membertype->AsString(p3pointer));
          }
          else if (!strcmp(member->GetFullTypeName(), "char*") ||
                   !strcmp(member->GetFullTypeName(), "const char*")) {
             sprintf(line[kvalue], "%s", *ppointer);
          } else {
             sprintf(line[kvalue],"->%x ", p3pointer);
             tlink = new TLink(0, 0, p3pointer);
//             tlink = (TObject *)p3pointer;
          }
       } else if (membertype)
            if (isdate) {
               cdatime = (UInt_t*)pointer;
               TDatime::GetDateTime(cdatime[0],cdate,ctime);
               sprintf(line[kvalue],"%d/%d",cdate,ctime);
            } else {
               strcpy(line[kvalue], membertype->AsString(pointer));
            }
       else
           sprintf(line[kvalue],"->%x ", (Long_t)pointer);

    //*-*- Encode data member title
       Int_t ltit = 0;
       if (strcmp(member->GetFullTypeName(), "char*") &&
           strcmp(member->GetFullTypeName(), "const char*")) {
          line[ktitle] = (char *)member->GetTitle();
       }
       if (tlink) {
         tlink->SetName((char *)member->GetTypeName());
         tlink->SetBit(kCanDelete);
       }
       ctrl->Add((const char **)line,3,tlink);
//       if (tlink) { Add(tlink); ctrl->Add(tlink); }
    }

}

//______________________________________________________________________________
TWin32InspectImp::~TWin32InspectImp()
{
  if (fCreated){
      fCreated = 0;
      if (GetWindow())
          DestroyWindow(GetWindow());

      TWin32CommCtrl *ctrl = GetCommCtrl();
      if (ctrl)
               delete ctrl;
  }
}


//______________________________________________________________________________
void TWin32InspectImp::CreateIcons()
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
void TWin32InspectImp::Shift(UInt_t lParam)
{
// expand branch
//  fhSelectedItem = fhLastCreated;
  GetCommCtrl(kID_LISTVIEW)->Shift(lParam);
  return;
}

#ifdef uuu
//______________________________________________________________________________
void TWin32InspectImp::ExecThreadCB(TWin32SendClass *command){
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
#endif


//______________________________________________________________________________
void TWin32InspectImp::MakeMenu(){

#ifdef draft
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
 fStaticMenuItems[i++] = new                     TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Save","&Save",SaveCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("SaveAs","Save &As",SaveAsCB);
 fStaticMenuItems[i++] = new                                  TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Print","&Print",PrintCB);
 fStaticMenuItems[i++] = new                                  TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("Close","&Close",CloseCB);
 Int_t iFileMenuEnd = i-1;


//*-*   Items for the Edit Menu

 Int_t iEditMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("Cut","Cu&t",CutCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Copy","&Copy",CopyCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("Paste","&Paste",PasteCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
 fStaticMenuItems[i++] = new TWin32MenuItem("SelectAll","Select &All",SelectAllCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("InvertSelection","&Invert Selection",InvertSelectionCB);
 Int_t iEditMenuEnd = i-1;

//*-*   Items for the View

 Int_t iViewMenuStart = i;
 fStaticMenuItems[i++] = new TWin32MenuItem("ToolBar","&Tool Bar",ToolBarCB);
 fStaticMenuItems[i++] = new TWin32MenuItem("StatusBar","&Status Bar", StatusBarCB);
 fStaticMenuItems[i++] = new                                   TWin32MenuItem(kSeparator);
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
#endif

}

//______________________________________________________________________________
void TWin32InspectImp::MakeStatusBar()
{
  // fStatusBar = new TGWin32StatusBar(this);
}

//______________________________________________________________________________
void TWin32InspectImp::MakeToolBar()
{
}

//______________________________________________________________________________
void TWin32InspectImp::RecursiveRemove(TObject *obj)
{
}

//______________________________________________________________________________
void TWin32InspectImp::Show(){
   if (!fCreated) return;

   HWND win = GetCtrlHandle(kID_TREEVIEW);
   if(win) ShowWindow(win, SW_SHOW);

   win = GetCtrlHandle(kID_LISTVIEW);
   if(win) ShowWindow(win, SW_SHOW); }

//______________________________________________________________________________
LRESULT APIENTRY TWin32InspectImp::OnClose(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //*-*    Message ID: WM_CLOSE
  //                =============
    CloseCB(this,NULL);
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32InspectImp::OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  //    Message ID: WM_COMMAND
  //                =============
    return TGWin32WindowsObject::OnCommand(hwnd,uMsg,wParam,lParam);
}

//______________________________________________________________________________
Bool_t TWin32InspectImp::OnSizeCtrls(UINT uMsg,LONG x, LONG y)
{
   return kTRUE;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32InspectImp::OnMouseButton(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  //    Message ID: WM_LBUTTONDOWN(UP) WM_MBUTTONDOWN(UP) WM_RBUTTONDOWN(UP)
  //                ================== ================== ==================
  //                WM_MOUSEMOVE
  //                ============
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32InspectImp::OnCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//*-* Create TreeView control child window;

//*-*  Create a toolbar that the user can customize and that has a tooltip
//*-*  associated with it.
//*-*  Check system error

    SetWindow(hwnd);

//*-*  Create ListView control

   fWin32CommCtrls[kID_LISTVIEW] =
                                 new TWin32ListViewCtrl(this,"Inspector Objects",0,0,1,1,WC_LISTVIEW,
                                 (DWORD) WS_VISIBLE | LVS_REPORT | LVS_SINGLESEL | LVS_AUTOARRANGE );
    return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32InspectImp::OnNotify(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
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
//*-* Convert the ctrl coordinat to the Browser windows coordinats
   MapWindowPoints(ctrl->GetWindow(),hwnd,(POINT *)(&prc),2);
#if 0
//*-*     The TObject has been detected, create PopUp menu
   TContextMenu *menu = fBrowser->GetContextMenu();
   if (obj && menu)
               menu->Popup(prc.left,prc.bottom, obj,fBrowser);
#endif
   return 0;
 }
 else
     return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32InspectImp::OnPaint      (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_PAINT
  //                =======
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32InspectImp::OnSize     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
//    Message ID: WM_SIZE
//                =======
//    cout <<" TWin32InspectImp::OnSize" << endl;

//*-* Adjust the size of the Control bar

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
LRESULT APIENTRY TWin32InspectImp::OnSysCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
  //    Message ID: WM_DESTROY
  //                =============
   return DefWindowProc(hwnd, uMsg, wParam, lParam);
}


//*-*   CallBack functions

//______________________________________________________________________________
void TWin32InspectImp::NewCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::OpenCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::SaveCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::SaveAsCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::PrintCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::CloseCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
//    Printf("TWin32InspectImp::CloseCB \n");
    if (obj) delete obj;
}

//______________________________________________________________________________
void TWin32InspectImp::CutCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::CopyCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::PasteCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::SelectAllCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::InvertSelectionCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::ToolBarCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::StatusBarCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::LargeIconsCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::SmallIconsCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::DetailsCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

//______________________________________________________________________________
void TWin32InspectImp::RefreshCB(TWin32InspectImp *obj, TVirtualMenuItem *item)
{
}

