// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   21/10/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWin32BrowserImp
#define ROOT_TWin32BrowserImp

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32BrowserImp                                                     //
//                                                                      //
// This class creates a main window with menubar, scrollbars and a      //
// list and a drawing area.                                             //
// The list displays all browsable ROOT classes and the drawing area    //
// contains the objects in the browsable classes. Selecting an object   //
// can create a new list and show the contents of the object in the     //
// drawing area, and so on.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TBrowserImp
#include "TBrowserImp.h"
#endif

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TOrdCollection
#include "TOrdCollection.h"
#endif

#ifndef ROOT_TGWin32WindowsObject
#include "TGWin32WindowsObject.h"
#endif

#ifndef ROOT_TWin32TreeViewCtrl
#include "TWin32TreeViewCtrl.h"
#endif

#ifndef ROOT_TWin32ListViewCtrl
#include "TWin32ListViewCtrl.h"
#endif

#ifndef ROOT_TWin32HookViaThread
#include "TWin32HookViaThread.h"
#endif

#ifndef ROOT_TContextMenu
#include "TContextMenu.h"
#endif

class TGWin32Command;

class TCollection;
class TNamed;
class TContextMenu;

enum ETypeOfView { kMultListView=0, kTreeOnly, kListViewOnly, kBoth };

class TWin32BrowserImp : public TBrowserImp, public TGWin32WindowsObject, protected TWin32HookViaThread {

private:

   Int_t  fCreated;  // The object was created (=1) It is used within Show();
   UInt_t          fListPosition;         // X-postion of the list view in the browser
   ETypeOfView     fTreeListFlag;         // Define the view of the browser
   HTREEITEM       fhCurrentParent;       // The current parent to make children
   TV_INSERTSTRUCT fhTVInstert;           // The structure to define the order of tern in thre TreeView
   TObjArray       fWin32CommCtrls;       // Array of the TWin32CommCtrl objects

   TContextMenu   *fContextMenu;          // Browser Context Menu
   TWin32Menu     *fBrowserMenu;          // Browser menu

   Bool_t          fListBlocked;          // List view is blocked to change;
   Bool_t          fStartMoving;          // Moving the border between ctrls
   Float_t         fDelBorder;            // relative position of the border between ctrls
   Int_t           fWidth;                // Size of the client area of the browser windows (pixels)
   HCURSOR         fhCursor;              // The cursor handle to mark the moving controls

   HIMAGELIST fhSmallIconList;            // List of the small icons
   HIMAGELIST fhNormalIconList;           // List of the normal icons

   void    CreateIcons(); // Create the list of the general icons

   enum { kListDefaultWidth = 150, kObjWindowMinSize = 300, kUptoCurrent = -1 };

 protected:

  //*-*    Message ID: WM_CLOSE
  //                =============
  virtual LRESULT APIENTRY OnClose(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  //    Message ID: WM_COMMAND
  //                =============
  virtual LRESULT APIENTRY OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  //    Message ID: WM_CREATE
  //                =========
  virtual LRESULT APIENTRY OnCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  //    Message ID: WM_NOTIFY
  //                =========
  virtual LRESULT APIENTRY OnNotify(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);


  //    Message ID: WM_LBUTTONDOWN(UP) WM_MBUTTONDOWN(UP) WM_RBUTTONDOWN(UP)
  //                ================== ================== ==================
  //                WM_MOUSEMOVE
  //                ============
  virtual LRESULT APIENTRY OnMouseButton(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  //    Message ID: WM_PAINT
  //                =======
  virtual LRESULT APIENTRY OnPaint      (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  //    Message ID: WM_SIZE
  //                =======
  virtual LRESULT APIENTRY OnSize     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  //    Message ID: WM_SYSCOMMAND
  //                =============
  virtual LRESULT APIENTRY OnSysCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);


  void  ExecThreadCB(TWin32SendClass *command);


  void CreateBrowser(TBrowser *b);     // Set the general attributes for TBrowserImp class instance

//====================
   TWin32BrowserImp();  // used by Dictionary()
   void         AddList(TObject *obj, const char *name, const char *title);
//===========
   void MakeMenu();
   void MakeStatusBar();
   void MakeToolBar();


public:

   TWin32BrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height);
   TWin32BrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   ~TWin32BrowserImp();

   void                    Add(TObject *obj, const char *name);
   void                    AddToList(TObject *obj, const char *name);
   void                    DeleteSelectedItem();
   virtual HWND            GetCtrlHandle(Int_t indx){ return GetCommCtrl(indx)->GetWindow(); } // Return a WIN32 Window handle for selected control
   virtual TWin32CommCtrl *GetCommCtrl(Int_t indx){ return (TWin32CommCtrl *) fWin32CommCtrls[(Int_t) indx]; } // Return a pointer to the selected TWin32CommCtrl class
   ETypeOfView             GetViewFlag(){ return fTreeListFlag; }

   void                    Iconify();
   Bool_t                  OnSizeCtrls(UINT uMsg, LONG x, LONG y); // Adjust the size of the child controls

//   void                    SetSelectedItem(HTREEITEM item=TVI_ROOT){fhSelectedItem = item;} // Set the item as the parent to be expanded
   void                    ClearViewFlag(ETypeOfView flag){fTreeListFlag = (ETypeOfView) ( fTreeListFlag ^ (fTreeListFlag & flag));}
   Bool_t                  GetListBlocked(){return fListBlocked;}                             // Get the lock state of the list view;

   HIMAGELIST              GetSmallIconList() { return fhSmallIconList; }
   HICON                   GetSmallIcon(Int_t IconIdx) {return fhSmallIconList  ? ImageList_GetIcon(fhSmallIconList,IconIdx,ILD_NORMAL):0; }
   HICON                   GetNormalIcon(Int_t IconIdx){return fhNormalIconList ? ImageList_GetIcon(fhNormalIconList,IconIdx,ILD_NORMAL):0; }
   HIMAGELIST              GetNormalIconList(){ return fhNormalIconList; }

   void                    RecursiveRemove(TObject *obj);
   void                    SetListBlocked(Bool_t block=kTRUE){fListBlocked = block;}          // Change the lock state of the list view;
   void                    SetViewFlag(ETypeOfView flag){fTreeListFlag = (ETypeOfView) (fTreeListFlag | flag);}
   void                    Shift(UInt_t obj=0);                                               // TObject pointer to shift items
   void                    Show();
   void                    Hide();
//
// Menu static callbacks

  static void NewCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void OpenCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void SaveCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void SaveAsCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void PrintCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void CloseCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);

  static void CutCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void CopyCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void PasteCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void SelectAllCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void InvertSelectionCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);

  static void ToolBarCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void StatusBarCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void LargeIconsCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void SmallIconsCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void DetailsCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);
  static void RefreshCB(TWin32BrowserImp *obj, TVirtualMenuItem *item);

  // ClassDef(TWin32BrowserImp,0)  //Win32 version of the ROOT object browser
};

#endif
