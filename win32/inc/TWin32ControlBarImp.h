// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   26/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TWin32ControlBarImp
#define ROOT_TWin32ControlBarImp


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TWin32ControlBarImp                                                        //
//                                                                            //
// is an implemetation of the ControlBarImp ABC class                         //
// describing GUI independent control bar  for WIN32 API                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TControlBarImp
#include "TControlBarImp.h"
#endif

#ifndef ROOT_TWin32WindowABC
#include "TWin32WindowABC.h"
#endif

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOT_TGWin32WindowsObject
#include "TGWin32WindowsObject.h"
#endif

#ifndef ROOT_TWin32HookViaThread
#include "TWin32HookViaThread.h"
#endif

// class TDialogFrame;
class TControlBarButton;
class TGWin32Command;
class TWin32ControlBarImp : protected TWin32HookViaThread, public TControlBarImp, public TGWin32WindowsObject {

 private:

// static TDialogFrame *fDialogFrame;  // The common frame for all Win32 dialogs
  HWND   fHwndTB;        // WIN32 handle for the ToolBar Window
  RECT   fButtonSize;    // The size of the single button;
  Int_t  fCreated;       // The object was created (=1) It is used within Show();
  Int_t  fNumCol;        // The number of the buttons in the row
  Int_t  fNumRow;        // The number of the buttons in the col
  Int_t  fTotalXNCCDiff; // total size of the non-client arean of the window
  Int_t  fTotalYNCCDiff;
  Bool_t fResizeFlag;    // Flag to mark the resize operation
  Int_t  fLastWidth;     // The last size of the bar window set by WM_SIZE message
  Int_t  fLastHeight;    //



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
  //    Message ID: WM_ERASEBKGND
  //                ===============
  virtual LRESULT APIENTRY OnEraseBkgnd(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  //    Message ID: WM_EXITSIZEMOVE
  //                ===============
  virtual LRESULT APIENTRY OnExitSizeMove(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  //    Message ID: WM_WM_GETMINMAXINFO
  //                ===================
  virtual LRESULT APIENTRY OnGetMinMaxInfo(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

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

  //    Message ID: WM_SIZING
  //                =======
  virtual LRESULT APIENTRY OnSizing   (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  //    Message ID: WM_SYSCOMMAND
  //                =============
  virtual LRESULT APIENTRY OnSysCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  //    Message ID: WM_USER+10 OnRootInput
  //                ==========
  virtual LRESULT APIENTRY OnRootInput(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  void  ExecThreadCB(TWin32SendClass *command);

 public:

   TWin32ControlBarImp(TControlBar *c = 0, Int_t x = -999, Int_t y = -999);
   virtual ~TWin32ControlBarImp();

   void Create();
   TControlBarButton *GetButton(Int_t id);
   void Hide();
   void Show();

   // ClassDef(TWin32ControlBarImp,0) //Control bar implementation
};

#endif
