// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   01/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWin32CommCtrl
#define ROOT_TWin32CommCtrl

////////////////////////////////////////////////////////
//                                                    //
// TWin32CommCtrl - class to use WIN32 common controls//
//                  for TGWin32WindowsObject objects  //
//                                                    //
////////////////////////////////////////////////////////

#include "TGWin32WindowsObject.h"

#ifndef ROOT_TWin32HookViaThread
#include "TWin32HookViaThread.h"
#endif

#include <commctrl.h>

typedef LRESULT (CALLBACK  *CallBack_t)();

class TWin32SendClass;

enum  EWin32CtrlIDs {kID_TREEVIEW=3, kID_LISTVIEW };
enum  EIconIndex {knClosed};

class TWin32CommCtrl : protected TWin32HookViaThread, public TObject {
private:
   const Text_t      *fWindowType;     // pointer to registered window class name (class in term of the WIN32 API not C++)
   UINT               fWindowStyle;    // Specifies the style of the control window being created.
   Float_t            fXCtrl,fYCtrl;   // The coordinats of the control in the reative units
   Float_t            fWidth, fHeight; // The size of the control in the reative units (=1.0 means as large as the parent window
   WNDPROC            fPrevWndFunc;    // Entry points to save the "native" windows procedure of the control
   UInt_t             fItemID;         // ID of this control for the parent window (to handle WM_COMMAND msgs)

   void MapToMasterWindow(Int_t *xpos, Int_t *ypos, Int_t *h, Int_t *w); // Map the relaitve units to pixel ones
   void SetSubClass();    // Subclass this control

   friend class TWin32SimpleEditCtrl;

protected:
   HWND               fhwndWindow;
   Bool_t             fMouseInit;     // Flag to take over the mouse
   Bool_t             fSizeChanged;   // Flag to mark the border of the control is moving
   TGWin32WindowsObject *fMasterWindow;

   virtual void ExecThreadCB(TWin32SendClass *command){ ; }

public:
   TWin32CommCtrl(){fhwndWindow=0;} //default ctor
   TWin32CommCtrl(TGWin32WindowsObject *winobj,const Text_t *title, Float_t x,Float_t y, Float_t width, Float_t  height, const Text_t *type, UInt_t style);
   virtual ~TWin32CommCtrl();  // default dtor
   virtual void             Add(TObject *obj, const char *name) = 0;
   virtual void             CreateAccessories() = 0;                     // Create some extra things depends of the type of the control.
   HWND                     GetWindow(){return fhwndWindow;}             // return a window handle
   virtual void             MoveControl();                               // Set the control to the new position
   virtual LRESULT APIENTRY OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam); // CallBack function to manage the WM_COMMAND parent's msgs
   virtual LRESULT APIENTRY OnNotify(LPARAM lParam) = 0;                 // CallBack function to manage the notify messages
   UINT                     GetCommandId(){ return fItemID;}
   virtual const Text_t    *GetType(){ return fWindowType;}
   virtual UInt_t           GetItemObject(RECT *rpc)= 0;                 // returns the selected item lParam value and its position
   virtual DWORD            GetStyle() { return fWindowStyle; }
   virtual UINT             SetCommandId(UINT id);                       // To set fItem
   virtual void             SetStyle(DWORD style);    // Specifies the style of the control window being created.
   virtual void             SetType(const Text_t *type){fWindowType = type;}   // pointer to registered window class name (class in tern of the WIN32 API not C++)
   void                     SetXCtrl(Float_t x=0);
   void                     SetYCtrl(Float_t y=0);
   void                     SetWCtrl(Float_t w=1);
   void                     SetHCtrl(Float_t h=1);
   virtual void             Shift(UInt_t lParam=0){ ; }                  //  Shift items in the control


   static   LRESULT APIENTRY HookClass(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
   virtual  LRESULT APIENTRY OnSubClassCtrl(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

   // ClassDef(TWin32CommCtrl,0)   // Basic Windows WIN32 common control class
};

#endif
