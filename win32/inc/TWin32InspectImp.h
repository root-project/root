// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   09/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32InspectImp
#define ROOT_TWin32InspectImp

///////////////////////////////////////////////////////////////
//                                                           //
//   TWin32InspectImp is a special WIN32 type of browser     //
//   to back TObject::Inspect  member function               //
//                                                           //
///////////////////////////////////////////////////////////////

#ifndef ROOT_TInspectorImp
#include "TInspectorImp.h"
#endif
#ifndef ROOT_TWin32BrowserImp
#include "TWin32BrowserImp.h"
#endif


class TWin32InspectImp : public TInspectorImp, public TWin32BrowserImp {

private:
  const TObject  *fObject;             // Pointer to displayed object
  TObjArray       fWin32CommCtrls;     // Array of the TWin32CommCtrl objects
  Int_t           fCreated;            // flag whether the object was created

  HIMAGELIST      fhSmallIconList;     // List of the small icons
  HIMAGELIST      fhNormalIconList;    // List of the normal icons

  void    CreateIcons(); // Create the list of the general icons

protected:

  virtual void AddValues();
  virtual void CreateInspector(const TObject *obj);
  virtual void MakeHeaders();
  virtual void MakeMenu();
  virtual void MakeTitle();
  virtual void MakeStatusBar();
  virtual void MakeToolBar();
  virtual void RecursiveRemove(TObject *obj);

public:
  TWin32InspectImp();
  TWin32InspectImp(const TObject *obj, const char *title, UInt_t width=400, UInt_t height=300);
  TWin32InspectImp(const TObject *obj, const char *title, Int_t x, Int_t y, UInt_t width=400, UInt_t height=300);
  virtual ~TWin32InspectImp();

  virtual HWND            GetCtrlHandle(Int_t indx=kID_LISTVIEW){ return GetCommCtrl(indx)->GetWindow(); } // Return a WIN32 Window handle for selected control
  virtual TWin32CommCtrl *GetCommCtrl(Int_t indx=kID_LISTVIEW){ return (TWin32CommCtrl *) fWin32CommCtrls[(Int_t) indx]; } // Return a pointer to the selected TWin32CommCtrl class

  virtual void Shift(UInt_t lParam);
  virtual void Show();

  LRESULT APIENTRY OnClose  (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  LRESULT APIENTRY OnCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  LRESULT APIENTRY OnCreate (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  LRESULT APIENTRY OnPaint  (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  LRESULT APIENTRY OnMouseButton(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  LRESULT APIENTRY OnNotify     (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  LRESULT APIENTRY OnSize       (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  LRESULT APIENTRY OnSysCommand (HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  Bool_t OnSizeCtrls(UINT uMsg,LONG x, LONG y);

//*-*   CallBack functions

  virtual void NewCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void OpenCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void SaveCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void SaveAsCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void PrintCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void CloseCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void CutCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void CopyCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void PasteCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void SelectAllCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void InvertSelectionCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void ToolBarCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void StatusBarCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void LargeIconsCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void SmallIconsCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void DetailsCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
  virtual void RefreshCB(TWin32InspectImp *obj, TVirtualMenuItem *item);
};

#endif
