// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   05/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWin32Canvas
#define ROOT_TWin32Canvas

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32Canvas                                                         //
//                                                                      //
// This class creates a main window with menubar, scrollbars and a      //
// drawing area.                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TCanvasImp
#include "TCanvasImp.h"
#endif

#include "TWin32Menu.h"
#include "TCanvas.h"
#include "TWin32Command.h"
#include "TGWin32Object.h"
#include "TGWin32WindowsObject.h"

class TWin32ContextMenuImp;
class TWin32InspectImp;

class TWin32Canvas : public TCanvasImp, public TGWin32WindowsObject {

private:

   friend  class TWin32ContextMenuImp;

   Int_t   fCanvasImpID;

   void   SetCanvas(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void   SetCanvas(const char *title);


public:

   TWin32Canvas();

   TWin32Canvas(TCanvas *c, const char *name, UInt_t width, UInt_t height);
   TWin32Canvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TWin32Canvas();

   void   DrawEventStatus(const char *text = 0, Int_t partidx = 0){SetStatusText(text,partidx);} // Set text for 'partidx field
   void   MakeMenu();

   void   FitCanvas();
   void   ForceUpdate();
   void   GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   TGWin32Object *GetWin32Obj(){ return ((TGWin32 *)gVirtualX)->GetMasterObjectbyId(fCanvasImpID);}
   void   Iconify();
   Int_t  InitWindow();
   void   NewCanvas();
   void   CreateStatusBar(Int_t nparts=1);
   void   CreateStatusBar(Int_t *parts, Int_t nparts=1);
   void   RootExec(const char *cmd);
   void   SetWindowPosition(Int_t x, Int_t y);
   void   SetWindowSize(UInt_t w, UInt_t h) { SetCanvasSize(w,h); }
   void   SetWindowTitle(const Text_t *newTitle) {SetCanvas(newTitle);}
   void   SetCanvasSize(UInt_t w, UInt_t h);
   void   SetStatusText(const char *text, Int_t partidx =0); // Set Text into the 'npart'-th part of the status bar
   void   ShowMenuBar(Bool_t show);
   void   ShowStatusBar(Bool_t show=kTRUE);
   void   Show();
   void   UpdateCanvasImp();

// Menu Callbacks

   static void ClearCanvasCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void ClearPadCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void CloseCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void EditorCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void HelpCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void NewCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void OpenCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void PrintCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void QuitCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void SaveAsCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void SaveCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void SaveSourceCB(TWin32Canvas *obj, TVirtualMenuItem *item);

   static void AutoFitCanvasCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void BrowserCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void ColorsCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void EventStatusCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void FontsCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void FullTreeCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void MarkersCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void IconifyCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void X3DViewCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void InterruptCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void FitCanvasCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void RefreshCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void OptStatCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void OptTitleCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void OptFitCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void CanEditHistogramsCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void ROOTInspectCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void UnDoCB(TWin32Canvas *obj, TVirtualMenuItem *item);
   static void PartialTreeCB(TWin32Canvas *obj, TVirtualMenuItem *item);

   // ClassDef(TWin32Canvas,0)  //Win32Canvas class describing main window protocol
};

#endif
