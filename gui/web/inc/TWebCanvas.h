// Author: Sergey Linev   7/12/2016

/*************************************************************************
 * Copyright (C) 2016, Sergey Linev                                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWebCanvas
#define ROOT_TWebCanvas

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWebCanvas                                                           //
//                                                                      //
// TCanvasImp ABI implementation for Web-based GUI                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TCanvasImp
#include "TCanvasImp.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#include "THttpEngine.h"

#include <list>

class THttpWSEngine;
class TVirtualPad;
class TPad;
class TList;
class TWebSnapshot;
class TPadWebSnapshot;

class TWebCanvas : public THttpWSHandler, public TCanvasImp {

protected:

   struct WebConn {
      THttpWSEngine  *fHandle;     ///<! websocket handle
      Bool_t          fReady;
      UInt_t          fGetMenu;    ///<! object id for menu request
      Bool_t          fModified;
      WebConn() : fHandle(0), fReady(kFALSE), fGetMenu(0), fModified(kFALSE) {}
   };

   typedef std::list<WebConn> WebConnList;

   WebConnList     fWebConn;      ///<! connections list

   TString         fAddr;         ///<! URL address of the canvas
   Bool_t          fHasSpecials;  ///<!  has special objects whic may require pad ranges

   virtual void   Lock() { }
   virtual void   Unlock() { }
   virtual Bool_t IsLocked() { return kFALSE; }

   virtual Bool_t PerformUpdate();
   virtual TVirtualPadPainter* CreatePadPainter();

   TPadWebSnapshot* CreateSnapshot(TPad* pad);
   TObject* FindPrimitive(UInt_t id, TPad *pad = 0);
   Bool_t DecodePadRanges(TPad *pad, const char *arg);
   Bool_t DecodeAllRanges(const char *arg);

   Bool_t IsAnyPadModified(TPad *pad);

   void CheckModifiedFlag();

   Bool_t IsJSSupportedClass(TObject* obj);

public:
   TWebCanvas();
   TWebCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height, TString addr);
   virtual ~TWebCanvas();

   virtual Int_t  InitWindow();
   virtual void   Close();
   virtual void   Show();

   virtual Bool_t ProcessWS(THttpCallArg *arg);

   virtual UInt_t GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);

/*
   virtual void   ForceUpdate() { }
   virtual void   Iconify() { }
   virtual void   SetStatusText(const char *text = 0, Int_t partidx = 0);
   virtual void   SetWindowPosition(Int_t x, Int_t y);
   virtual void   SetWindowSize(UInt_t w, UInt_t h);
   virtual void   SetWindowTitle(const char *newTitle);
   virtual void   SetCanvasSize(UInt_t w, UInt_t h);
   virtual void   ShowMenuBar(Bool_t show = kTRUE);
   virtual void   ShowStatusBar(Bool_t show = kTRUE);
   virtual void   RaiseWindow();
   virtual void   ReallyDelete();

   virtual void   ShowEditor(Bool_t show = kTRUE) {}
   virtual void   ShowToolBar(Bool_t show = kTRUE) {}
   virtual void   ShowToolTips(Bool_t show = kTRUE) {}

   virtual Bool_t HasEditor() const { return kFALSE; }
   virtual Bool_t HasMenuBar() const { return kFALSE; }
   virtual Bool_t HasStatusBar() const { return kFALSE; }
   virtual Bool_t HasToolBar() const { return kFALSE; }
   virtual Bool_t HasToolTips() const { return kFALSE; }
*/

   ClassDef(TWebCanvas,0)  //ABC describing main window protocol
};

#endif
