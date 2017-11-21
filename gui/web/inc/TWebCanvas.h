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
#include "THttpWSHandler.h"

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
      THttpWSEngine  *fHandle;       ///<! websocket handle
      Bool_t          fReady;
      TString         fGetMenu;      ///<! object id for menu request
      Long64_t        fDrawVersion;  ///<! canvas version drawn by client
      TString         fSend;         ///<! extra data which should be send to the client
      WebConn() : fHandle(0), fReady(kFALSE), fGetMenu(), fDrawVersion(0), fSend() {}
   };

   typedef std::list<WebConn> WebConnList;

   WebConnList     fWebConn;      ///<! connections list

   TString         fAddr;         ///<! URL address of the canvas
   void           *fServer;       ///<! THttpServer, required only for direct communications from CEF or Qt5
   Bool_t          fHasSpecials;  ///<! has special objects which may require pad ranges
   Long64_t        fCanvVersion;  ///<! actual canvas version, changed with every new Modified() call

   virtual void   Lock() { }
   virtual void   Unlock() { }
   virtual Bool_t IsLocked() { return kFALSE; }

   virtual Bool_t PerformUpdate();
   virtual TVirtualPadPainter* CreatePadPainter();

   Bool_t AddCanvasSpecials(TPadWebSnapshot *master);
   TString CreateSnapshot(TPad *pad, TPadWebSnapshot *master = 0, TList *tempbuf = 0);
   TWebSnapshot *CreateObjectSnapshot(TObject *obj, const char *opt);

   TObject* FindPrimitive(const char *id, TPad *pad = 0);
   Bool_t DecodePadRanges(TPad *pad, const char *arg);
   Bool_t DecodeAllRanges(const char *arg);

   Bool_t IsAnyPadModified(TPad *pad);

   void CheckDataToSend();

   Bool_t WaitWhenCanvasPainted(Long64_t ver);

   Bool_t IsJSSupportedClass(TObject* obj);

   void ShowCmd(const char *arg, Bool_t show);

public:
   TWebCanvas();
   TWebCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height, TString addr, void *server);
   virtual ~TWebCanvas();

   virtual Int_t  InitWindow();
   virtual void   Close();
   virtual void   Show();

   virtual Bool_t ProcessWS(THttpCallArg *arg);

   virtual UInt_t GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);

   virtual void   ShowMenuBar(Bool_t show = kTRUE) { ShowCmd("Menu", show); }
   virtual void   ShowStatusBar(Bool_t show = kTRUE) { ShowCmd("StatusBar", show); }
   virtual void   ShowEditor(Bool_t show = kTRUE) { ShowCmd("Editor", show); }
   virtual void   ShowToolBar(Bool_t show = kTRUE) { ShowCmd("ToolBar", show); }
   virtual void   ShowToolTips(Bool_t show = kTRUE) { ShowCmd("ToolTips", show); }

/*
   virtual void   ForceUpdate() { }
   virtual void   Iconify() { }
   virtual void   SetStatusText(const char *text = 0, Int_t partidx = 0);
   virtual void   SetWindowPosition(Int_t x, Int_t y);
   virtual void   SetWindowSize(UInt_t w, UInt_t h);
   virtual void   SetWindowTitle(const char *newTitle);
   virtual void   SetCanvasSize(UInt_t w, UInt_t h);
   virtual void   RaiseWindow();
   virtual void   ReallyDelete();

   virtual Bool_t HasEditor() const { return kFALSE; }
   virtual Bool_t HasMenuBar() const { return kFALSE; }
   virtual Bool_t HasStatusBar() const { return kFALSE; }
   virtual Bool_t HasToolBar() const { return kFALSE; }
   virtual Bool_t HasToolTips() const { return kFALSE; }
*/

   ClassDef(TWebCanvas,0)  //ABC describing main window protocol
};

#endif
