// Author: Sergey Linev, GSI   7/12/2016

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
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
// Basic TCanvasImp ABI implementation for Web-based GUI                //
// Provides painting of main ROOT6 classes in web browsers              //
// Major interactive features implemented in TWebCanvasFull class       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCanvasImp.h"

#include "TString.h"
#include "TList.h"

#include <ROOT/RWebWindow.hxx>

#include <vector>
#include <string>
#include <queue>
#include <functional>

class TPad;
class TPadWebSnapshot;
class TWebPS;

class TWebCanvas : public TCanvasImp {

public:
   /// Function type for signals, invoked when canvas drawing or update is completed
   using UpdatedSignal_t = std::function<void()>;

protected:

   /// Function called when pad painting produced
   using PadPaintingReady_t = std::function<void(TPadWebSnapshot *)>;

   struct WebConn {
      unsigned fConnId{0};             ///<! connection id
      Long64_t fSendVersion{0};        ///<! canvas version send to the client
      Long64_t fDrawVersion{0};        ///<! canvas version drawn (confirmed) by client
      std::queue<std::string> fSend;   ///<! send queue, processed after sending draw data
      WebConn(unsigned id) : fConnId(id) {}
   };

   std::vector<WebConn> fWebConn;  ///<! connections

   std::shared_ptr<ROOT::Experimental::RWebWindow> fWindow; ///!< configured display

   bool fHasSpecials{false};       ///<! has special objects which may require pad ranges
   Long64_t fCanvVersion{1};       ///<! actual canvas version, changed with every new Modified() call
   UInt_t fClientBits{0};          ///<! latest status bits from client like editor visible or not
   TList fPrimitivesLists;         ///<! list of lists of primitives, temporary collected during painting
   Int_t fStyleDelivery{0};        ///<! gStyle delivery to clients: 0:never, 1:once, 2:always
   Int_t fPaletteDelivery{1};      ///<! colors palette delivery 0:never, 1:once, 2:always, 3:per subpad
   Int_t fPrimitivesMerge{100};    ///<! number of PS primitives, which will be merged together
   Int_t fJsonComp{0};             ///<! compression factor for messages send to the client
   std::string fCustomScripts;     ///<! custom JavaScript code or URL on JavaScript files to load before start drawing

   UpdatedSignal_t fUpdatedSignal; ///<! signal emitted when canvas updated or state is changed

   void Lock() override {}
   void Unlock() override {}
   Bool_t IsLocked() override { return kFALSE; }

   Bool_t IsWeb() const override { return kTRUE; }
   Bool_t PerformUpdate() override;
   TVirtualPadPainter *CreatePadPainter() override;

   void AddColorsPalette(TPadWebSnapshot &master);
   void CreateObjectSnapshot(TPadWebSnapshot &master, TPad *pad, TObject *obj, const char *opt, TWebPS *masterps = nullptr);
   void CreatePadSnapshot(TPadWebSnapshot &paddata, TPad *pad, Long64_t version, PadPaintingReady_t func);

   Bool_t CheckPadModified(TPad *pad, Bool_t inc_version = kTRUE);

   Bool_t AddToSendQueue(unsigned connid, const std::string &msg);

   void CheckDataToSend(unsigned connid = 0);

   Bool_t WaitWhenCanvasPainted(Long64_t ver);

   virtual Bool_t IsJSSupportedClass(TObject *obj);

   Bool_t IsFirstConn(unsigned connid) const { return (connid!=0) && (fWebConn.size()>0) && (fWebConn[0].fConnId == connid) ;}

   void ShowCmd(const std::string &arg, Bool_t show);

   void AssignStatusBits(UInt_t bits);

   virtual Bool_t ProcessData(unsigned connid, const std::string &arg);

   virtual Bool_t DecodePadOptions(const std::string &) { return kFALSE; }

   virtual Bool_t CanCreateObject(const std::string &) { return !IsReadOnly(); }

public:
   TWebCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TWebCanvas() = default;

   void ShowWebWindow(const ROOT::Experimental::RWebDisplayArgs &user_args = "");

   virtual Bool_t IsReadOnly() const { return kTRUE; }

   Int_t InitWindow() override;
   void Close() override;
   void Show() override;

   UInt_t GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;

   void ShowMenuBar(Bool_t show = kTRUE) override { ShowCmd("Menu", show); }
   void ShowStatusBar(Bool_t show = kTRUE) override { ShowCmd("StatusBar", show); }
   void ShowEditor(Bool_t show = kTRUE) override { ShowCmd("Editor", show); }
   void ShowToolBar(Bool_t show = kTRUE) override { ShowCmd("ToolBar", show); }
   void ShowToolTips(Bool_t show = kTRUE) override { ShowCmd("ToolTips", show); }


   // web-canvas specific methods

   void ActivateInEditor(TPad *pad, TObject *obj);

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
    */

   Bool_t HasEditor() const override;
   Bool_t HasMenuBar() const override;
   Bool_t HasStatusBar() const override;
   Bool_t HasToolBar() const override { return kFALSE; }
   Bool_t HasToolTips() const override;

   void SetUpdatedHandler(UpdatedSignal_t func) { fUpdatedSignal = func; }

   void SetStyleDelivery(Int_t val) { fStyleDelivery = val; }
   Int_t GetStyleDelivery() const { return fStyleDelivery; }

   void SetPaletteDelivery(Int_t val) { fPaletteDelivery = val; }
   Int_t GetPaletteDelivery() const { return fPaletteDelivery; }

   void SetPrimitivesMerge(Int_t cnt) { fPrimitivesMerge = cnt; }
   Int_t GetPrimitivesMerge() const { return fPrimitivesMerge; }

   /** Configures custom script for canvas.
    * If started from "load:" or "assert:" prefix will be loaded with JSROOT.AssertPrerequisites function */
   void SetCustomScripts(const std::string &src) { fCustomScripts = src; }
   std::string GetCustomScripts() const { return fCustomScripts; }

   static TString CreateCanvasJSON(TCanvas *c, Int_t json_compression = 0);
   static Int_t StoreCanvasJSON(TCanvas *c, const char *filename, const char *option = "");

   ClassDefOverride(TWebCanvas, 0) // Web-based implementation for TCanvasImp, read-only mode
};

#endif
