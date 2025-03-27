// Author: Sergey Linev, GSI   7/12/2016

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebCanvas
#define ROOT_TWebCanvas

#include "TCanvasImp.h"

#include "TString.h"
#include "TList.h"
#include "TWebPadOptions.h"

#include <ROOT/RWebWindow.hxx>

#include <vector>
#include <string>
#include <queue>
#include <functional>
#include <map>

class TPad;
class TPadWebSnapshot;
class TWebPS;
class TObjLink;
class TExec;
class TWebCanvasTimer;

class TWebCanvas : public TCanvasImp {

friend class TWebCanvasTimer;

public:
   /// Function type for signals, invoked when canvas drawing or update is completed
   using UpdatedSignal_t = std::function<void()>;

   /// Function type for pad-related signals - like activate pad signal
   using PadSignal_t = std::function<void(TPad *)>;

   /// Function type for pad-click signals
   using PadClickedSignal_t = std::function<void(TPad *, int, int)>;

   /// Function type for signals, invoked when object is selected
   using ObjectSelectSignal_t = std::function<void(TPad *, TObject *)>;

protected:

   /// Function called when pad painting produced
   using PadPaintingReady_t = std::function<void(TPadWebSnapshot *)>;

   struct WebConn {
      unsigned fConnId{0};             ///<! connection id
      Long64_t fCheckedVersion{0};     ///<! canvas version checked before sending
      Long64_t fSendVersion{0};        ///<! canvas version send to the client
      Long64_t fDrawVersion{0};        ///<! canvas version drawn (confirmed) by client
      UInt_t fLastSendHash{0};         ///<! hash of last send draw message, avoid looping
      std::map<std::string, std::string> fCtrl; ///<! different ctrl parameters which can be send at once
      std::queue<std::string> fSend;   ///<! send queue, processed after sending draw data

      WebConn(unsigned id) : fConnId(id) {}
      bool is_batch() const { return fConnId == 0; }
      bool match(unsigned id) const { return !is_batch() && ((fConnId == id) || (id == 0)); }
      void reset()
      {
         fCheckedVersion = fSendVersion = fDrawVersion = 0;
         fLastSendHash = 0;
      }
   };

   struct PadStatus {
      Long64_t fVersion{0};    ///<! last pad version
      bool _detected{false};   ///<! if pad was detected during last scan
      bool _modified{false};   ///<! if pad was modified during last scan
      bool _has_specials{false}; ///<! are there any special objects with painting
   };

   std::vector<WebConn> fWebConn;  ///<! connections
   TWebCanvasTimer *fTimer{nullptr}; ///<! timer to submit control messages

   std::map<TPad*, PadStatus> fPadsStatus; ///<! map of pads in canvas and their status flags

   std::shared_ptr<ROOT::RWebWindow> fWindow; ///!< configured display

   Bool_t fReadOnly{kFALSE};       ///<! in read-only mode canvas cannot be changed from client side
   Long64_t fCanvVersion{1};       ///<! actual canvas version, changed with every new Modified() call
   Long64_t fLastDrawVersion{0};   ///<! last draw version
   UInt_t fClientBits{0};          ///<! latest status bits from client like editor visible or not
   std::vector<TPad *> fAllPads;   ///<! list of all pads recognized during streaming
   std::map<TObject *,bool> fUsedObjs; ///<! map of used objects during streaming
   Int_t fStyleDelivery{0};        ///<! gStyle delivery to clients: 0:never, 1:once, 2:always
   Int_t fPaletteDelivery{1};      ///<! colors palette delivery 0:never, 1:once, 2:always, 3:per subpad
   Int_t fPrimitivesMerge{100};    ///<! number of PS primitives, which will be merged together
   Int_t fJsonComp{0};             ///<! compression factor for messages send to the client
   Bool_t fCanCreateObjects{kTRUE}; ///<! indicates if canvas allowed to create extra objects for interactive painting
   Bool_t fLongerPolling{kFALSE};  ///<! when true, make longer polling in blocking operations
   Bool_t fProcessingData{kFALSE}; ///<! flag used to prevent blocking methods when process data is invoked
   Bool_t fAsyncMode{kFALSE};      ///<! when true, methods like TCanvas::Update will never block
   Long64_t fStyleVersion{0};      ///<! current gStyle object version, checked every time when new snapshot created
   UInt_t fStyleHash{0};           ///<! last hash of gStyle
   Long64_t fColorsVersion{0};     ///<! current colors/palette version, checked every time when new snapshot created
   UInt_t fColorsHash{0};          ///<! last hash of colors/palette
   Int_t fTF1UseSave{1};           ///<! use save buffer for TF1/TF2, 0:off, 1:prefer, 2:force
   std::vector<int> fWindowGeometry; ///<! last received window geometry
   Bool_t fFixedSize{kFALSE};      ///<! is canvas size fixed

   UpdatedSignal_t fUpdatedSignal; ///<! signal emitted when canvas updated or state is changed
   PadSignal_t fActivePadChangedSignal; ///<! signal emitted when active pad changed in the canvas
   PadClickedSignal_t fPadClickedSignal; ///<! signal emitted when simple mouse click performed on the pad
   PadClickedSignal_t fPadDblClickedSignal; ///<! signal emitted when simple mouse click performed on the pad
   ObjectSelectSignal_t fObjSelectSignal; ///<! signal emitted when new object selected in the pad

   std::vector<std::unique_ptr<ROOT::RWebDisplayHandle>> fHelpHandles; ///<! array of handles for help widgets

   static std::string gCustomScripts;     ///<! custom JavaScript code or URL on JavaScript files to load before start drawing
   static std::vector<std::string> gCustomClasses;  ///<! list of custom classes, which can be delivered as is to client

   static UInt_t gBatchImageMode;           ///<! configured batch size
   static std::string gBatchMultiPdf;           ///<! name of current multi-page pdf file
   static std::vector<std::string> gBatchFiles; ///<! file names for batch job
   static std::vector<std::string> gBatchJsons; ///<! converted jsons batch job
   static std::vector<int> gBatchWidths;   ///<! batch job widths
   static std::vector<int> gBatchHeights;  ///<! batch job heights

   void Lock() override {}
   void Unlock() override {}
   Bool_t IsLocked() override { return kFALSE; }

   Bool_t IsWeb() const override { return kTRUE; }
   Bool_t PerformUpdate(Bool_t async) override;
   TVirtualPadPainter *CreatePadPainter() override;

   UInt_t CalculateColorsHash();
   void AddColorsPalette(TPadWebSnapshot &master);
   void AddCustomFonts(TPadWebSnapshot &master);

   void CreateObjectSnapshot(TPadWebSnapshot &master, TPad *pad, TObject *obj, const char *opt, TWebPS *masterps = nullptr);
   void CreatePadSnapshot(TPadWebSnapshot &paddata, TPad *pad, Long64_t version, PadPaintingReady_t func);

   void CheckPadModified(TPad *pad);

   Bool_t CheckCanvasModified(bool force_modified = false);

   void AddCtrlMsg(unsigned connid, const std::string &key, const std::string &value);

   void AddSendQueue(unsigned connid, const std::string &msg);

   Bool_t CheckDataToSend(unsigned connid = 0);

   Bool_t WaitWhenCanvasPainted(Long64_t ver);

   virtual Bool_t IsJSSupportedClass(TObject *obj, Bool_t many_primitives = kFALSE);

   Bool_t IsFirstConn(unsigned connid) const { return (connid != 0) && (fWebConn.size() > 1) && (fWebConn[1].fConnId == connid); }

   Bool_t IsFirstDrawn() const { return (fWebConn.size() > 1) && (fWebConn[1].fDrawVersion > 0); }

   void ShowCmd(const std::string &arg, Bool_t show);

   void AssignStatusBits(UInt_t bits);

   virtual Bool_t ProcessData(unsigned connid, const std::string &arg);

   virtual Bool_t DecodePadOptions(const std::string &, bool process_execs = false);

   virtual Bool_t CanCreateObject(const std::string &) { return !IsReadOnly() && fCanCreateObjects; }

   TPad *ProcessObjectOptions(TWebObjectOptions &item, TPad *pad, int idcnt = 1);

   TObject *FindPrimitive(const std::string &id, int idcnt = 1, TPad *pad = nullptr, TObjLink **objlnk = nullptr, TPad **objpad = nullptr);

   void ProcessExecs(TPad *pad, TExec *extra = nullptr);

   void ProcessLinesForObject(TObject *obj, const std::string &lines);

   void SetWindowGeometry(const std::vector<int> &arr);

   static std::string ProcessCustomScripts(bool batch);

   static bool FlushBatchImages();

public:
   TWebCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height, Bool_t readonly = kTRUE);
   ~TWebCanvas() override;

   void CreateWebWindow();

   void ShowWebWindow(const ROOT::RWebDisplayArgs &user_args = "");

   const std::shared_ptr<ROOT::RWebWindow> &GetWebWindow() const { return fWindow; }

   virtual Bool_t IsReadOnly() const { return fReadOnly; }

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

   void ForceUpdate() override;

   void   SetWindowPosition(Int_t x, Int_t y) override;
   void   SetWindowSize(UInt_t w, UInt_t h) override;
   void   SetWindowTitle(const char *newTitle) override;
   void   SetCanvasSize(UInt_t w, UInt_t h) override;
   void   Iconify() override;
   void   RaiseWindow() override;

   /*
      virtual void   SetStatusText(const char *text = 0, Int_t partidx = 0);
      virtual void   ReallyDelete();
    */

   Bool_t HasEditor() const override;
   Bool_t HasMenuBar() const override;
   Bool_t HasStatusBar() const override;
   Bool_t HasToolBar() const override { return kFALSE; }
   Bool_t HasToolTips() const override;

   void SetUpdatedHandler(UpdatedSignal_t func) { fUpdatedSignal = func; }
   void SetActivePadChangedHandler(PadSignal_t func) { fActivePadChangedSignal = func; }
   void SetPadClickedHandler(PadClickedSignal_t func) { fPadClickedSignal = func; }
   void SetPadDblClickedHandler(PadClickedSignal_t func) { fPadDblClickedSignal = func; }
   void SetObjSelectHandler(ObjectSelectSignal_t func) { fObjSelectSignal = func; }

   void SetCanCreateObjects(Bool_t on = kTRUE) { fCanCreateObjects = on; }
   Bool_t GetCanCreateObjects() const { return fCanCreateObjects; }

   void SetStyleDelivery(Int_t val) { fStyleDelivery = val; }
   Int_t GetStyleDelivery() const { return fStyleDelivery; }

   void SetPaletteDelivery(Int_t val) { fPaletteDelivery = val; }
   Int_t GetPaletteDelivery() const { return fPaletteDelivery; }

   void SetPrimitivesMerge(Int_t cnt) { fPrimitivesMerge = cnt; }
   Int_t GetPrimitivesMerge() const { return fPrimitivesMerge; }

   void SetLongerPolling(Bool_t on) { fLongerPolling = on; }
   Bool_t GetLongerPolling() const { return fLongerPolling; }

   void SetAsyncMode(Bool_t on = kTRUE) { fAsyncMode = on; }
   Bool_t IsAsyncMode() const { return fAsyncMode; }

   Bool_t IsFixedSize() const { return fFixedSize; }

   static void SetCustomScripts(const std::string &src);
   static const std::string &GetCustomScripts();

   static void AddCustomClass(const std::string &clname, bool with_derived = false);
   static bool IsCustomClass(const TClass *cl);

   static Font_t AddFont(const char *name, const char *ttffile, Int_t precision = 2);

   static TString CreatePadJSON(TPad *pad, Int_t json_compression = 0, Bool_t batchmode = kFALSE);
   static TString CreateCanvasJSON(TCanvas *c, Int_t json_compression = 0, Bool_t batchmode = kFALSE);
   static Int_t StoreCanvasJSON(TCanvas *c, const char *filename, const char *option = "");

   static bool ProduceImage(TPad *pad, const char *filename, Int_t width = 0, Int_t height = 0);

   static bool ProduceImages(std::vector<TPad *> pads, const char *filename, Int_t width = 0, Int_t height = 0);

   static void BatchImageMode(UInt_t n = 100);

   static TCanvasImp *NewCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);

   static TCanvas *CreateWebCanvas(const char *name, const char *title, UInt_t width = 1200, UInt_t height = 800);

   ClassDefOverride(TWebCanvas, 0) // Web-based implementation for TCanvasImp
};

#endif
