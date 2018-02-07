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

#include <list>
#include <vector>
#include <string>
#include <functional>

#include <ROOT/TWebWindow.hxx>

class TVirtualPad;
class TPad;
class TList;
class TObjLink;
class TWebSnapshot;
class TPadWebSnapshot;
class THttpServer;
class TWebCanvas;

/// Class used to transport drawing options from the client
class TWebObjectOptions {
public:
   std::string snapid; ///< id of the object
   std::string opt;    ///< drawing options
   TWebObjectOptions() = default;
};

/// Class used to transport ranges from JSROOT canvas
class TWebPadRange {
public:
   std::string snapid;                        ///< id of pad
   bool active{false};                        ///< if pad selected as active
   int logx{0}, logy{0}, logz{0};             ///< pad log properties
   int gridx{0}, gridy{0};                    ///< pad grid properties
   int tickx{0}, ticky{0};                    ///< pad ticks properties
   Float_t mleft{0}, mright{0}, mtop{0}, mbottom{0}; ///< frame margins
   bool ranges{false};                        ///< if true, pad has ranges
   Double_t px1{0}, py1{0}, px2{0}, py2{0};   ///< pad range
   Double_t ux1{0}, uy1{0}, ux2{0}, uy2{0};   ///< pad axis range
   unsigned bits{0};                          ///< canvas status bits like tool editor
   std::vector<TWebObjectOptions> primitives; ///< drawing options for primitives
   TWebPadRange() = default;
};


/////////////////////////////////////////////////////////

/// Class used to transport pad click events
class TWebPadClick {
public:
   std::string padid;                         ///< id of pad
   std::string objid;                         ///< id of clicked object, "null" when not defined
   int x{-1};                                 ///< x coordinate of click event
   int y{-1};                                 ///< y coordinate of click event
   bool dbl{false};                           ///< when double-click was performed
   TWebPadClick() = default;
};

/////////////////////////////////////////////////////////

/// Function type for signals, connected when canvas drawing or update is completed
using TWebCanvasUpdatedSignal_t = std::function<void()>;

/// Function type called for signals, connected with pad like select pad
using TWebCanvasPadSignal_t = std::function<void(TPad *)>;

/// Function type called for signals, connected with pad like select pad
using TWebCanvasObjSelectSignal_t = std::function<void(TPad *, TObject *)>;

/// Function type for pad-click signals
using TWebCanvasPadClickedSignal_t = std::function<void(TPad *, int, int)>;

class TWebCanvas : public TCanvasImp {

protected:
   struct WebConn {
      unsigned fConnId;      ///<! websocket handle
      TString fGetMenu;      ///<! object id for menu request
      Long64_t fDrawVersion; ///<! canvas version drawn by client
      TString fSend;         ///<! extra data which should be send to the client
      WebConn() : fConnId(0), fGetMenu(), fDrawVersion(0), fSend() {}
   };

   typedef std::list<WebConn> WebConnList;

   WebConnList fWebConn; ///<! connections list

   std::shared_ptr<ROOT::Experimental::TWebWindow> fWindow; ///!< configured display

   Bool_t fHasSpecials;   ///<! has special objects which may require pad ranges
   Long64_t fCanvVersion; ///<! actual canvas version, changed with every new Modified() call
   Bool_t fWaitNewConnection; ///<! when true, Update() will wait for a new connection
   UInt_t fClientBits; ///<! latest status bits from client like editor visisble or not

   TWebCanvasUpdatedSignal_t  fUpdatedSignal; ///<! signal emitted when canvas updated or state is changed

   TWebCanvasPadSignal_t fActivePadChangedSignal; ///<!  signal emitted when active pad changed in the canvas

   TWebCanvasObjSelectSignal_t fObjSelectSignal; ///<! signal emitted when new object selected in the pad

   TWebCanvasPadClickedSignal_t fPadClickedSignal; ///<! signal emitted when simple mouse click performed on the pad

   TWebCanvasPadClickedSignal_t fPadDblClickedSignal; ///<! signal emitted when simple mouse click performed on the pad

   virtual void Lock() {}
   virtual void Unlock() {}
   virtual Bool_t IsLocked() { return kFALSE; }

   virtual Bool_t PerformUpdate();
   virtual TVirtualPadPainter *CreatePadPainter();

   Bool_t AddCanvasSpecials(TPadWebSnapshot *master);
   TString CreateSnapshot(TPad *pad, TPadWebSnapshot *master = 0, TList *tempbuf = 0);
   TWebSnapshot *CreateObjectSnapshot(TObject *obj, const char *opt);

   TObject *FindPrimitive(const char *id, TPad *pad = nullptr, TObjLink **padlnk = nullptr);
   Bool_t DecodeAllRanges(const char *arg);

   Bool_t IsAnyPadModified(TPad *pad);

   void CheckDataToSend();

   Bool_t WaitWhenCanvasPainted(Long64_t ver);

   Bool_t IsJSSupportedClass(TObject *obj);

   void ShowCmd(const char *arg, Bool_t show);

   void AssignStatusBits(UInt_t bits);

public:
   TWebCanvas();
   TWebCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TWebCanvas();

   TString CreateWebWindow(int limit = 0);
   THttpServer *GetServer();

   virtual Int_t InitWindow();
   virtual void Close();
   virtual void Show();

   void ProcessData(unsigned connid, const std::string &arg);

   virtual UInt_t GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);

   virtual void ShowMenuBar(Bool_t show = kTRUE) { ShowCmd("Menu", show); }
   virtual void ShowStatusBar(Bool_t show = kTRUE) { ShowCmd("StatusBar", show); }
   virtual void ShowEditor(Bool_t show = kTRUE) { ShowCmd("Editor", show); }
   virtual void ShowToolBar(Bool_t show = kTRUE) { ShowCmd("ToolBar", show); }
   virtual void ShowToolTips(Bool_t show = kTRUE) { ShowCmd("ToolTips", show); }


   // web-canvas specific methods

   void SetUpdatedHandler(TWebCanvasUpdatedSignal_t func) { fUpdatedSignal = func; }

   void SetActivePadChangedHandler(TWebCanvasPadSignal_t func) { fActivePadChangedSignal = func; }

   void SetObjSelectHandler(TWebCanvasObjSelectSignal_t func) { fObjSelectSignal = func; }

   void SetPadClickedHandler(TWebCanvasPadClickedSignal_t func) { fPadClickedSignal = func; }

   void SetPadDblClickedHandler(TWebCanvasPadClickedSignal_t func) { fPadDblClickedSignal = func; }

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

   virtual Bool_t HasEditor() const;
   virtual Bool_t HasMenuBar() const;
   virtual Bool_t HasStatusBar() const;
   virtual Bool_t HasToolBar() const { return kFALSE; }
   virtual Bool_t HasToolTips() const;

   ClassDef(TWebCanvas, 0) // ABC describing main window protocol
};

#endif
