// Author: Sergey Linev, GSI   7/12/2016

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
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

#include "TCanvasImp.h"

#include "TString.h"

#include <ROOT/RWebWindow.hxx>

#include <vector>
#include <string>
#include <functional>

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
};

/////////////////////////////////////////////////////////

class TWebCanvas : public TCanvasImp {

public:
   /// Function type for signals, connected when canvas drawing or update is completed
   using UpdatedSignal_t = std::function<void()>;

   /// Function type called for signals, connected with pad like select pad
   using PadSignal_t = std::function<void(TPad *)>;

   /// Function type called for signals, connected with pad like select pad
   using ObjectSelectSignal_t = std::function<void(TPad *, TObject *)>;

   /// Function type for pad-click signals
   using PadClickedSignal_t = std::function<void(TPad *, int, int)>;

protected:
   struct WebConn {
      unsigned fConnId{0};       ///<! connection id
      std::string fGetMenu;      ///<! object id for menu request
      Long64_t fDrawVersion{0};  ///<! canvas version drawn by client
      std::string fSend;         ///<! extra data which should be send to the client
      WebConn(unsigned id) : fConnId(id) {}
   };

   std::vector<WebConn> fWebConn; ///<! connections

   std::shared_ptr<ROOT::Experimental::RWebWindow> fWindow; ///!< configured display

   bool fHasSpecials{false};       ///<! has special objects which may require pad ranges
   Long64_t fCanvVersion{1};       ///<! actual canvas version, changed with every new Modified() call
   bool fWaitNewConnection{false}; ///<! when true, Update() will wait for a new connection
   UInt_t fClientBits{0};          ///<! latest status bits from client like editor visible or not

   UpdatedSignal_t fUpdatedSignal;          ///<! signal emitted when canvas updated or state is changed
   PadSignal_t fActivePadChangedSignal;     ///<!  signal emitted when active pad changed in the canvas
   ObjectSelectSignal_t fObjSelectSignal;   ///<! signal emitted when new object selected in the pad
   PadClickedSignal_t fPadClickedSignal;    ///<! signal emitted when simple mouse click performed on the pad
   PadClickedSignal_t fPadDblClickedSignal; ///<! signal emitted when simple mouse click performed on the pad

   virtual void Lock() {}
   virtual void Unlock() {}
   virtual Bool_t IsLocked() { return kFALSE; }

   virtual Bool_t IsWeb() const { return kTRUE; }
   virtual Bool_t PerformUpdate();
   virtual TVirtualPadPainter *CreatePadPainter();

   Bool_t AddCanvasSpecials(TPadWebSnapshot *master);
   TString CreateSnapshot(TPad *pad, TPadWebSnapshot *master = nullptr, TList *tempbuf = nullptr);
   TWebSnapshot *CreateObjectSnapshot(TPad *pad, TObject *obj, const char *opt);

   TObject *FindPrimitive(const char *id, TPad *pad = nullptr, TObjLink **padlnk = nullptr);
   Bool_t DecodeAllRanges(const char *arg);

   Bool_t IsAnyPadModified(TPad *pad);

   void CheckDataToSend();

   Bool_t WaitWhenCanvasPainted(Long64_t ver);

   Bool_t IsJSSupportedClass(TObject *obj);

   void ShowCmd(const char *arg, Bool_t show);

   void AssignStatusBits(UInt_t bits);

public:
   TWebCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TWebCanvas() = default;

   TString CreateWebWindow(int limit = 0);
   THttpServer *GetServer();
   void ShowWebWindow(const ROOT::Experimental::RWebDisplayArgs &user_args = "");

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

   void SetUpdatedHandler(UpdatedSignal_t func) { fUpdatedSignal = func; }

   void SetActivePadChangedHandler(PadSignal_t func) { fActivePadChangedSignal = func; }

   void SetObjSelectHandler(ObjectSelectSignal_t func) { fObjSelectSignal = func; }

   void SetPadClickedHandler(PadClickedSignal_t func) { fPadClickedSignal = func; }

   void SetPadDblClickedHandler(PadClickedSignal_t func) { fPadDblClickedSignal = func; }

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
