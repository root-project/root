// @(#)root/gpad:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCanvas
#define ROOT_TCanvas

#include "TPad.h"

#include "TAttCanvas.h"

class TCanvasImp;
class TContextMenu;
class TControlBar;

class TCanvas : public TPad {

friend class TCanvasImp;
friend class TThread;
friend class TInterpreter;

protected:
   TAttCanvas    fCatt;            ///< Canvas attributes
   TString       fDISPLAY;         ///< Name of destination screen
   Size_t        fXsizeUser;       ///< User specified size of canvas along X in CM
   Size_t        fYsizeUser;       ///< User specified size of canvas along Y in CM
   Size_t        fXsizeReal;       ///< Current size of canvas along X in CM
   Size_t        fYsizeReal;       ///< Current size of canvas along Y in CM
   Color_t       fHighLightColor;  ///< Highlight color of active pad
   Int_t         fDoubleBuffer;    ///< Double buffer flag (0=off, 1=on)
   Int_t         fWindowTopX;      ///< Top X position of window (in pixels)
   Int_t         fWindowTopY;      ///< Top Y position of window (in pixels)
   UInt_t        fWindowWidth;     ///< Width of window (including borders, etc.)
   UInt_t        fWindowHeight;    ///< Height of window (including menubar, borders, etc.)
   UInt_t        fCw;              ///< Width of the canvas along X (pixels)
   UInt_t        fCh;              ///< Height of the canvas along Y (pixels)
   Int_t         fEvent;           ///<! Type of current or last handled event
   Int_t         fEventX;          ///<! Last X mouse position in canvas
   Int_t         fEventY;          ///<! Last Y mouse position in canvas
   Int_t         fCanvasID;        ///<! Canvas identifier
   TObject      *fSelected;        ///<! Currently selected object
   TObject      *fClickSelected;   ///<! Currently click-selected object
   Int_t         fSelectedX;       ///<! X of selected object
   Int_t         fSelectedY;       ///<! Y of selected object
   TString       fSelectedOpt;     ///<! Drawing option of selected object
   TPad         *fSelectedPad;     ///<! Pad containing currently selected object
   TPad         *fClickSelectedPad;///<! Pad containing currently click-selected object
   TPad         *fPadSave;         ///<! Pointer to saved pad in HandleInput
   TCanvasImp   *fCanvasImp;       ///<! Window system specific canvas implementation
   TContextMenu *fContextMenu;     ///<! Context menu pointer
   Bool_t        fBatch;           ///<! True when in batchmode
   Bool_t        fUpdating;        ///<! True when Updating the canvas
   Bool_t        fRetained;        ///< Retain structure flag
   Bool_t        fUseGL;           ///<! True when rendering is with GL
   Bool_t        fDrawn;           ///<! Set to True when the Draw method is called
   //
   TVirtualPadPainter *fPainter;   ///<! Canvas (pad) painter.

   static Bool_t fgIsFolder;       ///< Indicates if canvas can be browsed as a folder

private:
   TCanvas(const TCanvas &canvas) = delete;
   TCanvas &operator=(const TCanvas &rhs) = delete;
   void     Build();
   void     CopyPixmaps() override;
   void     DrawEventStatus(Int_t event, Int_t x, Int_t y, TObject *selected);
   void     RunAutoExec();

   //Initialize PadPainter.
   void     CreatePainter();

protected:
   void     ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   //-- used by friend TThread class
   void     Init();

public:
   // TCanvas status bits
   enum {
      kShowEventStatus  = BIT(15),
      kAutoExec         = BIT(16),
      kMenuBar          = BIT(17),
      kShowToolBar      = BIT(18),
      kShowEditor       = BIT(19),
      kMoveOpaque       = BIT(20),
      kResizeOpaque     = BIT(21),
      kIsGrayscale      = BIT(22),
      kShowToolTips     = BIT(23)
   };

   TCanvas(Bool_t build=kTRUE);
   TCanvas(const char *name, const char *title="", Int_t form=1);
   TCanvas(const char *name, const char *title, Int_t ww, Int_t wh);
   TCanvas(const char *name, const char *title, Int_t wtopx, Int_t wtopy,
           Int_t ww, Int_t wh);
   TCanvas(const char *name, Int_t ww, Int_t wh, Int_t winid);
   virtual ~TCanvas();

   //-- used by friend TThread class
   void Constructor();
   void Constructor(const char *name, const char *title, Int_t form);
   void Constructor(const char *name, const char *title, Int_t ww, Int_t wh);
   void Constructor(const char *name, const char *title,
           Int_t wtopx, Int_t wtopy, Int_t ww, Int_t wh);
   void Destructor();

   TVirtualPad      *cd(Int_t subpadnumber=0) override;
   void              Browse(TBrowser *b) override;
   void              Clear(Option_t *option="") override;
   void              Close(Option_t *option="") override;
   void              Delete(Option_t * = "") override { MayNotUse("Delete()"); }
   void              DisconnectWidget();  // used by TCanvasImp
   void              Draw(Option_t *option="") override;
   TObject          *DrawClone(Option_t *option="") const override; // *MENU*
   virtual TObject  *DrawClonePad(); // *MENU*
   virtual void      EditorBar();
   void              EmbedInto(Int_t winid, Int_t ww, Int_t wh);
   void              EnterLeave(TPad *prevSelPad, TObject *prevSelObj);
   void              FeedbackMode(Bool_t set);
   void              Flush();
   void              UseCurrentStyle() override; // *MENU*
   void              ForceUpdate();
   const char       *GetDISPLAY() const {return fDISPLAY.Data();}
   TContextMenu     *GetContextMenu() const {return fContextMenu;};
   Int_t             GetDoubleBuffer() const {return fDoubleBuffer;}
   Int_t             GetEvent() const override { return fEvent; }
   Int_t             GetEventX() const override { return fEventX; }
   Int_t             GetEventY() const override { return fEventY; }
   Color_t           GetHighLightColor() const override { return fHighLightColor; }
   TVirtualPad      *GetPadSave() const override { return fPadSave; }
   void              ClearPadSave() { fPadSave = nullptr; }
   TObject          *GetSelected() const override { return fSelected; }
   TObject          *GetClickSelected() const { return fClickSelected; }
   Int_t             GetSelectedX() const { return fSelectedX; }
   Int_t             GetSelectedY() const { return fSelectedY; }
   Option_t         *GetSelectedOpt() const { return fSelectedOpt.Data(); }
   TVirtualPad      *GetSelectedPad() const override { return fSelectedPad; }
   TVirtualPad      *GetClickSelectedPad() const { return fClickSelectedPad; }
   Bool_t            GetShowEventStatus() const { return TestBit(kShowEventStatus); }
   Bool_t            GetShowToolBar() const { return TestBit(kShowToolBar); }
   Bool_t            GetShowEditor() const { return TestBit(kShowEditor); }
   Bool_t            GetShowToolTips() const { return TestBit(kShowToolTips); }
   Bool_t            GetAutoExec() const { return TestBit(kAutoExec); }
   Size_t            GetXsizeUser() const { return fXsizeUser; }
   Size_t            GetYsizeUser() const { return fYsizeUser; }
   Size_t            GetXsizeReal() const { return fXsizeReal; }
   Size_t            GetYsizeReal() const { return fYsizeReal; }
   Int_t             GetCanvasID() const override { return fCanvasID; }
   TCanvasImp       *GetCanvasImp() const override { return fCanvasImp; }
   Int_t             GetWindowTopX();
   Int_t             GetWindowTopY();
   UInt_t            GetWindowWidth() const { return fWindowWidth; }
   UInt_t            GetWindowHeight() const { return fWindowHeight; }
   UInt_t            GetWw() const override { return fCw; }
   UInt_t            GetWh() const override { return fCh; }
   virtual void      GetCanvasPar(Int_t &wtopx, Int_t &wtopy, UInt_t &ww, UInt_t &wh)
                     {wtopx=GetWindowTopX(); wtopy=fWindowTopY; ww=fWindowWidth; wh=fWindowHeight;}
   virtual void      HandleInput(EEventType button, Int_t x, Int_t y);
   Bool_t            HasMenuBar() const { return TestBit(kMenuBar); }
   virtual void      HighlightConnect(const char *slot);
   void              Iconify();
   Bool_t            IsBatch() const override { return fBatch; }
   Bool_t            IsDrawn() { return fDrawn; }
   Bool_t            IsFolder() const override;
   Bool_t            IsGrayscale();
   Bool_t            IsRetained() const override { return fRetained; }
   Bool_t            IsWeb() const;
   void              ls(Option_t *option="") const override;
   void              MoveOpaque(Int_t set=1);
   Bool_t            OpaqueMoving() const override { return TestBit(kMoveOpaque); }
   Bool_t            OpaqueResizing() const override { return TestBit(kResizeOpaque); }
   void              Paint(Option_t *option="") override;
   TPad             *Pick(Int_t px, Int_t py, TObjLink *&pickobj) override { return TPad::Pick(px, py, pickobj); }
   virtual TPad     *Pick(Int_t px, Int_t py, TObject *prevSelObj);
   virtual void      Picked(TPad *selpad, TObject *selected, Int_t event);             // *SIGNAL*
   virtual void      Highlighted(TVirtualPad *pad, TObject *obj, Int_t x, Int_t y);    // *SIGNAL*
   virtual void      ProcessedEvent(Int_t event, Int_t x, Int_t y, TObject *selected); // *SIGNAL*
   virtual void      Selected(TVirtualPad *pad, TObject *obj, Int_t event);            // *SIGNAL*
   virtual void      Cleared(TVirtualPad *pad);                                        // *SIGNAL*
   void              Closed() override;                                                // *SIGNAL*
   void              RaiseWindow();
   void              ResetDrawn() { fDrawn=kFALSE; }
   virtual void      Resize(Option_t *option="");
   void              ResizeOpaque(Int_t set=1);
   void              SaveSource(const char *filename="", Option_t *option="");
   void              SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void              SetCursor(ECursor cursor) override;
   void              SetDoubleBuffer(Int_t mode=1) override;
   void              SetName(const char *name="") override;
   void              SetFixedAspectRatio(Bool_t fixed = kTRUE) override;  // *TOGGLE*
   void              SetGrayscale(Bool_t set = kTRUE); // *TOGGLE* *GETTER=IsGrayscale
   void              SetWindowPosition(Int_t x, Int_t y);
   void              SetWindowSize(UInt_t ww, UInt_t wh);
   void              SetCanvasImp(TCanvasImp *i) { fCanvasImp = i; }
   void              SetCanvasSize(UInt_t ww, UInt_t wh) override; // *MENU*
   void              SetHighLightColor(Color_t col) { fHighLightColor = col; }
   void              SetSelected(TObject *obj) override;
   void              SetClickSelected(TObject *obj) { fClickSelected = obj; }
   void              SetSelectedPad(TPad *pad) { fSelectedPad = pad; }
   void              SetClickSelectedPad(TPad *pad) { fClickSelectedPad = pad; }
   void              Show();
   virtual void      Size(Float_t xsizeuser=0, Float_t ysizeuser=0);
   void              SetBatch(Bool_t batch=kTRUE) override;
   static  void      SetFolder(Bool_t isfolder=kTRUE);
   void              SetPadSave(TPad *pad) {fPadSave = pad;}
   bool              SetRealAspectRatio(const Int_t axis = 1); // *MENU*
   void              SetRetained(Bool_t retained=kTRUE) { fRetained=retained;}
   void              SetTitle(const char *title="") override;
   virtual void      ToggleEventStatus();
   virtual void      ToggleAutoExec();
   virtual void      ToggleToolBar();
   virtual void      ToggleEditor();
   virtual void      ToggleToolTips();
   void              Update() override;

   Bool_t              UseGL() const { return fUseGL; }
   void                SetSupportGL(Bool_t support) {fUseGL = support;}
   TVirtualPadPainter *GetCanvasPainter();
   void                DeleteCanvasPainter();

   static TCanvas   *MakeDefCanvas();
   static Bool_t     SupportAlpha();

   ClassDefOverride(TCanvas,8)  //Graphics canvas
};

#endif
