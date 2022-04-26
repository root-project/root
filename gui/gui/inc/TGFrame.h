// @(#)root/gui:$Id$
// Author: Fons Rademakers   03/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGFrame
#define ROOT_TGFrame


#include "TGWindow.h"
#include "TQObject.h"
#include "TGDimension.h"
#include "TGGC.h"
#include "TGFont.h"
#include "TGLayout.h"
#include "TGString.h"

class TGResourcePool;
class TGTextButton;
class TGVFileSplitter;
class TDNDData;
class TList;

//---- frame states

enum EFrameState {
   kIsVisible  = BIT(0),
   kIsMapped   = kIsVisible,
   kIsArranged = BIT(1)
};

//---- frame cleanup
enum EFrameCleanup {
   kNoCleanup    = 0,
   kLocalCleanup = 1,
   kDeepCleanup  = -1
};

//---- MWM hints stuff

enum EMWMHints {
   // functions
   kMWMFuncAll      = BIT(0),
   kMWMFuncResize   = BIT(1),
   kMWMFuncMove     = BIT(2),
   kMWMFuncMinimize = BIT(3),
   kMWMFuncMaximize = BIT(4),
   kMWMFuncClose    = BIT(5),

   // input mode
   kMWMInputModeless                = 0,
   kMWMInputPrimaryApplicationModal = 1,
   kMWMInputSystemModal             = 2,
   kMWMInputFullApplicationModal    = 3,

   // decorations
   kMWMDecorAll      = BIT(0),
   kMWMDecorBorder   = BIT(1),
   kMWMDecorResizeH  = BIT(2),
   kMWMDecorTitle    = BIT(3),
   kMWMDecorMenu     = BIT(4),
   kMWMDecorMinimize = BIT(5),
   kMWMDecorMaximize = BIT(6)
};

//---- drag and drop

enum EDNDFlags {
   kIsDNDSource = BIT(0),
   kIsDNDTarget = BIT(1)
};


class TGFrame : public TGWindow, public TQObject {

protected:
   enum { kDeleteWindowCalled = BIT(15) };

   Int_t    fX;             ///< frame x position
   Int_t    fY;             ///< frame y position
   UInt_t   fWidth;         ///< frame width
   UInt_t   fHeight;        ///< frame height
   UInt_t   fMinWidth;      ///< minimal frame width
   UInt_t   fMinHeight;     ///< minimal frame height
   UInt_t   fMaxWidth;      ///< maximal frame width
   UInt_t   fMaxHeight;     ///< maximal frame height
   Int_t    fBorderWidth;   ///< frame border width
   UInt_t   fOptions;       ///< frame options
   Pixel_t  fBackground;    ///< frame background color
   UInt_t   fEventMask;     ///< currently active event mask
   Int_t    fDNDState;      ///< EDNDFlags
   TGFrameElement *fFE;     ///< pointer to frame element

   static Bool_t      fgInit;
   static Pixel_t     fgDefaultFrameBackground;
   static Pixel_t     fgDefaultSelectedBackground;
   static Pixel_t     fgWhitePixel;
   static Pixel_t     fgBlackPixel;
   static const TGGC *fgBlackGC;
   static const TGGC *fgWhiteGC;
   static const TGGC *fgHilightGC;
   static const TGGC *fgShadowGC;
   static const TGGC *fgBckgndGC;
   static Time_t      fgLastClick;
   static UInt_t      fgLastButton;
   static Int_t       fgDbx, fgDby;
   static Window_t    fgDbw;
   static UInt_t      fgUserColor;

   static Time_t      GetLastClick();

   void  *GetSender() override { return this; }  //used to set gTQSender
   virtual void   Draw3dRectangle(UInt_t type, Int_t x, Int_t y,
                                  UInt_t w, UInt_t h);
   void   DoRedraw() override;

   const TGResourcePool *GetResourcePool() const
      { return fClient->GetResourcePool(); }

   TString GetOptionString() const;                //used in SavePrimitive()

   // some protected methods use in gui builder
   virtual void StartGuiBuilding(Bool_t on = kTRUE);

private:
   TGFrame(const TGFrame&) = delete;
   TGFrame& operator=(const TGFrame&) = delete;

public:
   // Default colors and graphics contexts
   static Pixel_t     GetDefaultFrameBackground();
   static Pixel_t     GetDefaultSelectedBackground();
   static Pixel_t     GetWhitePixel();
   static Pixel_t     GetBlackPixel();
   static const TGGC &GetBlackGC();
   static const TGGC &GetWhiteGC();
   static const TGGC &GetHilightGC();
   static const TGGC &GetShadowGC();
   static const TGGC &GetBckgndGC();

   TGFrame(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
           UInt_t options = 0, Pixel_t back = GetDefaultFrameBackground());
   TGFrame(TGClient *c, Window_t id, const TGWindow *parent = nullptr);
   virtual ~TGFrame();

   virtual void DeleteWindow();
   virtual void ReallyDelete() { delete this; }

   UInt_t GetEventMask() const { return fEventMask; }
   void   AddInput(UInt_t emask);
   void   RemoveInput(UInt_t emask);

           Bool_t HandleEvent(Event_t *event) override;
   virtual Bool_t HandleConfigureNotify(Event_t *event);
   virtual Bool_t HandleButton(Event_t *) { return kFALSE; }
   virtual Bool_t HandleDoubleClick(Event_t *) { return kFALSE; }
   virtual Bool_t HandleCrossing(Event_t *) { return kFALSE; }
   virtual Bool_t HandleMotion(Event_t *) { return kFALSE; }
   virtual Bool_t HandleKey(Event_t *) { return kFALSE; }
   virtual Bool_t HandleFocusChange(Event_t *) { return kFALSE; }
   virtual Bool_t HandleClientMessage(Event_t *event);
   virtual Bool_t HandleSelection(Event_t *) { return kFALSE; }
   virtual Bool_t HandleSelectionRequest(Event_t *) { return kFALSE; }
   virtual Bool_t HandleSelectionClear(Event_t *) { return kFALSE; }
   virtual Bool_t HandleColormapChange(Event_t *) { return kFALSE; }
   virtual Bool_t HandleDragEnter(TGFrame *) { return kFALSE; }
   virtual Bool_t HandleDragLeave(TGFrame *) { return kFALSE; }
   virtual Bool_t HandleDragMotion(TGFrame *) { return kFALSE; }
   virtual Bool_t HandleDragDrop(TGFrame *, Int_t /*x*/, Int_t /*y*/, TGLayoutHints*)
                     { return kFALSE; }
   virtual void   ProcessedConfigure(Event_t *event)
                     { Emit("ProcessedConfigure(Event_t*)", (Longptr_t)event); } //*SIGNAL*
   virtual void   ProcessedEvent(Event_t *event)
                     { Emit("ProcessedEvent(Event_t*)", (Longptr_t)event); } //*SIGNAL*

   virtual void   SendMessage(const TGWindow *w, Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
   virtual Bool_t ProcessMessage(Longptr_t, Longptr_t, Longptr_t) { return kFALSE; }

   virtual TGDimension GetDefaultSize() const ;
           void    Move(Int_t x, Int_t y) override;
           void    Resize(UInt_t w = 0, UInt_t h = 0) override;
   virtual void    Resize(TGDimension size);
           void    MoveResize(Int_t x, Int_t y, UInt_t w = 0, UInt_t h = 0) override;
   virtual UInt_t  GetDefaultWidth() const { return GetDefaultSize().fWidth; }
   virtual UInt_t  GetDefaultHeight() const { return GetDefaultSize().fHeight; }
   virtual Pixel_t GetBackground() const { return fBackground; }
   virtual void    ChangeBackground(Pixel_t back);
           void    SetBackgroundColor(Pixel_t back) override;
   virtual Pixel_t GetForeground() const;
   virtual void    SetForegroundColor(Pixel_t /*fore*/) { }
   virtual UInt_t  GetOptions() const { return fOptions; }
   virtual void    ChangeOptions(UInt_t options);
   virtual void    Layout() { }
           void    MapSubwindows() override {}  // Simple frames do not have subwindows
                                        // Redefine this in TGCompositeFrame!
           void    ReparentWindow(const TGWindow *p, Int_t x = 0, Int_t y = 0) override
                     { TGWindow::ReparentWindow(p, x, y); Move(x, y); }
           void    MapWindow() override { TGWindow::MapWindow(); if (fFE) fFE->fState |= kIsVisible; }
           void    MapRaised() override { TGWindow::MapRaised(); if (fFE) fFE->fState |= kIsVisible; }
           void    UnmapWindow() override { TGWindow::UnmapWindow(); if (fFE) fFE->fState &= ~kIsVisible; }

   virtual void    DrawBorder();
   virtual void    DrawCopy(Handle_t /*id*/, Int_t /*x*/, Int_t /*y*/) { }
   virtual void    Activate(Bool_t) { }
   virtual Bool_t  IsActive() const { return kFALSE; }
   virtual Bool_t  IsComposite() const { return kFALSE; }
           Bool_t  IsEditable() const override { return kFALSE; }
           void    SetEditable(Bool_t) override {}
   virtual void    SetLayoutBroken(Bool_t = kTRUE) {}
   virtual Bool_t  IsLayoutBroken() const { return kFALSE; }
   virtual void    SetCleanup(Int_t = kLocalCleanup) { /* backward compatibility */ }

   virtual void    SetDragType(Int_t type);
   virtual void    SetDropType(Int_t type);
   virtual Int_t   GetDragType() const;
   virtual Int_t   GetDropType() const;

   UInt_t GetWidth() const { return fWidth; }
   UInt_t GetHeight() const { return fHeight; }
   UInt_t GetMinWidth() const { return fMinWidth; }
   UInt_t GetMinHeight() const { return fMinHeight; }
   UInt_t GetMaxWidth() const { return fMaxWidth; }
   UInt_t GetMaxHeight() const { return fMaxHeight; }
   TGDimension GetSize() const { return TGDimension(fWidth, fHeight); }
   Int_t  GetX() const { return fX; }
   Int_t  GetY() const { return fY; }
   Int_t  GetBorderWidth() const { return fBorderWidth; }

   TGFrameElement *GetFrameElement() const { return fFE; }
   void SetFrameElement(TGFrameElement *fe) { fFE = fe; }

   Bool_t Contains(Int_t x, Int_t y) const
      { return ((x >= 0) && (x < (Int_t)fWidth) && (y >= 0) && (y < (Int_t)fHeight)); }
   virtual TGFrame *GetFrameFromPoint(Int_t x, Int_t y)
      { return Contains(x, y) ? this : nullptr; }

   // Modifiers (without graphic update)
   virtual void SetX(Int_t x) { fX = x; }
   virtual void SetY(Int_t y) { fY = y; }
   virtual void SetWidth(UInt_t w) { fWidth = w; }
   virtual void SetHeight(UInt_t h) { fHeight = h; }
   virtual void SetMinWidth(UInt_t w) { fMinWidth = w; }
   virtual void SetMinHeight(UInt_t h) { fMinHeight = h; }
   virtual void SetMaxWidth(UInt_t w) { fMaxWidth = w; }
   virtual void SetMaxHeight(UInt_t h) { fMaxHeight = h; }
   virtual void SetSize(const TGDimension &s) { fWidth = s.fWidth; fHeight = s.fHeight; }

   // Printing and saving
   void Print(Option_t *option="") const override;
   void SaveUserColor(std::ostream &out, Option_t *);
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   // dummy to remove from context menu
   void        Delete(Option_t * /*option*/ ="") override {}
   TObject    *DrawClone(Option_t * /*option */="") const override { return nullptr; }
   void        DrawClass() const override {}
   void        Dump() const override {}
   void        Inspect() const override {}
   void        SetDrawOption(Option_t * /*option*/="") override {}

   // drag and drop...
   void                SetDNDSource(Bool_t onoff)
                       { if (onoff) fDNDState |= kIsDNDSource; else fDNDState &= ~kIsDNDSource; }
   void                SetDNDTarget(Bool_t onoff)
                       { if (onoff) fDNDState |= kIsDNDTarget; else fDNDState &= ~kIsDNDTarget; }
   Bool_t              IsDNDSource() const { return fDNDState & kIsDNDSource; }
   Bool_t              IsDNDTarget() const { return fDNDState & kIsDNDTarget; }

   virtual TDNDData   *GetDNDData(Atom_t /*dataType*/) { return 0; }
   virtual Bool_t      HandleDNDDrop(TDNDData * /*DNDData*/) { return kFALSE; }
   virtual Atom_t      HandleDNDPosition(Int_t /*x*/, Int_t /*y*/, Atom_t /*action*/,
                                         Int_t /*xroot*/, Int_t /*yroot*/) { return kNone; }
   virtual Atom_t      HandleDNDEnter(Atom_t * /*typelist*/) { return kNone; }
   virtual Bool_t      HandleDNDLeave() { return kFALSE; }
   virtual Bool_t      HandleDNDFinished() { return kFALSE; }

   ClassDefOverride(TGFrame,0)  // Base class for simple widgets (button, etc.)
};


class TGCompositeFrame : public TGFrame {


protected:
   TGLayoutManager *fLayoutManager;   ///< layout manager
   TList           *fList;            ///< container of frame elements
   Bool_t           fLayoutBroken;    ///< no layout manager is used
   Int_t            fMustCleanup;     ///< cleanup mode (see EFrameCleanup)
   Bool_t           fMapSubwindows;   ///< kTRUE - map subwindows

   static TGLayoutHints *fgDefaultHints;  // default hints used by AddFrame()

private:
   TGCompositeFrame(const TGCompositeFrame&) = delete;
   TGCompositeFrame& operator=(const TGCompositeFrame&) = delete;

public:
   TGCompositeFrame(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
                    UInt_t options = 0,
                    Pixel_t back = GetDefaultFrameBackground());
   TGCompositeFrame(TGClient *c, Window_t id, const TGWindow *parent = nullptr);
   virtual ~TGCompositeFrame();

   virtual TList *GetList() const { return fList; }

   UInt_t GetDefaultWidth() const override
                     { return GetDefaultSize().fWidth; }
   UInt_t GetDefaultHeight() const override
                     { return GetDefaultSize().fHeight; }
   TGDimension GetDefaultSize() const override
                     { return (IsLayoutBroken() ? TGDimension(fWidth, fHeight) :
                               fLayoutManager->GetDefaultSize()); }
   TGFrame *GetFrameFromPoint(Int_t x, Int_t y) override;
   virtual Bool_t TranslateCoordinates(TGFrame *child, Int_t x, Int_t y,
                                       Int_t &fx, Int_t &fy);
   void   MapSubwindows() override;
   void   Layout() override;
   Bool_t HandleButton(Event_t *) override { return kFALSE; }
   Bool_t HandleDoubleClick(Event_t *) override { return kFALSE; }
   Bool_t HandleCrossing(Event_t *) override { return kFALSE; }
   Bool_t HandleMotion(Event_t *) override { return kFALSE; }
   Bool_t HandleKey(Event_t *) override { return kFALSE; }
   Bool_t HandleFocusChange(Event_t *) override { return kFALSE; }
   Bool_t HandleSelection(Event_t *) override { return kFALSE; }
   Bool_t HandleDragEnter(TGFrame *) override;
   Bool_t HandleDragLeave(TGFrame *) override;
   Bool_t HandleDragMotion(TGFrame *) override;
   Bool_t HandleDragDrop(TGFrame *frame, Int_t x, Int_t y, TGLayoutHints *lo) override;
   void   ChangeOptions(UInt_t options) override;
   Bool_t ProcessMessage(Longptr_t, Longptr_t, Longptr_t) override { return kFALSE; }

   virtual TGLayoutManager *GetLayoutManager() const { return fLayoutManager; }
   virtual void SetLayoutManager(TGLayoutManager *l);

   virtual TGFrameElement* FindFrameElement(TGFrame *f) const;

   virtual void   AddFrame(TGFrame *f, TGLayoutHints *l = 0);
   virtual void   RemoveAll();
   virtual void   RemoveFrame(TGFrame *f);
   virtual void   ShowFrame(TGFrame *f);
   virtual void   HideFrame(TGFrame *f);
   Int_t          GetState(TGFrame *f) const;
   Bool_t         IsVisible(TGFrame *f) const;
   Bool_t         IsVisible(TGFrameElement *ptr) const { return (ptr->fState & kIsVisible); }
   Bool_t         IsArranged(TGFrame *f) const;
   Bool_t         IsArranged(TGFrameElement *ptr) const { return (ptr->fState & kIsArranged); }
   Bool_t         IsComposite() const override { return kTRUE; }
   Bool_t         IsEditable() const override;
   void           SetEditable(Bool_t on = kTRUE) override;
   void           SetLayoutBroken(Bool_t on = kTRUE) override;
   Bool_t         IsLayoutBroken() const  override { return fLayoutBroken || !fLayoutManager; }
   void           SetEditDisabled(UInt_t on = 1) override;
   void           SetCleanup(Int_t mode = kLocalCleanup) override;
   Int_t          MustCleanup() const override { return fMustCleanup; }
   virtual void   Cleanup();
   void           SetMapSubwindows(Bool_t on) override {  fMapSubwindows = on; }
   Bool_t         IsMapSubwindows() const override { return fMapSubwindows; }

   void           Print(Option_t *option="") const override;
   virtual void   ChangeSubframesBackground(Pixel_t back);
   void           SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void   SavePrimitiveSubframes(std::ostream &out, Option_t *option = "");

   ClassDefOverride(TGCompositeFrame,0)  // Base class for composite widgets (menubars, etc.)
};


class TGVerticalFrame : public TGCompositeFrame {
public:
   TGVerticalFrame(const TGWindow *p = 0, UInt_t w = 1, UInt_t h = 1,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground()) :
      TGCompositeFrame(p, w, h, options | kVerticalFrame, back) { SetWindowName(); }
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGVerticalFrame,0)  // Composite frame with vertical child layout
};

class TGHorizontalFrame : public TGCompositeFrame {
public:
   TGHorizontalFrame(const TGWindow *p = 0, UInt_t w = 1, UInt_t h = 1,
                     UInt_t options = kChildFrame,
                     Pixel_t back = GetDefaultFrameBackground()) :
      TGCompositeFrame(p, w, h, options | kHorizontalFrame, back) { SetWindowName(); }
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGHorizontalFrame,0)  // Composite frame with horizontal child layout
};


class TGMainFrame : public TGCompositeFrame {

protected:
   enum { kDontCallClose = BIT(14) };

   // mapping between key and window
   class TGMapKey : public TObject {
   private:
      TGMapKey(const TGMapKey&);
      TGMapKey& operator=(const TGMapKey&);
   public:
      UInt_t     fKeyCode;
      TGWindow  *fWindow;
      TGMapKey(UInt_t keycode, TGWindow *w): fKeyCode(keycode), fWindow(w) { }
   };

   Atom_t       *fDNDTypeList;  ///< handles DND types
   TList        *fBindList;     ///< list with key bindings
   TString       fWindowName;   ///< window name
   TString       fIconName;     ///< icon name
   TString       fIconPixmap;   ///< icon pixmap name
   TString       fClassName;    ///< WM class name
   TString       fResourceName; ///< WM resource name
   UInt_t        fMWMValue;     ///< MWM decoration hints
   UInt_t        fMWMFuncs;     ///< MWM functions
   UInt_t        fMWMInput;     ///< MWM input modes
   Int_t         fWMX;          ///< WM x position
   Int_t         fWMY;          ///< WM y position
   UInt_t        fWMWidth;      ///< WM width
   UInt_t        fWMHeight;     ///< WM height
   UInt_t        fWMMinWidth;   ///< WM min width
   UInt_t        fWMMinHeight;  ///< WM min height
   UInt_t        fWMMaxWidth;   ///< WM max width
   UInt_t        fWMMaxHeight;  ///< WM max height
   UInt_t        fWMWidthInc;   ///< WM width increments
   UInt_t        fWMHeightInc;  ///< WM height increments
   EInitialState fWMInitState;  ///< WM initial state

   TString GetMWMvalueString() const;  ///< used in SaveSource()
   TString GetMWMfuncString() const;   ///< used in SaveSource()
   TString GetMWMinpString() const;    ///< used in SaveSource()

private:
   TGMainFrame(const TGMainFrame&) = delete;
   TGMainFrame& operator=(const TGMainFrame&) = delete;

public:
   TGMainFrame(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
               UInt_t options = kVerticalFrame);
   virtual ~TGMainFrame();

   Bool_t HandleKey(Event_t *event) override;
   Bool_t HandleClientMessage(Event_t *event) override;
   Bool_t HandleSelection(Event_t *event) override;
   Bool_t HandleSelectionRequest(Event_t *event) override;
   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;
   virtual Bool_t SaveFrameAsCodeOrImage();
   virtual Bool_t SaveFrameAsCodeOrImage(const TString &fileName);
   virtual void   SendCloseMessage();
   virtual void   CloseWindow();   //*SIGNAL*

   void DontCallClose();
   void SetWindowName(const char *name = nullptr) override;
   void SetIconName(const char *name);
   const TGPicture *SetIconPixmap(const char *iconName);
   void SetIconPixmap(char **xpm_array);
   void SetClassHints(const char *className, const char *resourceName);
   void SetMWMHints(UInt_t value, UInt_t funcs, UInt_t input);
   void SetWMPosition(Int_t x, Int_t y);
   void SetWMSize(UInt_t w, UInt_t h);
   void SetWMSizeHints(UInt_t wmin, UInt_t hmin, UInt_t wmax, UInt_t hmax,
                       UInt_t winc, UInt_t hinc);
   void SetWMState(EInitialState state);

   virtual Bool_t BindKey(const TGWindow *w, Int_t keycode, Int_t modifier) const;
   virtual void   RemoveBind(const TGWindow *w, Int_t keycode, Int_t modifier) const;
   TList *GetBindList() const { return fBindList; }

   const char *GetWindowName() const { return fWindowName; }
   const char *GetIconName() const override { return fIconName; }
   const char *GetIconPixmap() const { return fIconPixmap; }
   void GetClassHints(const char *&className, const char *&resourceName) const
      { className = fClassName.Data(); resourceName = fResourceName.Data(); }
   void GetMWMHints(UInt_t &value, UInt_t &funcs, UInt_t &input) const
      { value = fMWMValue; funcs = fMWMFuncs; input = fMWMInput; }
   void GetWMPosition(Int_t &x, Int_t &y) const { x = fWMX; y = fWMY; }
   void GetWMSize(UInt_t &w, UInt_t &h) const { w = fWMWidth; h = fWMHeight; }
   void GetWMSizeHints(UInt_t &wmin, UInt_t &hmin, UInt_t &wmax, UInt_t &hmax,
                       UInt_t &winc, UInt_t &hinc) const
      { wmin = fWMMinWidth; hmin = fWMMinHeight; wmax = fWMMaxWidth;
        hmax = fWMMaxHeight; winc = fWMWidthInc; hinc = fWMHeightInc; }
   EInitialState GetWMState() const { return fWMInitState; }

   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void SaveSource(const char *filename = "Rootappl.C", Option_t *option = ""); // *MENU*icon=bld_save.png*

   ClassDefOverride(TGMainFrame,0)  // Top level window frame
};


class TGTransientFrame : public TGMainFrame {

protected:
   const TGWindow   *fMain;  // window over which to popup dialog

private:
   TGTransientFrame(const TGTransientFrame&) = delete;
   TGTransientFrame& operator=(const TGTransientFrame&) = delete;

public:
   TGTransientFrame(const TGWindow *p = nullptr, const TGWindow *main = nullptr, UInt_t w = 1, UInt_t h = 1,
                    UInt_t options = kVerticalFrame);

   enum EPlacement { kCenter, kLeft, kRight, kTop, kBottom, kTopLeft, kTopRight,
                     kBottomLeft, kBottomRight };
   virtual void    CenterOnParent(Bool_t croot = kTRUE, EPlacement pos = kCenter);
   const TGWindow *GetMain() const { return fMain; }
   void    SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SaveSource(const char *filename = "Rootdlog.C", Option_t *option = "") override; // *MENU*icon=bld_save.png*

   ClassDefOverride(TGTransientFrame,0)  // Frame for dialog (transient) windows
};


class TGGroupFrame : public TGCompositeFrame {

protected:
   TGString      *fText;         ///< title text
   FontStruct_t   fFontStruct;   ///< title fontstruct
   GContext_t     fNormGC;       ///< title graphics context
   Int_t          fTitlePos;     ///< *OPTION={GetMethod="GetTitlePos";SetMethod="SetTitlePos";Items=(-1="Left",0="Center",1="Right")}*
   Bool_t         fHasOwnFont;   ///< kTRUE - font defined locally,  kFALSE - globally

   void DoRedraw() override;

   static const TGFont *fgDefaultFont;
   static const TGGC   *fgDefaultGC;

private:
   TGGroupFrame(const TGGroupFrame&) = delete;
   TGGroupFrame& operator=(const TGGroupFrame&) = delete;

public:
   enum ETitlePos { kLeft = -1, kCenter = 0, kRight = 1 };

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   TGGroupFrame(const TGWindow *p, TGString *title,
                UInt_t options = kVerticalFrame,
                GContext_t norm = GetDefaultGC()(),
                FontStruct_t font = GetDefaultFontStruct(),
                Pixel_t back = GetDefaultFrameBackground());
   TGGroupFrame(const TGWindow *p = nullptr, const char *title = nullptr,
                UInt_t options = kVerticalFrame,
                GContext_t norm = GetDefaultGC()(),
                FontStruct_t font = GetDefaultFontStruct(),
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGGroupFrame();

   TGDimension GetDefaultSize() const override;
   void  DrawBorder() override;
   virtual void  SetTitle(TGString *title);
   virtual void  SetTitle(const char *title);
   virtual void  Rename(const char *title)  { SetTitle(title); } //*MENU*icon=bld_rename.png*
           Int_t GetTitlePos() const { return fTitlePos; }
   virtual void  SetTitlePos(ETitlePos pos = kLeft) { fTitlePos = pos; }  //*SUBMENU*
   virtual void  SetTextColor(Pixel_t color, Bool_t local = kTRUE);
   virtual void  SetTextFont(const char *fontName, Bool_t local = kTRUE);
   virtual void  SetTextFont(FontStruct_t font, Bool_t local = kTRUE);
   GContext_t GetNormGC() const { return fNormGC; }
   FontStruct_t GetFontStruct() const { return fFontStruct; }

   const char *GetTitle() const override { return fText->GetString(); }
   Bool_t HasOwnFont() const;

   void  SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGGroupFrame,0)  // A composite frame with border and title
};


class TGHeaderFrame : public TGHorizontalFrame {
private:
   TGHeaderFrame(const TGHeaderFrame&) = delete;
   TGHeaderFrame& operator=(const TGHeaderFrame&) = delete;

protected:
   Int_t              fNColumns;     ///< number of columns
   TGTextButton     **fColHeader;    ///< column headers for in detailed mode
   TGVFileSplitter  **fSplitHeader;  ///< column splitters
   Cursor_t           fSplitCursor;  ///< split cursor;
   Bool_t             fOverSplitter; ///< Indicates if the cursor is over a splitter
   Int_t              fOverButton;   ///< Indicates over which button the mouse is
   Int_t              fLastButton;   ///< Indicates the last button clicked if any

public:
   TGHeaderFrame(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
                 UInt_t options = kChildFrame,
                 Pixel_t back = GetDefaultFrameBackground());

   Bool_t HandleButton(Event_t* event) override;
   Bool_t HandleMotion(Event_t* event) override;
   Bool_t HandleDoubleClick(Event_t *event) override;

   void SetColumnsInfo(Int_t nColumns, TGTextButton  **colHeader, TGVFileSplitter  **splitHeader);

   ClassDefOverride(TGHeaderFrame,0)  // Header frame with buttons and splitters
};


#endif
