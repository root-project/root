// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   03/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGFrame
#define ROOT_TGFrame


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFrame, TGCompositeFrame, TGVerticalFrame, TGHorizontalFrame,       //
// TGMainFrame, TGTransientFrame and TGGroupFrame                       //
//                                                                      //
// This header contains all different Frame classes.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGWindow
#include "TGWindow.h"
#endif
#ifndef ROOT_TGDimension
#include "TGDimension.h"
#endif
#ifndef ROOT_TGLayout
#include "TGLayout.h"
#endif
#ifndef ROOT_TGString
#include "TGString.h"
#endif

class TList;


//---- frame states

enum EFrameState {
   kIsVisible  = BIT(0),
   kIsMapped   = kIsVisible,
   kIsArranged = BIT(1)
};

//---- types of frames (and borders)

enum EFrameType {
   kChildFrame      = 0,
   kMainFrame       = BIT(0),
   kVerticalFrame   = BIT(1),
   kHorizontalFrame = BIT(2),
   kSunkenFrame     = BIT(3),
   kRaisedFrame     = BIT(4),
   kDoubleBorder    = BIT(5),
   kFitWidth        = BIT(6),
   kFixedWidth      = BIT(7),
   kFitHeight       = BIT(8),
   kFixedHeight     = BIT(9),
   kFixedSize       = (kFixedWidth | kFixedHeight)
};

//---- MWM Hints stuff

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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFrame                                                              //
//                                                                      //
// This class subclasses TGWindow, used as base class for some simple   //
// widgets (buttons, labels, etc.).                                     //
// It provides:                                                         //
//  - position & dimension fields                                       //
//  - an 'options' attribute (see constant above)                       //
//  - a generic event handler                                           //
//  - a generic layout mechanism                                        //
//  - a generic border                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGFrame : public TGWindow {

friend class TGClient;

protected:
   Int_t    fX;             // frame x position
   Int_t    fY;             // frame y position
   UInt_t   fWidth;         // frame width
   UInt_t   fHeight;        // frame height
   Int_t    fBorderWidth;   // frame border width
   UInt_t   fOptions;       // frame options
   ULong_t  fBackground;    // frame background color

   static ULong_t     fgDefaultFrameBackground;
   static ULong_t     fgDefaultSelectedBackground;
   static ULong_t     fgWhitePixel;
   static ULong_t     fgBlackPixel;
   static GContext_t  fgBlackGC, fgWhiteGC;
   static GContext_t  fgHilightGC;
   static GContext_t  fgShadowGC;
   static GContext_t  fgBckgndGC;
   static Time_t      fgLastClick;
   static UInt_t      fgLastButton, fgDbx, fgDby;
   static Window_t    fgDbw;

   static ULong_t     GetDefaultFrameBackground();
   static ULong_t     GetDefaultSelectedBackground();
   static ULong_t     GetWhitePixel();
   static ULong_t     GetBlackPixel();
   static GContext_t  GetBlackGC();
   static GContext_t  GetWhiteGC();
   static GContext_t  GetHilightGC();
   static GContext_t  GetShadowGC();
   static GContext_t  GetBckgndGC();
   static Time_t      GetLastClick();

   virtual void DoRedraw();
   virtual const TGWindow *GetMainFrame() const { return TGWindow::GetMainFrame(); }

public:
   TGFrame(const TGWindow *p, UInt_t w, UInt_t h,
           UInt_t options = 0, ULong_t back = fgDefaultFrameBackground);
   TGFrame(TGClient *c, Window_t id, const TGWindow *parent = 0);
   virtual ~TGFrame() { }

   virtual Bool_t HandleEvent(Event_t *event);
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

   virtual void SendMessage(const TGWindow *w, Long_t msg, Long_t parm1, Long_t parm2);
   virtual Bool_t ProcessMessage(Long_t, Long_t, Long_t) { return kFALSE; }

   virtual void Move(Int_t x, Int_t y);
   virtual void Resize(UInt_t w, UInt_t h);
   virtual void Resize(TGDimension size);
   virtual void MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual UInt_t GetDefaultWidth() const { return GetDefaultSize().fWidth; }
   virtual UInt_t GetDefaultHeight() const { return GetDefaultSize().fHeight; }
   virtual TGDimension GetDefaultSize() const
      { return TGDimension(fWidth, fHeight); }

   virtual ULong_t GetBackground() const { return fBackground; }
   virtual void    ChangeBackground(ULong_t back);
   virtual UInt_t  GetOptions() const { return fOptions; }
   virtual void    ChangeOptions(UInt_t options);
   virtual void    Layout() { }
   virtual void    MapSubwindows() { }  // Simple frames do not have subwindows
                                        // Redefine this in TGCompositeFrame!
   virtual void    DrawBorder();

   UInt_t GetWidth() const { return fWidth; }
   UInt_t GetHeight() const { return fHeight; }
   TGDimension GetSize() const { return TGDimension(fWidth, fHeight); }
   Int_t GetX() const { return fX; }
   Int_t GetY() const { return fY; }
   Int_t GetBorderWidth() const { return fBorderWidth; }

   // Modifiers (without graphic update)
   void SetWidth(UInt_t w) { fWidth = w; }
   void SetHeight(UInt_t h) { fHeight = h; }
   void SetSize(const TGDimension &s) { fWidth = s.fWidth; fHeight = s.fHeight; }

   ClassDef(TGFrame,0)  // Base class for simple widgets (button, etc.)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGCompositeFrame                                                     //
//                                                                      //
// This class is the base class for composite widgets                   //
// (menu bars, list boxes, etc.).                                       //
//                                                                      //
// It provides:                                                         //
//  - a layout manager                                                  //
//  - a frame container (TList *)                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGCompositeFrame : public TGFrame {

private:
   TGLayoutManager *fLayoutManager;   // layout manager

protected:
   TList           *fList;            // container of frame elements

   static TGLayoutHints *fgDefaultHints;  // default hints used by AddFrame()

public:
   TGCompositeFrame(const TGWindow *p, UInt_t w, UInt_t h,
                    UInt_t options = 0,
                    ULong_t back = fgDefaultFrameBackground);
   TGCompositeFrame(TGClient *c, Window_t id, const TGWindow *parent = 0);
   virtual ~TGCompositeFrame();

   virtual UInt_t GetDefaultWidth() const
                 { return GetDefaultSize().fWidth; }
   virtual UInt_t GetDefaultHeight() const
                 { return GetDefaultSize().fHeight; }
   virtual TGDimension GetDefaultSize() const
                 { return fLayoutManager->GetDefaultSize(); }

   virtual void   MapSubwindows();
   virtual void   Layout();
   virtual Bool_t HandleButton(Event_t *) { return kFALSE; }
   virtual Bool_t HandleDoubleClick(Event_t *) { return kFALSE; }
   virtual Bool_t HandleCrossing(Event_t *) { return kFALSE; }
   virtual Bool_t HandleMotion(Event_t *) { return kFALSE; }
   virtual Bool_t HandleKey(Event_t *) { return kFALSE; }
   virtual Bool_t HandleFocusChange(Event_t *) { return kFALSE; }
   virtual Bool_t HandleSelection(Event_t *) { return kFALSE; }
   virtual void   ChangeOptions(UInt_t options);
   virtual Bool_t ProcessMessage(Long_t, Long_t, Long_t) { return kFALSE; }

   TGLayoutManager *GetLayoutManager() const { return fLayoutManager; }
   void             SetLayoutManager(TGLayoutManager *l);

   virtual void AddFrame(TGFrame *f, TGLayoutHints *l = 0);
   void   RemoveFrame(TGFrame *f);
   void   ShowFrame(TGFrame *f);
   void   HideFrame(TGFrame *f);
   Int_t  GetState(TGFrame *f) const;
   Bool_t IsVisible(TGFrame *f) const;
   Bool_t IsVisible(TGFrameElement *ptr) const { return (ptr->fState & kIsVisible); }
   Bool_t IsArranged(TGFrame *f) const;
   Bool_t IsArranged(TGFrameElement *ptr) const { return (ptr->fState & kIsArranged); }
   TList *GetList() { return fList; }

   ClassDef(TGCompositeFrame,0)  // Base class for composite widgets (menubars, etc.)
};


class TGVerticalFrame : public TGCompositeFrame {
public:
   TGVerticalFrame(const TGWindow *p, UInt_t w, UInt_t h,
                   UInt_t options = kChildFrame,
                   ULong_t back = fgDefaultFrameBackground) :
      TGCompositeFrame(p, w, h, options | kVerticalFrame, back) { }

   ClassDef(TGVerticalFrame,0)  // Composite frame with vertical child layout
};

class TGHorizontalFrame : public TGCompositeFrame {
public:
   TGHorizontalFrame(const TGWindow *p, UInt_t w, UInt_t h,
                     UInt_t options = kChildFrame,
                     ULong_t back = fgDefaultFrameBackground) :
      TGCompositeFrame(p, w, h, options | kHorizontalFrame, back) { }

   ClassDef(TGHorizontalFrame,0)  // Composite frame with horizontal child layout
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMainFrame                                                          //
//                                                                      //
// This class defines top level windows that interact with the system   //
// Window Manager (WM or MWM for Motif Window Manager).                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGMainFrame : public TGCompositeFrame {

protected:
   TList    *fBindList;    // list with key bindings

   virtual const TGWindow *GetMainFrame() const { return this; }

public:
   TGMainFrame(const TGWindow *p, UInt_t w, UInt_t h,
               UInt_t options = kMainFrame | kVerticalFrame);
   virtual ~TGMainFrame();

   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleClientMessage(Event_t *event);
   virtual void   CloseWindow();

   void SetWindowName(const char *name);
   void SetIconName(const char *name);
   void SetIconPixmap(const char *iconName);
   void SetClassHints(const char *className, const char *resourceName);
   void SetMWMHints(UInt_t value, UInt_t funcs, UInt_t input);
   void SetWMPosition(Int_t x, Int_t y);
   void SetWMSize(UInt_t w, UInt_t h);
   void SetWMSizeHints(UInt_t wmin, UInt_t hmin, UInt_t wmax, UInt_t hmax,
                       UInt_t winc, UInt_t hinc);
   void SetWMState(EInitialState state);

   virtual Bool_t BindKey(const TGWindow *w, Int_t keycode, Int_t modifier) const;
   virtual void   RemoveBind(const TGWindow *w, Int_t keycode, Int_t modifier) const;

   ClassDef(TGMainFrame,0)  // Top level window frame
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTransientFrame                                                     //
//                                                                      //
// This class defines transient windows that typically are used for     //
// dialogs.                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGTransientFrame : public TGMainFrame {

protected:
   const TGWindow   *fMain;  // window over which to popup dialog

public:
   TGTransientFrame(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
                    UInt_t options = kMainFrame | kVerticalFrame);

   const TGWindow *GetMain() const { return fMain; }
   virtual void CloseWindow();

   ClassDef(TGTransientFrame,0)  // Frame for dialog (transient) windows
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGGroupFrame                                                         //
//                                                                      //
// A group frame is a composite frame with a border and a title.        //
// It is typically used to group a number of logically related widgets  //
// visually together.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGGroupFrame : public TGCompositeFrame {

friend class TGClient;

protected:
   TGString      *fText;
   FontStruct_t   fFontStruct;
   GContext_t     fNormGC;

   static GContext_t    fgDefaultGC;
   static FontStruct_t  fgDefaultFontStruct;

public:
   TGGroupFrame(const TGWindow *p, TGString *title,
                UInt_t options = kVerticalFrame,
                GContext_t norm = fgDefaultGC,
                FontStruct_t font = fgDefaultFontStruct,
                ULong_t back = fgDefaultFrameBackground);
   TGGroupFrame(const TGWindow *p, const char *title,
                UInt_t options = kVerticalFrame,
                GContext_t norm = fgDefaultGC,
                FontStruct_t font = fgDefaultFontStruct,
                ULong_t back = fgDefaultFrameBackground);
   virtual ~TGGroupFrame();

   virtual void DrawBorder();

   ClassDef(TGGroupFrame,0)  // A composite frame with border and title
};

#endif
