// @(#)root/gui:$Id$
// Author: Bertrand Bellenot   20/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGMdiDecorFrame
#define ROOT_TGMdiDecorFrame


#include "TGFrame.h"
#include "TGMenu.h"
#include "TGButton.h"
#include "TGIcon.h"
#include "TGLabel.h"
#include "TGFont.h"
#include "TGMdiMainFrame.h"
#include "TGMdiFrame.h"


/// placement of window resizers
enum EMdiResizerPlacement {
   kMdiResizerTop    = 1,
   kMdiResizerBottom = 2,
   kMdiResizerLeft   = 4,
   kMdiResizerRight  = 8
};


class TGMdiMainFrame;
class TGMdiDecorFrame;
class TGMdiFrame;
class TGMdiTitleBar;
class TGMdiTitleIcon;
class TGMdiButtons;


//----------------------------------------------------------------------

class TGMdiWinResizer : public TGFrame, public TGWidget {

friend class TGMdiMainFrame;

protected:
   const TGWindow  *fMdiWin;                       ///< owner MDI Window
   Int_t            fWinX, fWinY, fWinW, fWinH;    ///< MDI win position and dimension
   Int_t            fOldX, fOldY, fOldW, fOldH;    ///< old MDI win position and dimension
   Int_t            fNewX, fNewY, fNewW, fNewH;    ///< new MDI win position and dimension
   Int_t            fMinW, fMinH;                  ///< Min width and height of the MDI Win
   Int_t            fMdiOptions;                   ///< MDI options
   Int_t            fPos;                          ///< position of the resizer
   Int_t            fX0, fY0;                      ///< initial position of the mouse click
   Bool_t           fLeftButPressed, fRightButPressed, fMidButPressed; ///< mouse buttons flags

   const TGGC      *fBoxGC;                        ///< GC used for resizing rectangle
   Int_t            fLineW;                        ///< line width of resizing rectangle

   void             MoveResizeIt();
   void             DrawBox(Int_t x, Int_t y, UInt_t width, UInt_t height);

public:
   TGMdiWinResizer(const TGWindow *p, const TGWindow *mdiwin, Int_t pos,
                   const TGGC *boxGC, Int_t linew,
                   Int_t mdioptions = kMdiDefaultResizeMode,
                   Int_t w = 1, Int_t h = 1, UInt_t options = kOwnBackground);

   virtual Bool_t HandleButton(Event_t *event);
   virtual void   DrawBorder() {};

   void SetResizeMode(Int_t mode) { fMdiOptions = mode; }
   void SetMinSize(Int_t w = 50, Int_t h = 20) { fMinW = w; fMinH = h; }

   ClassDef(TGMdiWinResizer, 0) // Window Resizer
};


class TGMdiVerticalWinResizer : public TGMdiWinResizer {

public:
   TGMdiVerticalWinResizer(const TGWindow *p, const TGWindow *mdiwin,
                           Int_t pos, const TGGC *boxGC, Int_t linew,
                           Int_t mdioptions = kMdiDefaultResizeMode,
                           Int_t w = 4, Int_t h = 5);

   virtual Bool_t HandleMotion(Event_t *event);
   virtual void   DrawBorder();

   ClassDef(TGMdiVerticalWinResizer, 0) // Vertical Window Resizer
};


class TGMdiHorizontalWinResizer : public TGMdiWinResizer {

public:
   TGMdiHorizontalWinResizer(const TGWindow *p, const TGWindow *mdiwin,
                             Int_t pos, const TGGC *boxGC, Int_t linew,
                             Int_t mdioptions = kMdiDefaultResizeMode,
                             Int_t w = 5, Int_t h = 4);

   virtual Bool_t HandleMotion(Event_t *event);
   virtual void   DrawBorder();

   ClassDef(TGMdiHorizontalWinResizer, 0) // Horizontal Window Resizer
};


class TGMdiCornerWinResizer : public TGMdiWinResizer {

public:
   TGMdiCornerWinResizer(const TGWindow *p, const TGWindow *mdiwin,
                         Int_t pos, const TGGC *boxGC, Int_t linew,
                         Int_t mdioptions = kMdiDefaultResizeMode,
                         Int_t w = 20, Int_t h = 20);

   virtual Bool_t  HandleMotion(Event_t *event);
   virtual void DrawBorder();

   ClassDef(TGMdiCornerWinResizer, 0) // Corner Window Resizer
};


//----------------------------------------------------------------------

class TGMdiButtons : public TGCompositeFrame {

friend class TGMdiTitleBar;

protected:
   TGPictureButton      *fButton[5];                  ///< MDI picture buttons
   TGLayoutHints        *fDefaultHint, *fCloseHint;   ///< Layout hints
   const TGWindow       *fMsgWindow;                  ///< window handling container messages

public:
   TGMdiButtons(const TGWindow *p, const TGWindow *titlebar);
   virtual ~TGMdiButtons();

   TGPictureButton *GetButton(Int_t no) const { return fButton[no]; }

   ClassDef(TGMdiButtons, 0)  // MDI Buttons Frame
};


//----------------------------------------------------------------------

class TGMdiTitleIcon : public TGIcon {

friend class TGMdiFrame;
friend class TGMdiTitleBar;

protected:
   const TGWindow   *fMsgWindow;    ///< window handling container messages
   TGPopupMenu      *fPopup;        ///< Popup menu associated to the title icon

   virtual void     DoRedraw();

public:
   TGMdiTitleIcon(const TGWindow *p, const TGWindow *titlebar,
                  const TGPicture *pic, Int_t w, Int_t h);
   virtual ~TGMdiTitleIcon();

   virtual Bool_t HandleDoubleClick(Event_t *event);
   virtual Bool_t HandleButton(Event_t *event);
   TGPopupMenu *GetPopup() const { return fPopup; }

   ClassDef(TGMdiTitleIcon, 0) // MDI Title Icon
};


//----------------------------------------------------------------------

class TGMdiTitleBar : public TGCompositeFrame {

friend class TGMdiDecorFrame;
friend class TGMdiMainFrame;

protected:
   const TGWindow       *fMdiWin;         ///< owner MDI window
   TGMdiButtons         *fButtons;        ///< MDI buttons
   TGMdiTitleIcon       *fWinIcon;        ///< title icon
   TGLabel              *fWinName;        ///< window name label
   TGCompositeFrame     *fLFrame, *fMFrame,*fRFrame; ///< the three parts of the title bar
   TGLayoutHints        *fLHint, *fLeftHint, *fMiddleHint, *fRightHint; ///< layout hints
   Int_t                fX0, fY0;         ///< initial position of the mouse click
   Bool_t               fLeftButPressed, fRightButPressed, fMidButPressed; ///< mouse button pressed

   TGMdiTitleBar(const TGWindow *p, const TGWindow *mdiwin,
                 const char *name = "Untitled");
   void LayoutButtons(UInt_t buttonmask, Bool_t isMinimized,
                      Bool_t isMaximized);

   void AddFrames(TGMdiTitleIcon *icon, TGMdiButtons *buttons);
   void RemoveFrames(TGMdiTitleIcon *icon, TGMdiButtons *buttons);

public:
   virtual ~TGMdiTitleBar();

   virtual Bool_t       HandleButton(Event_t *event);
   virtual Bool_t       HandleDoubleClick(Event_t *event);
   virtual Bool_t       HandleMotion(Event_t *event);
   virtual Bool_t       ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   void                 SetTitleBarColors(UInt_t fore, UInt_t back, TGFont *f);
   TGMdiButtons        *GetButtons() const { return fButtons; }
   TGMdiTitleIcon      *GetWinIcon() const { return fWinIcon; }
   TGLabel             *GetWinName() const { return fWinName; }
   Int_t                GetX0() { return fX0; }
   Int_t                GetY0() { return fY0; }
   Bool_t               IsLeftButPressed() { return fLeftButPressed; }
   Bool_t               IsRightButPressed() { return fRightButPressed; }
   Bool_t               IsMidButPressed() { return fMidButPressed; }

   void                 SetX0(Int_t x0) { fX0 = x0; }
   void                 SetY0(Int_t y0) { fY0 = y0; }
   void                 SetLeftButPressed(Bool_t press = kTRUE) { fLeftButPressed = press; }
   void                 SetRightButPressed(Bool_t press = kTRUE) { fRightButPressed= press; }
   void                 SetMidButPressed(Bool_t press = kTRUE) { fMidButPressed = press; }

   ClassDef(TGMdiTitleBar, 0) // MDI Title Bar
};


//----------------------------------------------------------------------

class TGMdiDecorFrame : public TGCompositeFrame {

friend class TGMdiMainFrame;

protected:
   Int_t                      fPreResizeX, fPreResizeY;           ///< position before resizing
   Int_t                      fPreResizeWidth, fPreResizeHeight;  ///< dimension before resizing
   Int_t                      fMinimizedX, fMinimizedY;           ///< minimized position
   Bool_t                     fIsMinimized, fIsMaximized;         ///< minimized and maximized flags
   Bool_t                     fMinimizedUserPlacement;            ///< user placement flag
   Bool_t                     fIsCurrent;                         ///< current mdi window flag

   TGMdiFrame                *fFrame;                             ///< parent (owner) MDI frame
   TGMdiMainFrame            *fMdiMainFrame;                      ///< MDI main frame

   TGMdiVerticalWinResizer   *fUpperHR, *fLowerHR;                ///< upper and lower vertical resizers
   TGMdiCornerWinResizer     *fUpperLeftCR, *fLowerLeftCR;        ///< upper left and lower left resizers
   TGMdiCornerWinResizer     *fUpperRightCR, *fLowerRightCR;      ///< upper right and lower right resizers
   TGMdiHorizontalWinResizer *fLeftVR, *fRightVR;                 ///< left and right horizontal resizers
   TGLayoutHints             *fLHint, *fExpandHint;               ///< layout hints

   ULong_t                    fButtonMask;                        ///< mask used to display/hide buttons
   TGMdiTitleBar             *fTitlebar;                          ///< MDI window title bar

public:
   enum {
      // border width of decorated windows
      kMdiBorderWidth = 5
   };

   TGMdiDecorFrame(TGMdiMainFrame *main, TGMdiFrame *frame, Int_t w, Int_t h,
                   const TGGC *boxGC, UInt_t options = 0,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGMdiDecorFrame();

   virtual Bool_t   HandleButton(Event_t *event);
   virtual Bool_t   HandleConfigureNotify(Event_t *event);

   virtual Int_t    CloseWindow() { return fFrame->CloseWindow(); }
   virtual void     Layout();

   virtual void     Move(Int_t x, Int_t y);
   virtual void     MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h);

   void             SetMdiButtons(ULong_t buttons);
   ULong_t          GetMdiButtons() const { return fButtonMask; }

   void             SetResizeMode(Int_t mode = kMdiDefaultResizeMode);

   void             SetWindowName(const char *name);
   void             SetWindowIcon(const TGPicture *pic);
   const char      *GetWindowName() {
                     return (const char *)fTitlebar->GetWinName()->GetText()->GetString();
                     }
   const TGPicture *GetWindowIcon() { return fTitlebar->GetWinIcon()->GetPicture(); }
   Bool_t           IsCurrent() const { return fIsCurrent; }
   Bool_t           IsMinimized() const { return fIsMinimized; }
   Bool_t           IsMaximized() const { return fIsMaximized; }
   Int_t            GetPreResizeX() const { return fPreResizeX; }
   Int_t            GetPreResizeY() const { return fPreResizeY; }
   Int_t            GetPreResizeWidth() const { return fPreResizeWidth; }
   Int_t            GetPreResizeHeight() const { return fPreResizeHeight; }
   Int_t            GetMinimizedX() const { return fMinimizedX; }
   Int_t            GetMinimizedY() const { return fMinimizedY; }
   Bool_t           GetMinUserPlacement() const { return fMinimizedUserPlacement; }

   void             SetCurrent(Bool_t cur = kTRUE) {fIsCurrent = cur; }
   void             SetDecorBorderWidth(Int_t bw);
   void             SetPreResizeX(Int_t x) { fPreResizeX = x; }
   void             SetPreResizeY(Int_t y) { fPreResizeY = y; }
   void             SetPreResizeWidth(Int_t w) { fPreResizeWidth = w; }
   void             SetPreResizeHeight(Int_t h) { fPreResizeHeight = h; }
   void             SetMinimizedX(Int_t x) { fMinimizedX = x; }
   void             SetMinimizedY(Int_t y) { fMinimizedY = y; }
   void             Minimize(Bool_t min = kTRUE) { fIsMinimized = min; }
   void             Maximize(Bool_t max = kTRUE) { fIsMaximized = max; }
   void             SetMinUserPlacement(Bool_t place = kTRUE) { fMinimizedUserPlacement = place; }

   TGMdiFrame      *GetMdiFrame() const { return fFrame; }
   TGMdiTitleBar   *GetTitleBar() const { return fTitlebar; }

   TGMdiVerticalWinResizer   *GetUpperHR() const { return fUpperHR; }
   TGMdiVerticalWinResizer   *GetLowerHR() const { return fLowerHR; }
   TGMdiCornerWinResizer     *GetUpperLeftCR() const { return fUpperLeftCR; }
   TGMdiCornerWinResizer     *GetLowerLeftCR() const { return fLowerLeftCR; }
   TGMdiCornerWinResizer     *GetUpperRightCR() const { return fUpperRightCR; }
   TGMdiCornerWinResizer     *GetLowerRightCR() const { return fLowerRightCR; }
   TGMdiHorizontalWinResizer *GetLeftVR() const { return fLeftVR; }
   TGMdiHorizontalWinResizer *GetRightVR() const { return fRightVR; }

   ClassDef(TGMdiDecorFrame, 0) // MDI Decor Frame
};

#endif
