// @(#)root/gui:$Id$
// Author: Bertrand Bellenot 23/01/2008

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGSplitFrame
#define ROOT_TGSplitFrame

#include "TGFrame.h"

#include "TMap.h"

class TGSplitter;
class TContextMenu;

class TGRectMap : public TObject {

private:
   TGRectMap(const TGRectMap&) = delete;
   TGRectMap& operator=(const TGRectMap&) = delete;

public:
   Int_t         fX;    ///< x position
   Int_t         fY;    ///< y position
   UInt_t        fW;    ///< width
   UInt_t        fH;    ///< height

   // constructors
   TGRectMap(Int_t rx, Int_t ry, UInt_t rw, UInt_t rh):
             fX(rx), fY(ry), fW(rw), fH(rh) { }
   virtual ~TGRectMap() { }

   // methods
   Bool_t Contains(Int_t px, Int_t py) const
                { return ((px >= fX) && (px < fX + (Int_t) fW) &&
                          (py >= fY) && (py < fY + (Int_t) fH)); }

   ClassDef(TGRectMap, 0)  // Rectangle used in TMap
};

class TGSplitTool : public TGCompositeFrame {

private:
   const TGFrame     *fWindow;      ///< frame to which tool tip is associated
   TGGC               fRectGC;      ///< rectangles drawing context
   TMap               fMap;         ///< map of rectangles/subframes
   TContextMenu      *fContextMenu; ///< Context menu for the splitter
   Int_t              fX;           ///< X position in fWindow where to popup
   Int_t              fY;           ///< Y position in fWindow where to popup

   TGSplitTool(const TGSplitTool&) = delete;
   TGSplitTool& operator=(const TGSplitTool&) = delete;

public:
   TGSplitTool(const TGWindow *p = nullptr, const TGFrame *f = nullptr);
   virtual ~TGSplitTool();

   void   AddRectangle(TGFrame *frm, Int_t x, Int_t y, Int_t w, Int_t h);
   void   DoRedraw();
   void   DrawBorder();
   Bool_t HandleButton(Event_t *event);
   Bool_t HandleMotion(Event_t *event);
   void   Show(Int_t x, Int_t y);
   void   Hide();
   void   Reset();
   void   SetPosition(Int_t x, Int_t y);

   ClassDef(TGSplitTool, 0)  // Split frame tool utility
};

class TGSplitFrame : public TGCompositeFrame {

private:
   TGSplitFrame(const TGSplitFrame&) = delete;
   TGSplitFrame& operator=(const TGSplitFrame&) = delete;

protected:
   TGFrame          *fFrame;       ///< Pointer to the embedded frame (if any)
   TGTransientFrame *fUndocked;    ///< Main frame used when "undocking" frame
   TGSplitter       *fSplitter;    ///< Pointer to the (H/V) Splitter (if any)
   TGSplitFrame     *fFirst;       ///< Pointer to the first child (if any)
   TGSplitFrame     *fSecond;      ///< Pointer to the second child (if any)
   TGSplitTool      *fSplitTool;   ///< SplitFrame Tool
   Float_t           fWRatio;      ///< Width ratio between the first child and this
   Float_t           fHRatio;      ///< Height ratio between the first child and this

public:
   TGSplitFrame(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
                UInt_t options = 0);
   virtual ~TGSplitFrame();

   virtual void   AddFrame(TGFrame *f, TGLayoutHints *l = nullptr);
   virtual void   Cleanup();
   virtual Bool_t HandleConfigureNotify(Event_t *);
   virtual void   HSplit(UInt_t h = 0);
   virtual void   VSplit(UInt_t w = 0);
   virtual void   RemoveFrame(TGFrame *f);

   TGSplitFrame  *GetFirst() const { return fFirst; }
   TGFrame       *GetFrame() const { return fFrame; }
   TGSplitFrame  *GetSecond() const { return fSecond; }
   TGSplitter    *GetSplitter() const { return fSplitter; }
   TGSplitTool   *GetSplitTool() const { return fSplitTool; }
   TGSplitFrame  *GetTopFrame();
   TGFrame       *GetUndocked() const { return fUndocked; }
   Float_t        GetHRatio() const { return fHRatio; }
   Float_t        GetWRatio() const { return fWRatio; }
   void           MapToSPlitTool(TGSplitFrame *top);
   void           OnSplitterClicked(Event_t *event);
   void           SetHRatio(Float_t r) { fHRatio = r; }
   void           SetWRatio(Float_t r) { fWRatio = r; }
   void           SplitHorizontal(const char *side = "top");
   void           SplitVertical(const char *side = "left");
   void           UnSplit(const char *which);

   // methods accessible via context menu

   void           Close();             // *MENU*
   void           CloseAndCollapse();  // *MENU*
   void           ExtractFrame();      // *MENU*
   void           SwallowBack();       // *MENU*
   void           SwitchToMain();      // *MENU*
   void           SplitHor();          // *MENU*
   void           SplitVer();          // *MENU*

   void           Docked(TGFrame* frame);    //*SIGNAL*
   void           Undocked(TGFrame* frame);  //*SIGNAL*

   static  void   SwitchFrames(TGFrame *frame, TGCompositeFrame *dest,
                               TGFrame *prev);
   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGSplitFrame, 0) // Splittable composite frame
};

#endif
