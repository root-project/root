// @(#)root/guibuilder:$Name:  $:$Id: TGFrame.cxx,v 1.78 2004/09/13 09:10:08 rdm Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGuiBldDragManager
#define ROOT_TGuiBldDragManager


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldDragManager                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

#ifndef ROOT_TVirtualDragManager
#include "TVirtualDragManager.h"
#endif

class TTimer;
class TGuiBldDragManagerPimpl;
class TGuiBuilder;
class TQUndoManager;
class TGPopupMenu;
class TGuiBldEditor;
class TGuiBldQuickHandler;

class TGuiBldDragManager : public TVirtualDragManager, public TGFrame {

friend class TGClient;
friend class TGFrame;
friend class TGMainFrame;

private:
   TGuiBldDragManagerPimpl *fPimpl;    // private data

   TGuiBuilder   *fBuilder;            // pointer to gui builder
   TGuiBldEditor *fEditor;             // frame property editor
   TGuiBldQuickHandler *fQuickHandler; // quick action handler
   Bool_t         fLassoDrawn;         // kTRUE if  lasso drawn
   TString        fPasteFileName;      // 
   Bool_t         fSelectionIsOn;      // selection with Shift key pressed
   TGPopupMenu   *fFrameMenu;          // context menu for frames
   TGPopupMenu   *fLassoMenu;          // context menu for lasso drawn
   Window_t       fTargetId;           // 
   Bool_t         fDropStatus;         // kTRUE if drop was successfull

   void           Init();

public:
   TGFrame       *InEditable(Window_t id);
   void           SelectFrame(TGFrame *frame, Bool_t add = kFALSE);
   void           GrabFrame(TGFrame *frame);
   void           UngrabFrame();
   void           DrawGrabRectangles(TGWindow *win = 0);
   void           DrawGrabRect(Int_t i, Int_t x, Int_t y);
   Bool_t         CheckDragResize(Event_t *event);
   TList         *GetFramesInside(Int_t x0, Int_t y0, Int_t x, Int_t y);
   void           ToGrid(Int_t &x, Int_t &y);
   void           DoReplace(TGFrame *frame);
   void           DeleteFrame(TGFrame *frame);
   void           HandleDelete(Bool_t crop = kFALSE);
   void           HandleReturn(Bool_t on = kFALSE);
   void           HandleAlignment(Int_t to, Bool_t lineup = kFALSE);
   void           HandleCut();
   void           HandleCopy();
   void           HandlePaste();
   void           HandleReplace();
   void           HandleGrid();
   void           CloneEditable();
   void           Save(const char *file = "");
   void           HandleLayoutOrder(Bool_t forward = kTRUE);
   void           DoResize();
   void           DoMove();
   void           DrawLasso();
   void           PlaceFrame(TGFrame*);
   void           ReparentFrames(TGFrame *newfr,
                                 TGCompositeFrame *oldfr);
   TGCompositeFrame *FindCompositeFrame(Window_t id);
   void           SetCursorType(Int_t cur);
   void           CheckTargetUnderGrab();
   void           HighlightCompositeFrame(Window_t);
   void           Compact(Bool_t global = kTRUE);
   Bool_t         StartDrag(TGFrame *src, Int_t x, Int_t y);
   Bool_t         EndDrag();
   Bool_t         Drop();
   Bool_t         Cancel(Bool_t delSrc);
   void           Menu4Frame(TGFrame *, Int_t x, Int_t y);
   void           Menu4Lasso(Int_t x, Int_t y);
   void           CreatePropertyEditor();
   void           DeletePropertyEditor();
   void           SetEditable(Bool_t on = kTRUE);
   void           DoRedraw();
   void           SwitchEditable(TGFrame *frame);

   Bool_t         HandleEvent(Event_t *);
   Bool_t         RecognizeGesture(Event_t *, TGFrame *frame = 0);
   Bool_t         HandleButtonPress(Event_t *);
   Bool_t         HandleButtonRelease(Event_t *);
   Bool_t         HandleButton(Event_t *);
   Bool_t         HandleDoubleClick(Event_t*);
   Bool_t         HandleKey(Event_t *);
   Bool_t         HandleMotion(Event_t *);
   Bool_t         HandleClientMessage(Event_t *);
   Bool_t         HandleSelection(Event_t *);
   Bool_t         HandleExpose(Event_t *);
   Bool_t         HandleConfigureNotify(Event_t *);
   Bool_t         HandleSelectionRequest(Event_t *);
   void           HandleButon3Pressed(Event_t *, TGFrame *frame = 0);

   virtual        ~TGuiBldDragManager();

public:
   TGuiBldDragManager();

   Bool_t         HandleTimer(TTimer *);
   void           HandleAction(Int_t act);

   Bool_t         IsMoveWaiting() const;
   TGFrame       *GetTarget() const { return fTarget; }
   void           Snap2Grid();
   void           SetGridStep(UInt_t step);
   UInt_t         GetGridStep();
   void           HandleUpdateSelected(TGFrame *);
   void           HideGrabRectangles();
   Bool_t         IgnoreEvent(Event_t *e);
   Int_t          GetStrartDragX() const;     
   Int_t          GetStrartDragY() const;
   Int_t          GetEndDragX() const;
   Int_t          GetEndDragY() const;
   void           ExecuteQuickAction(Event_t *);
   void           BreakLayout();
   void           SwitchLayout();
   Bool_t         GetDropStatus() const { return fDropStatus; }
   void           SetBuilder(TGuiBuilder *b) { fBuilder = b; }

   void           Selected(TGFrame *frame); //*SIGNAL*

   ClassDef(TGuiBldDragManager,0)  // drag and drop manager
};


#endif
