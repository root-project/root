// @(#)root/guibuilder:$Id$
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

#include "TGFrame.h"

#include "TVirtualDragManager.h"

class TTimer;
class TGuiBldDragManagerPimpl;
class TRootGuiBuilder;
class TQUndoManager;
class TGPopupMenu;
class TGuiBldEditor;
class TGColorDialog;
class TGFontDialog;
class TGTextButton;
class TGPictureButton;
class TGCanvas;
class TGComboBox;
class TGLabel;
class TGListBox;
class TGProgressBar;
class TGScrollBar;
class TGTextEntry;
class TGIcon;


enum EActionType {
   kNoneAct, kPropertyAct, kEditableAct, kReparentAct,
   kDropAct, kCutAct, kCopyAct, kPasteAct, kCropAct,
   kCompactAct, kCompactGlobalAct, kLayUpAct, kLayDownAct,
   kCloneAct, kSaveAct, kSaveFrameAct, kGrabAct, kDeleteAct,
   kLeftAct, kRightAct, kUpAct, kDownAct, kEndEditAct, kReplaceAct,
   kGridAct, kBreakLayoutAct, kSwitchLayoutAct, kNewAct,
   kOpenAct, kLayoutHAct, kLayoutVAct, kUndoAct, kRedoAct,
   kSelectAct, kMethodMenuAct, kToggleMenuAct
};

//////////////////////////////////////////////////////////////////////////
class TGuiBldDragManager : public TVirtualDragManager, public TGFrame {

friend class TGClient;
friend class TGFrame;
friend class TGMainFrame;
friend class TGGrabRect;
friend class TRootGuiBuilder;
friend class TGuiBldDragManagerRepeatTimer;
friend class TGuiBldMenuDialog;
friend class TGuiBldGeometryFrame;
friend class TGuiBldEditor;

private:
   TGuiBldDragManagerPimpl *fPimpl;    // private data

   TRootGuiBuilder   *fBuilder;        // pointer to gui builder
   TGuiBldEditor *fEditor;             // frame property editor
   Bool_t         fLassoDrawn;         // kTRUE if  lasso drawn
   TString        fPasteFileName;      // paste_clippboard file name
   TString        fTmpBuildFile;       // temporary file name
   Bool_t         fSelectionIsOn;      // selection with Shift key pressed
   TGPopupMenu   *fFrameMenu;          // context menu for frames
   TGPopupMenu   *fLassoMenu;          // context menu for lasso drawn
   Window_t       fTargetId;           // an id of window where drop
   Bool_t         fDropStatus;         // kTRUE if drop was successfull
   Bool_t         fStop;               // kTRUE if stopped
   TGFrame       *fSelected;           // selected frame. In most cases selected is
                                       // the same frame as grabbed frame.
   TList         *fListOfDialogs;      // list of dialog methods

   static TGColorDialog *fgGlobalColorDialog;   // color dialog
   static TGColorDialog *GetGlobalColorDialog(Bool_t create = kTRUE);

   static TGFontDialog *fgGlobalFontDialog;     // font dialog
   static TGFontDialog *GetGlobalFontDialog();  //


   void           Reset1();
   void           DrawGrabRectangles(TGWindow *win = nullptr);
   void           DrawGrabRect(Int_t i, Int_t x, Int_t y);
   TGCompositeFrame *FindLayoutFrame(TGFrame *f);
   Bool_t         IsPointVisible(Int_t x, Int_t y);
   Bool_t         IsSelectedVisible();
   void           CloseMenus();
   Bool_t         IsEditDisabled(TGWindow *f) const { return (f && (f->GetEditDisabled() & kEditDisable)); }
   Bool_t         IsGrabDisabled(TGWindow *f) const { return (f && (f->GetEditDisabled() & kEditDisableGrab)); }
   Bool_t         IsEventsDisabled(TGWindow *f) const { return (f && (f->GetEditDisabled() & kEditDisableEvents)); }
   Bool_t         IsFixedLayout(TGWindow *f) const { return (f && (f->GetEditDisabled() & kEditDisableLayout)); }
   Bool_t         IsFixedH(TGWindow *f) const { return (f && (f->GetEditDisabled() & kEditDisableHeight)); }
   Bool_t         IsFixedW(TGWindow *f) const { return (f && (f->GetEditDisabled() & kEditDisableWidth)); }
   Bool_t         IsFixedSize(TGWindow *f) const { return (f && (f->GetEditDisabled() & kEditDisableResize)); }
   Bool_t         CanChangeLayout(TGWindow *w) const;
   Bool_t         CanChangeLayoutOrder(TGWindow *w) const;
   Bool_t         CanCompact(TGWindow *w) const;

   void           ChangeSelected(TGFrame *f);
   TGFrame       *GetEditableParent(TGFrame *f);
   TGFrame       *GetMovableParent(TGWindow *p);
   TGFrame       *GetBtnEnableParent(TGFrame *fr);
   TGWindow      *GetResizableParent(TGWindow *p);
   TGFrame       *FindMdiFrame(TGFrame *in);
   void           RaiseMdiFrame(TGFrame *in);
   Bool_t         CheckTargetAtPoint(Int_t x, Int_t y);
   void           AddClassMenuMethods(TGPopupMenu *menu, TObject *object);
   void           AddDialogMethods(TGPopupMenu *menu, TObject *object);
   void           DeleteMenuDialog();
   void           CreateListOfDialogs();

private:
   TGFrame       *InEditable(Window_t id);
   void           GrabFrame(TGFrame *frame);
   void           UngrabFrame();
   void           SetPropertyEditor(TGuiBldEditor *e);
   void           DeletePropertyEditor();

   TList         *GetFramesInside(Int_t x0, Int_t y0, Int_t x, Int_t y);
   void           ToGrid(Int_t &x, Int_t &y);
   void           DoReplace(TGFrame *frame);
   void           DeleteFrame(TGFrame *frame);
   void           HandleDelete(Bool_t crop = kFALSE);
   void           HandleReturn(Bool_t on = kFALSE);
   void           HandleAlignment(Int_t to, Bool_t lineup = kFALSE);
   void           HandleCut();
   void           HandleCopy(Bool_t brk_layout = kTRUE);
   void           HandlePaste();
   void           HandleReplace();
   void           HandleGrid();
   void           CloneEditable();
   void           DropCanvas(TGCanvas *canvas);
   void           PutToCanvas(TGCompositeFrame *cont);
   Bool_t         Save(const char *file = "");
   Bool_t         SaveFrame(const char *file = nullptr);
   void           HandleLayoutOrder(Bool_t forward = kTRUE);
   void           DoResize();
   void           DoMove();
   void           DrawLasso();
   void           PlaceFrame(TGFrame*, TGLayoutHints *);
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
   void           DoRedraw();
   void           SwitchEditable(TGFrame *frame);
   void           UnmapAllPopups();
   void           BreakLayout();
   void           SwitchLayout();

   Bool_t         RecognizeGesture(Event_t *, TGFrame *frame = nullptr);
   Bool_t         HandleButtonPress(Event_t *);
   Bool_t         HandleButtonRelease(Event_t *);
   Bool_t         HandleButton(Event_t *);
   Bool_t         HandleDoubleClick(Event_t*);
   Bool_t         HandleMotion(Event_t *);
   Bool_t         HandleClientMessage(Event_t *);
   Bool_t         HandleDestroyNotify(Event_t *);
   Bool_t         HandleSelection(Event_t *);
   Bool_t         HandleExpose(Event_t *);
   Bool_t         HandleConfigureNotify(Event_t *);
   Bool_t         HandleSelectionRequest(Event_t *);
   void           HandleButon3Pressed(Event_t *, TGFrame *frame = nullptr);
   Bool_t         HandleEvent(Event_t *);
   Bool_t         HandleTimer(TTimer *);

   Bool_t         IsMoveWaiting() const;
   Bool_t         IsLassoDrawn() const { return fLassoDrawn; }
   void           SetLassoDrawn(Bool_t on);
   void           HideGrabRectangles();
   Bool_t         IgnoreEvent(Event_t *e);
   Bool_t         CheckDragResize(Event_t *event);
   Bool_t         IsPasteFrameExist();

public:
   TGuiBldDragManager();
   virtual        ~TGuiBldDragManager();

   void           HandleAction(Int_t act);
   Bool_t         HandleKey(Event_t *);

   TGFrame       *GetTarget() const { return fTarget; }
   TGFrame       *GetSelected() const;
   void           Snap2Grid();
   void           SetGridStep(UInt_t step);
   UInt_t         GetGridStep();
   void           HandleUpdateSelected(TGFrame *);
   Int_t          GetStrartDragX() const;
   Int_t          GetStrartDragY() const;
   Int_t          GetEndDragX() const;
   Int_t          GetEndDragY() const;

   Bool_t         GetDropStatus() const { return fDropStatus; }
   void           SetBuilder(TRootGuiBuilder *b) { fBuilder = b; }

   Bool_t         IsStopped() const { return fStop; }
   void           SetEditable(Bool_t on = kTRUE);
   void           SelectFrame(TGFrame *frame, Bool_t add = kFALSE);

   static void    MapGlobalDialog(TGMainFrame *dialog, TGFrame *fr);

   Bool_t         HandleTimerEvent(Event_t *ev, TTimer *t);
   void           TimerEvent(Event_t *ev)
                     { Emit("TimerEvent(Event_t*)", (Longptr_t)ev); } // *SIGNAL*

   // hadndling dynamic context menus
   void DoClassMenu(Int_t);
   void DoDialogOK();
   void DoDialogApply();
   void DoDialogCancel();

   void ChangeProperties(TGLabel *);         //*MENU* *DIALOG*icon=bld_fontselect.png*
   void ChangeProperties(TGTextButton *);    //*MENU* *DIALOG*icon=bld_fontselect.png*

   void ChangeTextFont(TGGroupFrame *);      //*MENU* *DIALOG*icon=bld_fontselect.png*
   void ChangeTextFont(TGTextEntry *);       //*MENU* *DIALOG*icon=bld_fontselect.png*

   void ChangePicture(TGPictureButton *);    //*MENU* *DIALOG*icon=bld_open.png*
   void ChangeImage(TGIcon *);               //*MENU* *DIALOG*icon=bld_open.png*

   void ChangeBarColor(TGProgressBar *);     //*MENU* *DIALOG*icon=bld_colorselect.png*

   void ChangeTextColor(TGGroupFrame *);     //*MENU* *DIALOG*icon=bld_colorselect.png*
   void ChangeTextColor(TGLabel *);          //*MENU* *DIALOG*icon=bld_colorselect.png*
   void ChangeTextColor(TGTextButton *);     //*MENU* *DIALOG*icon=bld_colorselect.png*
   void ChangeTextColor(TGProgressBar *);    //*MENU* *DIALOG*icon=bld_colorselect.png*
   void ChangeTextColor(TGTextEntry *);      //*MENU* *DIALOG*icon=bld_colorselect.png*

   void ChangeBackgroundColor(TGListBox *);  //*MENU* *DIALOG*icon=bld_colorselect.png*
   void ChangeBackgroundColor(TGCanvas *);   //*MENU* *DIALOG*icon=bld_colorselect.png*
   void ChangeBackgroundColor(TGComboBox *); //*MENU* *DIALOG*icon=bld_colorselect.png*
   void ChangeBackgroundColor(TGFrame *);             //*MENU* *DIALOG*icon=bld_colorselect.png*
   void ChangeBackgroundColor(TGCompositeFrame *);    //*MENU* *DIALOG*icon=bld_colorselect.png*

   ClassDef(TGuiBldDragManager,0)  // drag and drop manager
};


#endif
