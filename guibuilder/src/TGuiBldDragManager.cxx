// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldDragManager.cxx,v 1.31 2004/12/09 22:55:06 brun Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TGuiBldDragManager.h"
#include "TGuiBldEditor.h"
#include "TRootGuiBuilder.h"
#include "TGuiBldQuickHandler.h"

#include "TTimer.h"
#include "TList.h"
#include "TSystem.h"
#include "TMath.h"
#include "TGResourcePool.h"
#include "TROOT.h"
#include "TColor.h"
#include "KeySymbols.h"
#include "TGMenu.h"
#include "TGFileDialog.h"
#include "TGMsgBox.h"
#include "TRandom.h"
#include "TGButton.h"
#include "TGMdi.h"


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldDragManager                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


ClassImp(TGuiBldDragManager)


static UInt_t gGridStep = 10;
static TGuiBldDragManager *gGuiBldDragManager = 0;

///////////////////////// auxilary static functions ///////////////////////////
//______________________________________________________________________________
static Window_t GetWindowFromPoint(Int_t x, Int_t y)
{
   // returns a window located at point x,y (screen coordinates)

   Window_t src, dst, child;
   Window_t ret = 0;
   Int_t xx = x;
   Int_t yy = y;

   dst = src = child = gVirtualX->GetDefaultRootWindow();

   while (child && dst) {
      src = dst;
      dst = child;
      gVirtualX->TranslateCoordinates(src, dst, xx, yy, xx, yy, child);
      ret = dst;
   }
   return ret;
}


////////////////////////////////////////////////////////////////////////////////
class TGuiBldDragManagerGrid {

public:
   static UInt_t   fgStep;
   static ULong_t  fgPixel;
   static TGGC    *fgBgnd;

   Pixmap_t    fPixmap;
   TGWindow   *fWindow;
   Int_t       fWinId;

   TGuiBldDragManagerGrid();
   ~TGuiBldDragManagerGrid();
   void  Draw();
   void  SetStep(UInt_t step);
   void  InitPixmap();
   void  InitBgnd();
};

UInt_t   TGuiBldDragManagerGrid::fgStep = gGridStep;
ULong_t  TGuiBldDragManagerGrid::fgPixel = 0;
TGGC    *TGuiBldDragManagerGrid::fgBgnd = 0;

//______________________________________________________________________________
TGuiBldDragManagerGrid::TGuiBldDragManagerGrid()
{
   //

   fPixmap = 0;
   fWindow = 0;
   fWinId = 0;

   if (!fgBgnd) InitBgnd();
   SetStep(fgStep);
}

//______________________________________________________________________________
TGuiBldDragManagerGrid::~TGuiBldDragManagerGrid()
{
   //

   if (gClient && gClient->GetWindowById(fWinId)) {
      fWindow->SetBackgroundPixmap(0);
      fWindow->SetBackgroundColor(((TGFrame*)fWindow)->GetBackground());
      gClient->NeedRedraw(fWindow);
   }
   if (fPixmap) {
      gVirtualX->DeletePixmap(fPixmap);
   }

   fPixmap = 0;
   fWindow = 0;
   fWinId = 0;
}

//______________________________________________________________________________
void TGuiBldDragManagerGrid::SetStep(UInt_t step)
{
   //

   if (!gClient || !gClient->IsEditable()) return;

   fWindow = (TGWindow*)gClient->GetRoot();
   fWinId = fWindow->GetId();
   fgStep = step;
   InitPixmap();
}

//______________________________________________________________________________
void TGuiBldDragManagerGrid::InitBgnd()
{
   //

   if (fgBgnd) return;

   fgBgnd = new TGGC(TGFrame::GetBckgndGC());
   fgPixel = fgBgnd->GetForeground();

   Float_t r, g, b;
   TColor::Pixel2RGB(fgPixel, r, g, b);
   r = r * 0.9;
   g = g * 0.9;
   //b = b * 0.9;
   fgPixel = TColor::RGB2Pixel(r, g, b);
   fgBgnd->SetForeground(fgPixel);
}

//______________________________________________________________________________
void TGuiBldDragManagerGrid::InitPixmap()
{
   //

   if (fPixmap) {
      gVirtualX->DeletePixmap(fPixmap);
   }

   fPixmap = gVirtualX->CreatePixmap(gClient->GetDefaultRoot()->GetId(), fgStep, fgStep);
   gVirtualX->FillRectangle(fPixmap, fgBgnd->GetGC(), 0, 0, fgStep, fgStep);

   if(fgStep > 2) {
      gVirtualX->FillRectangle(fPixmap, TGFrame::GetShadowGC()(),
                               fgStep - 1, fgStep - 1, 1, 1);
   }
}

//______________________________________________________________________________
void TGuiBldDragManagerGrid::Draw()
{
   //

   if (fWindow && (fWindow != gClient->GetRoot())) {
      fWindow->SetBackgroundPixmap(0);
      fWindow->SetBackgroundColor(((TGFrame*)fWindow)->GetBackground());
      gClient->NeedRedraw(fWindow);
   }

   if (!gClient || !gClient->IsEditable()) return;

   if (!fPixmap) InitPixmap();

   fWindow = (TGWindow*)gClient->GetRoot();
   fWinId = fWindow->GetId();
   fWindow->SetBackgroundPixmap(fPixmap);

   gClient->NeedRedraw(fWindow);
}


////////////////////////////////////////////////////////////////////////////////
class TGuiBldDragManagerRepeatTimer : public TTimer {

private:
   TGuiBldDragManager *fManager;   // back pointer

public:
   TGuiBldDragManagerRepeatTimer(TGuiBldDragManager *m, Long_t ms) :
                                 TTimer(ms, kTRUE) { fManager = m; }
   Bool_t Notify() { fManager->HandleTimer(this); Reset(); return kFALSE; }
};


////////////////////////////////////////////////////////////////////////////////
class TGDragGrabber : public TGFrame {

private:
   TGFrame    *fFrame;     //
   Pixmap_t    fPixmap;    //
   GContext_t  fGC;        //

public:
   TGDragGrabber(TGFrame *src, Int_t x, Int_t y);
   void DoRedraw();
};

//______________________________________________________________________________
TGDragGrabber::TGDragGrabber(TGFrame *src, Int_t x, Int_t y) :
               TGFrame(gClient->GetDefaultRoot(), 20, 20)
{
   //

   fFrame = src;

   fGC = gClient->GetResourcePool()->GetFrameGC()->GetGC();
   Resize(fFrame->GetWidth(), fFrame->GetHeight());
   fPixmap = gVirtualX->CreatePixmap(gVirtualX->GetDefaultRootWindow(), fWidth, fHeight);
   gVirtualX->CopyArea(fFrame->GetId(), fPixmap, fGC, 0, 0, fWidth, fHeight, 0, 0);
   gVirtualX->SetMWMHints(fId, 0, 0, 0);

   Move(x, y);
}

//______________________________________________________________________________
void TGDragGrabber::DoRedraw()
{
   //

   gVirtualX->CopyArea(fPixmap, fId, fGC, 0, 0, fWidth, fHeight, 0, 0);
}


////////////////////////////////////////////////////////////////////////////////
class TGGrabRect : public TGFrame {

private:
   Pixmap_t    fPixmap;
   ECursor     fType;

public:
   TGGrabRect(Int_t type);
   ~TGGrabRect() {}

   Bool_t HandleButton(Event_t *ev);
   ECursor GetType() const { return fType; }
};

//______________________________________________________________________________
TGGrabRect::TGGrabRect(Int_t type) : TGFrame(gClient->GetDefaultRoot(), 7, 7, kTempFrame)
{
   //

   switch (type) {
      case 0:
         fType = kTopLeft;
         break;
      case 1:
         fType = kTopSide;
         break;
      case 2:
         fType = kTopRight;
         break;
      case 3:
         fType = kBottomLeft;
         break;
      case 4:
         fType = kLeftSide;
         break;
      case 5:
         fType = kRightSide;
         break;
      case 6:
         fType = kBottomSide;
         break;
      case 7:
         fType = kBottomRight;
         break;
   }

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);

   fPixmap = gVirtualX->CreatePixmap(gVirtualX->GetDefaultRootWindow(), 7, 7);
   const TGGC *bgc = fClient->GetResourcePool()->GetSelectedBckgndGC();
   const TGGC *gc = fClient->GetResourcePool()->GetSelectedGC();

   gVirtualX->FillRectangle(fPixmap, bgc->GetGC(), 0, 0, 6, 6);
   gVirtualX->DrawRectangle(fPixmap, gc->GetGC(), 0, 0, 6, 6);

   AddInput(kButtonPressMask);
   SetBackgroundPixmap(fPixmap);

   gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(fType));
}

//______________________________________________________________________________
Bool_t TGGrabRect::HandleButton(Event_t *ev)
{
   //

   gGuiBldDragManager->CheckDragResize(ev);
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
class TGAroundFrame : public TGFrame {

public:
   TGAroundFrame();
   ~TGAroundFrame() {}
};

//______________________________________________________________________________
TGAroundFrame::TGAroundFrame() : TGFrame(gClient->GetDefaultRoot(), 1, 1,
                                         kTempFrame | kOwnBackground)
{
   //

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);
   ULong_t red;
   fClient->GetColorByName("red", red);
   SetBackgroundColor(red);
}


////////////////////////////////////////////////////////////////////////////////
class TGuiBldDragManagerPimpl {

friend class TGuiBldDragManager;

private:
   TGuiBldDragManager     *fManager;   // dnd manager
   TTimer            *fRepeatTimer;    // repeat rate timer (when mouse stays pressed)
   TGFrame           *fGrab;           //
   TGLayoutHints     *fGrabLayout;
   TGFrame           *fSaveGrab;       // used during context menu handling
   TGFrame           *fLastFrame;       // last clicked frame
   TGuiBldDragManagerGrid *fGrid;           //
   ECursor            fResizeType;     //
   Int_t              fX0, fY0;        // initial drag position in pixels
   Int_t              fX, fY;          // current drag position in pixels
   Int_t              fXf, fYf;        // offset of inititial position inside frame
   Int_t              fGrabX, fGrabY;  //
   const TGWindow    *fGrabParent;     //
   Int_t              fLastPopupAction;//
   Bool_t             fReplaceOn;
   TGGrabRect        *fGrabRect[8];    //
   TGFrame           *fAroundFrame[4]; //
   Bool_t             fGrabRectHidden;
   TGFrameElement    *fGrabListPosition;
   Bool_t             fButtonPressed;
   Bool_t             fCompacted;
   TGFrame           *fPlane;          //

public:
   TGuiBldDragManagerPimpl(TGuiBldDragManager *m) {
      fManager = m;
      fRepeatTimer = new TGuiBldDragManagerRepeatTimer(m, 100);

      int i = 0;
      for (i = 0; i <8; i++) {
         fGrabRect[i] = new TGGrabRect(i);
      }
      for (i = 0; i <4; i++) {
         fAroundFrame[i] = new TGAroundFrame();
      }

      fPlane = 0;
      ResetParams();
   }
   void ResetParams() {
      fGrab = 0;
      fSaveGrab = 0;
      fLastFrame = 0;
      fGrid = 0;
      fX0 = fY0 = fX = fY = fXf = fYf = fGrabX = fGrabY = 0;
      fGrabParent = 0;
      fResizeType = kPointer;
      fLastPopupAction = kNoneAct;
      fReplaceOn = kFALSE;
      fGrabLayout = 0;
      fGrabRectHidden = kFALSE;
      fGrabListPosition = 0;
      fButtonPressed = kFALSE;
      fCompacted = kFALSE;
   }

   ~TGuiBldDragManagerPimpl() {
      int i;
      for (i = 0; i <8; i++) {
         delete fGrabRect[i];
      }
      for (i = 0; i <4; i++) {
         delete fAroundFrame[i];
      }

      delete fRepeatTimer;
      delete fGrab;

      if (fPlane) {
         fPlane->ChangeOptions(fPlane->GetOptions() & ~kRaisedFrame);
         gClient->NeedRedraw(fPlane, kTRUE);
         fPlane = 0;
      }
   }
};


////////////////////////////////////////////////////////////////////////////////
TGuiBldDragManager::TGuiBldDragManager() : TVirtualDragManager() ,
                    TGFrame(gClient->GetDefaultRoot(), 1, 1)
{
   // Constructor. Create "fantom window".

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);

   gGuiBldDragManager = this;
   fPimpl = new TGuiBldDragManagerPimpl(this);

   Init();
   fSelectionIsOn = kFALSE;
   fFrameMenu     = 0;
   fLassoMenu     = 0;
   fEditor        = 0;
   fBuilder       = 0;
   fQuickHandler  = 0;
   fLassoDrawn    = kFALSE;
   fDropStatus    = kFALSE;

   TString tmpfile = gSystem->TempDirectory();
   fPasteFileName = gSystem->ConcatFileName(tmpfile.Data(),
                             Form("RootGuiBldClipboard%d.C", gSystem->GetPid()));

   fName = "Gui Builder Drag Manager";
   SetWindowName(fName.Data());

   fClient->UnregisterWindow(this);
}

//______________________________________________________________________________
TGuiBldDragManager::~TGuiBldDragManager()
{
   // Destructor

   delete fPimpl;

   delete fBuilder;
   fBuilder = 0;

//   delete fEditor;
//   fEditor = 0;

   delete fFrameMenu;
   fFrameMenu =0;

   delete fLassoMenu;
   fLassoMenu = 0;

   if (!gSystem->AccessPathName(fPasteFileName.Data())) {
      gSystem->Unlink(fPasteFileName.Data());
   }

   gGuiBldDragManager = 0;
}

//______________________________________________________________________________
void TGuiBldDragManager::Init()
{
   //

   TVirtualDragManager::Init();
   fTargetId = 0;
   SetCursorType(kPointer);
}

//______________________________________________________________________________
void TGuiBldDragManager::Snap2Grid()
{
   //

   delete fPimpl->fGrid;
   fPimpl->fGrid = new TGuiBldDragManagerGrid();
   fPimpl->fGrid->Draw();
}

//______________________________________________________________________________
UInt_t TGuiBldDragManager::GetGridStep()
{
   //

   return fPimpl->fGrid ? fPimpl->fGrid->fgStep : 1;
}

//______________________________________________________________________________
void TGuiBldDragManager::SetGridStep(UInt_t step)
{
   //

   fPimpl->fGrid->SetStep(step);
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::IgnoreEvent(Event_t *event)
{
   //

   if (!fClient || !fClient->IsEditable()) return kTRUE;
   if (event->fType == kClientMessage) return kFALSE;

   const TGWindow *w = fClient->GetWindowById(event->fWindow);
   return w ? w->IsEditDisabled() : kTRUE;
}

//______________________________________________________________________________
TGFrame* TGuiBldDragManager::InEditable(Window_t id)
{
   //

   if (!id) return 0;

   Window_t preparent = id;
   Window_t parent = (Window_t)gVirtualX->GetParent(id);

   while (!parent || (parent != fClient->GetDefaultRoot()->GetId())) {
      if (parent == fClient->GetRoot()->GetId()) {
         TGWindow *w = fClient->GetWindowById(preparent);
         return (w ? (TGFrame*)w : 0);
      }
      preparent = parent;
      parent = gVirtualX->GetParent(parent);
   }
   return 0;
}

//______________________________________________________________________________
TGCompositeFrame *TGuiBldDragManager::FindCompositeFrame(Window_t id)
{
   //

   if (!id) return 0;

   Window_t parent = id;

   while (!parent || (parent != fClient->GetDefaultRoot()->GetId())) {
      TGWindow *w = fClient->GetWindowById(parent);
      if (w) {
         if (w->InheritsFrom(TGCompositeFrame::Class())) {
            return (TGCompositeFrame*)w;
         }
      }
      parent = gVirtualX->GetParent(parent);
   }
   return 0;
}

//______________________________________________________________________________
void TGuiBldDragManager::SetCursorType(Int_t cur)
{
   // sets cursor

   static UInt_t gid = 0;
   static UInt_t rid = 0;

   if (fPimpl->fGrab && (gid != fPimpl->fGrab->GetId())) {
      gVirtualX->SetCursor(fPimpl->fGrab->GetId(),
                           gVirtualX->CreateCursor((ECursor)cur));
      gid = fPimpl->fGrab->GetId();
   }
   if (fClient->IsEditable() && (rid != fClient->GetRoot()->GetId())) {
      gVirtualX->SetCursor(fClient->GetRoot()->GetId(),
                           gVirtualX->CreateCursor((ECursor)cur));
      rid = fClient->GetRoot()->GetId();
   }
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::CheckDragResize(Event_t *event)
{
   // check resize type event

   Bool_t ret = kFALSE;
   fPimpl->fResizeType = kPointer;

   for (int i = 0; i < 8; i++) {
      if (fPimpl->fGrabRect[i]->GetId() == event->fWindow) {
         fPimpl->fResizeType = fPimpl->fGrabRect[i]->GetType();
         ret = kTRUE;
      }
   }

   if ((event->fType == kButtonPress) && (fPimpl->fResizeType != kPointer)) {
      fDragType = kDragResize;
      ret = kTRUE;
   }

   SetCursorType(ret ? fPimpl->fResizeType : kPointer);
   return ret;
}

//______________________________________________________________________________
void TGuiBldDragManager::DoRedraw()
{
   //

   if (!fClient || !fClient->IsEditable()) return;

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   fClient->NeedRedraw(root, kTRUE);
   fLassoDrawn = kFALSE;
}

//______________________________________________________________________________
void TGuiBldDragManager::SwitchEditable(TGFrame *frame)
{
   //

   if (!frame) return;
   TGCompositeFrame *comp = 0;

   if (frame->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame *)frame;
   } else if (frame->GetParent()->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame *)frame->GetParent();
   }

   if (!comp) return;

   comp->SetEditable(kTRUE);

   if (fBuilder) {
      TString str = comp->ClassName();
      str += "::";
      str += comp->GetName();
      str += " set editable";
      fBuilder->UpdateStatusBar(str.Data());
   }

   if (frame != comp) {
      SelectFrame(frame);
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::SelectFrame(TGFrame *frame, Bool_t add)
{
   //

   if (!frame || frame->IsEditable()) return;

   static Int_t x, x0, y, y0, xx, yy;
   Window_t c;

   frame->MapRaised();

   if (!add) {
      fDragType = kDragMove;
      gVirtualX->TranslateCoordinates(frame->GetId(),
                                      fClient->GetDefaultRoot()->GetId(),
                                      0, 0, x0, y0, c);
      x = x0 + frame->GetWidth();
      y = y0 + frame->GetHeight();

      Selected(frame);
      DrawGrabRectangles(frame);

      if (fBuilder) {
         TString str = frame->ClassName();
         str += "::";
         str += frame->GetName();
         str += " selected";
         fBuilder->UpdateStatusBar(str.Data());
      }

   } else {
      gVirtualX->TranslateCoordinates(frame->GetId(),
                                      fClient->GetDefaultRoot()->GetId(),
                                      0, 0, xx, yy, c);
      fDragType = kDragLasso;
      fPimpl->fX0 = x0 = TMath::Min(x0, xx);
      fPimpl->fX = x = TMath::Max(x, xx + (Int_t)frame->GetWidth());
      fPimpl->fY0 = y0 = TMath::Min(y0, yy);
      fPimpl->fY = y = TMath::Max(y, yy + (Int_t)frame->GetHeight());

      DrawLasso();
   }

   fFrameUnder = fPimpl->fGrab = frame;
   SetCursorType(kMove);

   if (fBuilder) fBuilder->Update();
}

//______________________________________________________________________________
void TGuiBldDragManager::Selected(TGFrame *frame)
{
   //

   if (!frame || (fPimpl->fGrab == frame)) return;

   Emit("Selected(TGFrame*)", (Long_t)frame);
}

//______________________________________________________________________________
void TGuiBldDragManager::GrabFrame(TGFrame *frame)
{
   //

   if (!frame) return;

   fPimpl->fGrabParent = frame->GetParent();
   fPimpl->fGrabX = frame->GetX();
   fPimpl->fGrabY = frame->GetY();

   Window_t c;
   gVirtualX->TranslateCoordinates(frame->GetId(),
                                   fClient->GetDefaultRoot()->GetId(),
                                   0, 0, fPimpl->fX0, fPimpl->fY0, c);
   fPimpl->fX = fPimpl->fX0;
   fPimpl->fY = fPimpl->fY0;

   if (frame->GetFrameElement() && frame->GetFrameElement()->fLayout) {
      fPimpl->fGrabLayout = frame->GetFrameElement()->fLayout;
   }

   if (fPimpl->fGrabParent && frame->GetFrameElement() &&
       fPimpl->fGrabParent->InheritsFrom(TGCompositeFrame::Class())) {
      TList *li = ((TGCompositeFrame*)fPimpl->fGrabParent)->GetList();
      fPimpl->fGrabListPosition = (TGFrameElement*)li->Before(frame->GetFrameElement());
      ((TGCompositeFrame*)fPimpl->fGrabParent)->RemoveFrame(frame);
   }

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(frame->GetId(), &attr);
   frame->ReparentWindow(fClient->GetDefaultRoot(), fPimpl->fX0, fPimpl->fY0);
   gVirtualX->Update(1);

   if (fBuilder) {
      fBuilder->Update();
      TString str = frame->ClassName();
      str += "::";
      str += frame->GetName();
      str += " is being dragged";
      fBuilder->UpdateStatusBar(str.Data());
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::UngrabFrame()
{
   //

   if (!fPimpl->fGrab) return;

   SetCursorType(kPointer);
   HideGrabRectangles();

   DoRedraw();

   if (fBuilder) {
      fBuilder->Update();
      TString str = fPimpl->fGrab->ClassName();
      str += "::";
      str += fPimpl->fGrab->GetName();
      str += " ungrabbed";
      fBuilder->UpdateStatusBar(str.Data());
   }
   fPimpl->fGrab = 0;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::IsSelectedVisible()
{
   //

   if (!fPimpl->fGrab) return kFALSE;

   Window_t w = gVirtualX->GetDefaultRootWindow();
   Window_t src, dst, child;
   Int_t x, y;
   Bool_t ret = kFALSE;

   gVirtualX->TranslateCoordinates(fPimpl->fGrab->GetId(), w,
                                   fPimpl->fGrab->GetWidth()/2,
                                   fPimpl->fGrab->GetHeight()/2, x, y, child);
   dst = src = child = w;

   while (child) {
      src = dst;
      dst = child;
      gVirtualX->TranslateCoordinates(src, dst, x, y, x, y, child);
      if (child == fPimpl->fGrab->GetId()) {
         ret = kTRUE;
         break;
      }
   }
   return ret;
}

//______________________________________________________________________________
void TGuiBldDragManager::DrawGrabRectangles(TGWindow *win)
{
   //

   TGFrame *frame = win ? (TGFrame *)win : fPimpl->fGrab;

   if (!frame) return;

   Window_t w = gVirtualX->GetDefaultRootWindow();
   Window_t c; Int_t x, y;

   gVirtualX->TranslateCoordinates(frame->GetId(), w,  0, 0, x, y, c);

   if (frame->InheritsFrom(TGCompositeFrame::Class()) && !frame->IsLayoutBroken()) {
      fPimpl->fAroundFrame[0]->MoveResize(x-3, y-3, frame->GetWidth()+6, 2);
      fPimpl->fAroundFrame[0]->MapRaised();
      fPimpl->fAroundFrame[1]->MoveResize(x+frame->GetWidth()+3, y-3, 2, frame->GetHeight()+6);
      fPimpl->fAroundFrame[1]->MapRaised();
      fPimpl->fAroundFrame[2]->MoveResize(x-3, y+frame->GetHeight()+2, frame->GetWidth()+6, 2);
      fPimpl->fAroundFrame[2]->MapRaised();
      fPimpl->fAroundFrame[3]->MoveResize(x-3, y-3, 2, frame->GetHeight()+6);
      fPimpl->fAroundFrame[3]->MapRaised();
   } else {
      for (int i=0; i<4; i++) fPimpl->fAroundFrame[i]->UnmapWindow();
   }

   // draw rectangles
   DrawGrabRect(0, x - 6, y - 6);
   DrawGrabRect(1, x + frame->GetWidth()/2 - 3, y - 6);
   DrawGrabRect(2, x + frame->GetWidth(), y - 6);
   DrawGrabRect(3, x - 6, y + frame->GetHeight());
   DrawGrabRect(4, x - 6, y + frame->GetHeight()/2 - 3);
   DrawGrabRect(5, x + frame->GetWidth(), y + frame->GetHeight()/2 - 3);
   DrawGrabRect(6, x + frame->GetWidth()/2 - 3, y + frame->GetHeight());
   DrawGrabRect(7, x + frame->GetWidth(), y + frame->GetHeight());

   fPimpl->fGrabRectHidden = kFALSE;
}

//______________________________________________________________________________
void TGuiBldDragManager::DrawGrabRect(Int_t i, Int_t x, Int_t y)
{
   //

   fPimpl->fGrabRect[i]->Move(x, y);
   fPimpl->fGrabRect[i]->MapRaised();
}

//______________________________________________________________________________
void TGuiBldDragManager::HighlightCompositeFrame(Window_t win)
{
   //

   static Window_t gw = 0;

   if (!win || (win == gw)) return;

   TGWindow *w = fClient->GetWindowById(win);

   if (!w || (w == fPimpl->fPlane) || w->IsEditDisabled() || w->IsEditable() ||
       !w->InheritsFrom(TGCompositeFrame::Class())) return;

   TGFrame *frame = (TGFrame*)w;
   UInt_t opt = frame->GetOptions();

   if ((opt & kRaisedFrame) || (opt & kSunkenFrame)) return;

   gw = win;
   if (fPimpl->fPlane) {
      fPimpl->fPlane->ChangeOptions(fPimpl->fPlane->GetOptions() & ~kRaisedFrame);
      fClient->NeedRedraw(fPimpl->fPlane, kTRUE);
   }
   fPimpl->fPlane = frame;
   fPimpl->fPlane->ChangeOptions(opt | kRaisedFrame);
   fClient->NeedRedraw(fPimpl->fPlane, kTRUE);

   if (fBuilder) {
      TString str = frame->ClassName();
      str += "::";
      str += frame->GetName();
      fBuilder->UpdateStatusBar(str.Data());
   }
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleTimer(TTimer *t)
{
   // Handle repeat timer while dragging. Every time timer times out we move
   // dragged frame set to another position.

   // reset everything
   if (!fClient || !fClient->IsEditable()) {
      if (fPimpl->fRepeatTimer) fPimpl->fRepeatTimer->Remove();

      Init();
      if (fPimpl) fPimpl->ResetParams();
      if (fBuilder) {
         fBuilder->SetAction(0);
      } else {
         DeletePropertyEditor();
      }
      return kFALSE;
   }

   if (!IsSelectedVisible()) UngrabFrame();

   Window_t dum;
   Event_t ev;
   ev.fCode = kButton1;
   ev.fType = kMotionNotify;
   ev.fState = 0;

   gVirtualX->QueryPointer(gVirtualX->GetDefaultRootWindow(), dum, dum,
                           ev.fXRoot, ev.fYRoot, ev.fX, ev.fY, ev.fState);

   static Int_t gy = 0;
   static Int_t gx = 0;
   static UInt_t gstate = 0;
   static Window_t gw = 0;

   ev.fWindow = GetWindowFromPoint(ev.fXRoot, ev.fYRoot);

   if (ev.fWindow && (gw == ev.fWindow) && (gstate == ev.fState) &&
       (ev.fYRoot == gy) && (ev.fXRoot == gx)) {
      return kFALSE;
   }

   gw = ev.fWindow;
   gstate = ev.fState;

   if (!fDragging && !fMoveWaiting && !fPimpl->fButtonPressed &&
       ((ev.fState == kButton1Mask) || (ev.fState == kButton3Mask) ||
        (ev.fState == (kButton1Mask | kKeyShiftMask)) ||
        (ev.fState == (kButton1Mask | kKeyControlMask)))) {

      if (ev.fState & kButton1Mask) ev.fCode = kButton1;
      if (ev.fState & kButton3Mask) ev.fCode = kButton3;

      ev.fType = kButtonPress;
      t->SetTime(40);

      if (fPimpl->fPlane && fClient->GetWindowById(fPimpl->fPlane->GetId())) {
         fPimpl->fPlane->ChangeOptions(fPimpl->fPlane->GetOptions() & ~kRaisedFrame);
         fClient->NeedRedraw(fPimpl->fPlane, kTRUE);
      } else {
         fPimpl->fPlane = 0;
      }

      return HandleButtonPress(&ev);
   }

   if ((fDragging || fMoveWaiting) && (!ev.fState || (ev.fState == kKeyShiftMask)) &&
       fPimpl->fButtonPressed) {

      ev.fType = kButtonRelease;
      t->SetTime(100);

      return HandleButtonRelease(&ev);
   }

   fPimpl->fButtonPressed = (ev.fState & kButton1Mask) ||
                            (ev.fState & kButton2Mask) ||
                            (ev.fState & kButton3Mask);

   if ((ev.fYRoot == gy) && (ev.fXRoot == gx)) return kFALSE;

   gy = ev.fYRoot;
   gx = ev.fXRoot;

   if (!fMoveWaiting && !fDragging && !ev.fState) {
      if (!CheckDragResize(&ev) && fClient->GetWindowById(ev.fWindow)) {
         HighlightCompositeFrame(ev.fWindow);
      }
   } else if (ev.fState & kButton1Mask) {
      HandleMotion(&ev);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::RecognizeGesture(Event_t *event, TGFrame *frame)
{
   //

   UInt_t estate = event->fState & 0xFF;

   if (((event->fCode != kButton1) && (event->fCode != kButton3)) || !frame ||
       (estate && !((estate & kKeyShiftMask) || (estate & kKeyControlMask)))) {
      return kFALSE;
   }

   if (event->fCode == kButton3) {
      SelectFrame(frame);
      HandleButon3Pressed(event, frame);
      return kTRUE;
   } else if (((event->fCode == kButton1) && (event->fState & kKeyControlMask)) ||
             ((event->fCode == kButton1) && fBuilder && fBuilder->IsSelectMode())) {
      SwitchEditable(frame);
      return kTRUE;
   } else if (!fSelectionIsOn) {
      fPimpl->fX0 = event->fXRoot;
      fPimpl->fY0 = event->fYRoot;
   }

   if (fPimpl->fLastFrame != frame) {
      fPimpl->fLastFrame = frame;
      if (fEditor && fPimpl->fLastFrame) {
         if (!fEditor->IsEmbedded()) {
            TGMainFrame *m = (TGMainFrame *)fEditor->GetMainFrame();
            m->MapRaised();
         }
         fEditor->ChangeSelected(fPimpl->fLastFrame);
      }
   }

   TGFrame *base = InEditable(frame->GetId());

   HideGrabRectangles();

   if (fBuilder && fBuilder->IsExecutalble() &&
       frame->InheritsFrom(TGCompositeFrame::Class()) &&
       !frame->IsEditable()) {

      UngrabFrame();
      frame->SetEditable(kTRUE);
      fSource = 0;
      fDragType = kDragLasso;
      goto out;
   }

   if (event->fState & kKeyShiftMask) {
      if (!fSelectionIsOn) {
         fSelectionIsOn = kTRUE;
      } else {
         fPimpl->fX = event->fXRoot;
         fPimpl->fY = event->fYRoot;
         fDragType = kDragLasso;
         DrawLasso();
         return kTRUE;
      }
   }

   if (frame->IsEditable()) {
      fSource = 0;
      CheckDragResize(event);

      if (fDragType != kDragResize) {
         fDragType = kDragLasso;
      }
   } else if (base) {
      fSource = base;
      SelectFrame(base, event->fState & kKeyShiftMask);
   } else {
      SwitchEditable(frame);
      return kTRUE;
   }

   if (fDragType == kDragNone) return kFALSE;

out:
   Window_t c;
   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   base ? base->GetId() : frame->GetId(),
                                   event->fXRoot, event->fYRoot,
                                   fPimpl->fXf, fPimpl->fYf, c);
   fPimpl->fX = event->fXRoot;
   fPimpl->fY = event->fYRoot;

   fMoveWaiting = kTRUE;
   DoRedraw();

   return kTRUE;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleButon3Pressed(Event_t *event, TGFrame *frame)
{
   //

   if (frame == fPimpl->fGrab) {
      Menu4Frame(frame, event->fXRoot, event->fYRoot);
   } else if (frame->IsEditable())  {
      if (fLassoDrawn) {
         Menu4Lasso(event->fXRoot, event->fYRoot);
      } else {
         Menu4Frame(frame, event->fXRoot, event->fYRoot);
      }
   } else {
      TGFrame *base = InEditable(frame->GetId());
      if (base) {
         //SelectFrame(base);
         Menu4Frame(base, event->fXRoot, event->fYRoot);
      } else {
         Menu4Frame(frame, event->fXRoot, event->fYRoot);
      }
   }
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleButton(Event_t *event)
{
   // handle button event occured in some ROOT frame

   if (event->fCode != kButton3) CloseMenus();

   if (event->fType == kButtonPress) return HandleButtonPress(event);
   else return HandleButtonRelease(event);
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleConfigureNotify(Event_t *event)
{
   // resize events

   TGWindow *w = fClient->GetWindowById(event->fWindow);

   if (!w) return kFALSE;

   fPimpl->fCompacted = kFALSE;
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleExpose(Event_t *event)
{
   // repaint events

   static Long_t was = gSystem->Now();
   static Window_t win = 0;
   Long_t now = (long)gSystem->Now();

   if (event->fCount || (win == event->fWindow) || (now-was < 40) ||
       fDragging) {
      if (fDragging) HideGrabRectangles();
      return kFALSE;
   }

   DrawGrabRectangles();
   win = event->fWindow;
   was = now;

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleEvent(Event_t *event)
{
   // Handle all events.

   if (IgnoreEvent(event)) return kFALSE;

   switch (event->fType) {

      case kExpose:
         return HandleExpose(event);

      case kConfigureNotify:
         while (gVirtualX->CheckEvent(fId, kConfigureNotify, *event))
            ;
         return HandleConfigureNotify(event);

      case kGKeyPress:
      case kKeyRelease:
         return HandleKey(event);

      case kFocusIn:
      case kFocusOut:
         //HandleFocusChange(event);
         break;

      case kButtonPress:
         {
            Int_t dbl_clk = kFALSE;

            static Window_t gDbw = 0;
            static Long_t gLastClick = 0;
            static UInt_t gLastButton = 0;
            static Int_t gDbx = 0;
            static Int_t gDby = 0;

            if ((event->fTime - gLastClick < 350) &&
                (event->fCode == gLastButton) &&
                (TMath::Abs(event->fXRoot - gDbx) < 6) &&
                (TMath::Abs(event->fYRoot - gDby) < 6) &&
                (event->fWindow == gDbw)) {
               dbl_clk = kTRUE;
            }

            if (dbl_clk) {
               if (event->fState & kKeyControlMask) {
                  TGWindow *root = (TGWindow *)fClient->GetRoot();
                  root->SetEditable(kFALSE);
                  SetEditable(kFALSE);
                  if (fBuilder) {
                     fBuilder->UpdateStatusBar("Edit is OFF");
                  }
                  return kTRUE;
               } else if (!(event->fState & 0xFF)) {
                  ExecuteQuickAction(event);
                  return kTRUE;
               }
            } else {
               gDbw = event->fWindow;
               gLastClick = event->fTime;
               gLastButton = event->fCode;
               gDbx = event->fXRoot;
               gDby = event->fYRoot;

               return HandleButtonPress(event);
            }

            return kFALSE;
         }

      case kButtonRelease:
         return HandleButtonRelease(event);

      case kEnterNotify:
      case kLeaveNotify:
         //HandleCrossing(event);
         break;

      case kMotionNotify:
         while (gVirtualX->CheckEvent(fId, kMotionNotify, *event))
            ;
         return HandleMotion(event);

      case kClientMessage:
         HandleClientMessage(event);
         break;

      case kSelectionNotify:
         //HandleSelection(event);
         break;

      case kSelectionRequest:
         //HandleSelectionRequest(event);
         break;

      case kSelectionClear:
         //HandleSelectionClear(event);
         break;

      case kColormapNotify:
         //HandleColormapChange(event);
         break;

      default:
         //Warning("HandleEvent", "unknown event (%#x) for (%#x)", event->fType, fId);
         break;
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleDoubleClick(Event_t *)
{
   //

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleButtonPress(Event_t *event)
{
   // handle button press event

   fPimpl->fButtonPressed = kTRUE;

   if (fPimpl->fPlane) {
      fPimpl->fPlane->ChangeOptions(fPimpl->fPlane->GetOptions() & ~kRaisedFrame);
      fClient->NeedRedraw(fPimpl->fPlane, kTRUE);
   }

   if ( ((event->fCode != kButton1) && (event->fCode != kButton3)) ||
        (event->fType != kButtonPress) || IgnoreEvent(event) ) return kFALSE;

   Init();
   HideGrabRectangles();

   Window_t w = GetWindowFromPoint(event->fXRoot, event->fYRoot);
   TGFrame *fr = 0;

   if (w) {
      fr = (TGFrame*)fClient->GetWindowById(w);
   } else {
      return kFALSE;
   }

   return RecognizeGesture(event, fr);
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleButtonRelease(Event_t *event)
{
   // handle button release event

   fPimpl->fButtonPressed = kFALSE;
   gVirtualX->SetCursor(fClient->GetRoot()->GetId(), gVirtualX->CreateCursor(kPointer));

   EndDrag();
   fSelectionIsOn &= (event->fState & kKeyShiftMask);

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleKey(Event_t *event)
{
   // handle key event


   static const char *gSaveMacroTypes[] = { "Macro files", "*.C",
                                            "All files",   "*",
                                            0,             0 };
   char   tmp[10];
   UInt_t keysym;
   Bool_t ret = kFALSE;
   TGFileInfo fi;
   static TString dir(".");
   static Bool_t overwr = kFALSE;
   const char *fname;

   if (event->fType != kGKeyPress) return kFALSE;

   if (fPimpl->fPlane) {
      fPimpl->fPlane->ChangeOptions(fPimpl->fPlane->GetOptions() & ~kRaisedFrame);
      fClient->NeedRedraw(fPimpl->fPlane, kTRUE);
   }

   CloseMenus();

   fi.fFileTypes = gSaveMacroTypes;
   fi.fIniDir    = StrDup(dir);
   fi.fOverwrite = overwr;

   gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);

   if (event->fState & kKeyControlMask) {

      switch ((EKeySym)keysym & ~0x20) {
         case kKey_Return:
         case kKey_Enter:
            HandleReturn(kTRUE);
            ret = kTRUE;
            break;
         case kKey_X:
            HandleCut();
            ret = kTRUE;
            break;
         case kKey_C:
            HandleCopy();
            ret = kTRUE;
            break;
         case kKey_V:
            if (fPimpl->fLastFrame && !fPimpl->fLastFrame->IsEditable()) {
               fPimpl->fLastFrame->SetEditable(kTRUE);
            }
            HandlePaste();
            ret = kTRUE;
            break;
         case kKey_B:
         {
            if (fPimpl->fGrab && (fPimpl->fLastFrame != fClient->GetRoot())) {
               if (fPimpl->fGrab->InheritsFrom(TGCompositeFrame::Class())) {
                  BreakLayout();
               }
            }
            ret = kTRUE;
            break;
         }
         case kKey_L:
         {
            if (fPimpl->fGrab && (fPimpl->fLastFrame != fClient->GetRoot())) {
               Compact(kFALSE);
            } else {
               Compact(kTRUE);
            }
            ret = kTRUE;
            break;
         }
         case kKey_R:
            HandleReplace();
            ret = kTRUE;
            break;
         case kKey_S:
            Save();
            ret = kTRUE;
            break;
         case kKey_G:
            HandleGrid();
            ret = kTRUE;
            break;
         case kKey_H:
            SwitchLayout();
            ret = kTRUE;
            break;
         case kKey_N:
            if (fBuilder) {
               fBuilder->NewProject();
            } else {
               TGMainFrame *main = new TGMainFrame(fClient->GetDefaultRoot(), 300, 300);
               main->MapRaised();
               main->SetEditable(kTRUE);
            }
            ret = kTRUE;
            break;
         case kKey_O:
            if (fBuilder) {
               fBuilder->NewProject();
            } else {
               TGMainFrame *main = new TGMainFrame(fClient->GetDefaultRoot(), 300, 300);
               main->MapRaised();
               main->SetEditable(kTRUE);
            }
            new TGFileDialog(fClient->GetDefaultRoot(), this, kFDSave, &fi);

            if (!fi.fFilename) return kTRUE;
            dir = fi.fIniDir;
            overwr = fi.fOverwrite;
            fname = gSystem->BaseName(gSystem->UnixPathName(fi.fFilename));

            if (strstr(fname, ".C")) {
               gROOT->Macro(fname);
            } else {
               Int_t retval;
               new TGMsgBox(fClient->GetDefaultRoot(), this, "Error...",
                            Form("file (%s) must have extension .C", fname),
                            kMBIconExclamation, kMBRetry | kMBCancel, &retval);
               if (retval == kMBRetry) {
                  HandleKey(event);
               }
            }
            ret = kTRUE;
            break;
         default:
            break;
      }
   } else {
      switch ((EKeySym)keysym ) {
         case kKey_Delete:
         case kKey_Backspace:
            HandleDelete(event->fState & kKeyShiftMask);
            ret = kTRUE;
            break;
         case kKey_Return:
         case kKey_Enter:
            HandleReturn(kFALSE);
            ret = kTRUE;
            break;
         case kKey_Left:
         case kKey_Right:
         case kKey_Up:
         case kKey_Down:
            if (fLassoDrawn) {
               HandleAlignment(keysym, event->fState & kKeyShiftMask);
            } else if (fPimpl->fGrab) {
               HandleLayoutOrder((keysym == kKey_Right) || (keysym == kKey_Down));
            }
            ret = kTRUE;
            break;
         case kKey_Space:
            //Compact(kFALSE);
            ret = kTRUE;
            break;
         default:
            break;
      }
   }
   if (fBuilder) {
      fBuilder->SetAction(0);
      fBuilder->Update();
   }

   return ret;
}

//______________________________________________________________________________
void TGuiBldDragManager::ReparentFrames(TGFrame *newfr, TGCompositeFrame *oldfr)
{
   //

   Int_t x0, y0, xx, yy;
   Window_t c;

   if (!newfr || !newfr->GetId() || !oldfr || !oldfr->GetId()) return;

   gVirtualX->TranslateCoordinates(newfr->GetId(), oldfr->GetId(),
                                   0, 0, x0, y0, c);
   x0 = x0 < 0 ? 0 : x0;
   y0 = y0 < 0 ? 0 : y0;
   Int_t x = x0 + newfr->GetWidth();
   Int_t y = y0 + newfr->GetHeight();

   TGCompositeFrame *comp = 0;
   if (newfr->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame*)newfr;
      comp->SetLayoutBroken();
   }

   TIter next(oldfr->GetList());
   TGFrameElement *el;

   while ((el = (TGFrameElement*)next())) {
      TGFrame *frame = el->fFrame;

      if ((frame->GetX() >= x0) && (frame->GetY() >= y0) &&
          (frame->GetX() + (Int_t)frame->GetWidth() <= x) &&
          (frame->GetY() + (Int_t)frame->GetHeight() <= y)) {

         if (frame == fPimpl->fGrab) UngrabFrame();

         oldfr->RemoveFrame(frame);

         gVirtualX->TranslateCoordinates(oldfr->GetId(), newfr->GetId(),
                                        frame->GetX(), frame->GetY(), xx, yy, c);

         frame->ReparentWindow(newfr, xx, yy);

         if (comp) {
            comp->AddFrame(frame, el->fLayout);
         }
      }
   }
}

//______________________________________________________________________________
TList *TGuiBldDragManager::GetFramesInside(Int_t x0, Int_t y0, Int_t x, Int_t y)
{
   //

   Int_t xx, yy;

   if (!fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class())) return 0;
   TList *list = new TList();

   xx = x0; yy = y0;
   x0 = TMath::Min(xx, x); x = TMath::Max(xx, x);
   y0 = TMath::Min(yy, y); y = TMath::Max(yy, y);

   TIter next(((TGCompositeFrame*)fClient->GetRoot())->GetList());
   TGFrameElement *el;

   while ((el = (TGFrameElement*)next())) {
      if ((el->fFrame->GetX() >= x0) && (el->fFrame->GetY() >= y0) &&
          (el->fFrame->GetX() + (Int_t)el->fFrame->GetWidth() <= x) &&
          (el->fFrame->GetY() + (Int_t)el->fFrame->GetHeight() <= y)) {
         list->Add(el->fFrame);
      }
   }
   if (list->IsEmpty()) {
      delete list;
      return 0;
   }
   return list;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleReturn(Bool_t on)
{
   //

   Int_t x0, y0, x, y, xx, yy;
   Window_t c;
   TGCompositeFrame *parent = 0;
   TList *li = 0;

   if (!fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class())) return;

   TGCompositeFrame *comp = (TGCompositeFrame*)fClient->GetRoot();

   if (fLassoDrawn) {
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX, fPimpl->fY, x, y, c);
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX0, fPimpl->fY0, x0, y0, c);
      xx = x0; yy = y0;
      x0 = TMath::Min(xx, x); x = TMath::Max(xx, x);
      y0 = TMath::Min(yy, y); y = TMath::Max(yy, y);

      li = GetFramesInside(x0, y0, x, y);

      if (!on && li) {
         parent = new TGCompositeFrame(comp, x - x0, y - y0);
         parent->MoveResize(x0, y0, x - x0, y - y0);
         ReparentFrames(parent, comp);

         comp->AddFrame(parent);
         parent->MapWindow();
         fLassoDrawn = kFALSE;
         SelectFrame(parent);

         if (fBuilder) {
            fBuilder->UpdateStatusBar("Grab action performed");
         }
      }
   } else if (on && fPimpl->fGrab) {
      if (fPimpl->fGrab->InheritsFrom(TGCompositeFrame::Class())) {
         parent = (TGCompositeFrame*)fPimpl->fGrab;
      } else {
         //parent = (TGCompositeFrame*)fPimpl->fGrab->GetParent();
      }
      if (parent) {
         ReparentFrames(comp, parent);
         DeleteFrame(fPimpl->fGrab);
         fPimpl->fGrab = 0;

         if (fBuilder) {
            fBuilder->UpdateStatusBar("Drop action performed");
         }
      }
   }
   delete li;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleAlignment(Int_t to, Bool_t lineup)
{
   // align frames located inside lasso

   Int_t x0, y0, x, y, xx, yy;
   Window_t c;
   TGCompositeFrame *comp = 0;

   if (!fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class())) return;

   if (fLassoDrawn) {
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX, fPimpl->fY, x, y, c);
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX0, fPimpl->fY0, x0, y0, c);
      xx = x0; yy = y0;
      x0 = TMath::Min(xx, x); x = TMath::Max(xx, x);
      y0 = TMath::Min(yy, y); y = TMath::Max(yy, y);

      comp = (TGCompositeFrame*)fClient->GetRoot();

      ToGrid(x, y);
      ToGrid(x0, y0);

      TIter next(comp->GetList());
      TGFrameElement *el;
      TGFrame *prev = 0;

      while ((el = (TGFrameElement*)next())) {
         TGFrame *fr = el->fFrame;

         if ((fr->GetX() >= x0) && (fr->GetY() >= y0) &&
             (fr->GetX() + (Int_t)fr->GetWidth() <= x) &&
             (fr->GetY() + (Int_t)fr->GetHeight() <= y)) {

            switch ((EKeySym)to) {
               case kKey_Left:
                  fr->Move(x0, fr->GetY());
                  if (lineup) {
                     if (prev) fr->Move(fr->GetX(), prev->GetY() + prev->GetHeight());
                     else fr->Move(x0, y0);
                  }
                  break;
               case kKey_Right:
                  fr->Move(x - fr->GetWidth(), fr->GetY());
                  if (lineup) {
                     if (prev) fr->Move(fr->GetX(), prev->GetY() + prev->GetHeight());
                     else fr->Move(x - fr->GetWidth(), y0);
                  }
                  break;
               case kKey_Up:
                  fr->Move(fr->GetX(), y0);
                  if (lineup) {
                     if (prev) fr->Move(prev->GetX() + prev->GetWidth(), fr->GetY());
                     else fr->Move(x0, y0);
                  }
                  break;
               case kKey_Down:
                  fr->Move(fr->GetX(), y - fr->GetHeight());
                  if (lineup) {
                     if (prev) fr->Move(prev->GetX() + prev->GetWidth(), fr->GetY());
                     else fr->Move(x0, y - fr->GetHeight());
                  }
                  break;
               default:
                  break;
            }
            prev = fr;
         }
      }
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleDelete(Bool_t crop)
{
   //

   Int_t x0, y0, x, y, xx, yy, w, h;
   Window_t c;

   if (!fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class())) return;

   TGCompositeFrame *comp = 0;
   Bool_t fromGrab = kFALSE;

   if (fBuilder && crop) {
      comp = fBuilder->FindEditableMdiFrame(fClient->GetRoot());
   } else {
      comp = (TGCompositeFrame*)fClient->GetRoot();
   }

   if (fPimpl->fGrab && !fLassoDrawn && crop &&
       fPimpl->fGrab->InheritsFrom(TGCompositeFrame::Class())) {
      ReparentFrames(comp, (TGCompositeFrame*)fPimpl->fGrab);

      gVirtualX->TranslateCoordinates(fClient->GetRoot()->GetId(),
                                      fClient->GetDefaultRoot()->GetId(),
                                      fPimpl->fGrab->GetX(),
                                      fPimpl->fGrab->GetY(),
                                      fPimpl->fX0, fPimpl->fY0, c);
      fPimpl->fX = fPimpl->fX0 + fPimpl->fGrab->GetWidth();
      fPimpl->fY = fPimpl->fY0 + fPimpl->fGrab->GetHeight();
      fromGrab = kTRUE;
   }

   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX, fPimpl->fY, x, y, c);
   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX0, fPimpl->fY0, x0, y0, c);
   xx = x0; yy = y0;
   x0 = TMath::Min(xx, x); x = TMath::Max(xx, x);
   y0 = TMath::Min(yy, y); y = TMath::Max(yy, y);
   w = x - x0;
   h = y - y0;

   if (fLassoDrawn || fromGrab) {
      TIter next(comp->GetList());
      TGFrameElement *el;

      while ((el = (TGFrameElement*)next())) {
         TGFrame *fr = el->fFrame;

         if ((fr->GetX() >= x0) && (fr->GetY() >= y0) &&
             (fr->GetX() + (Int_t)fr->GetWidth() <= x) &&
             (fr->GetY() + (Int_t)fr->GetHeight() <= y)) {
            if (!crop) {
               DeleteFrame(fr);
            } else {
               fr->Move(fr->GetX() - x0, fr->GetY() - y0);
            }
         } else {
            if (crop) {
               DeleteFrame(fr);
            }
         }
      }
      if (crop && comp) {
         gVirtualX->TranslateCoordinates(comp->GetId(), comp->GetParent()->GetId(),
                                          x0, y0, xx, yy, c);
         comp->MoveResize(xx, yy, w, h);

         if (comp->GetParent()->InheritsFrom(TGMdiDecorFrame::Class())) {
            TGMdiDecorFrame *decor = (TGMdiDecorFrame *)comp->GetParent();

            gVirtualX->TranslateCoordinates(decor->GetId(), decor->GetParent()->GetId(),
                                            xx, yy, xx, yy, c);
            Int_t b = 2 * decor->GetBorderWidth();
            decor->MoveResize(xx, yy, comp->GetWidth() + b,
                              comp->GetHeight() + b + decor->GetTitleBar()->GetDefaultHeight());
         }
      }
      if (fromGrab)  {
         DeleteFrame(fPimpl->fGrab);
         fPimpl->fGrab = 0;
      }
   } else { //  no lasso drawn
      TString sav = fPasteFileName;
      fPasteFileName = gSystem->ConcatFileName(gSystem->TempDirectory(),
                             Form("delete_RootGuiBldClipboard%d.C", gSystem->GetPid()));
      HandleCut();
      fPasteFileName = sav;
   }
   fLassoDrawn = kFALSE;

   if (fBuilder) {
      fBuilder->UpdateStatusBar(crop ? "Crop action performed" : "Delete action performed");
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::DeleteFrame(TGFrame *frame)
{
   //

   if (!frame) return;

   frame->UnmapWindow();

   TGCompositeFrame *comp = 0;

   if (frame->GetParent()->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame*)frame->GetParent();
   }

   if (comp) comp->RemoveFrame(frame);

   if (frame == fPimpl->fGrab) UngrabFrame();

   fClient->UnregisterWindow(frame);
   //frame->DeleteWindow();
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleCut()
{
   //

   if (!fPimpl->fGrab) return;

   HandleCopy();
   DeleteFrame(fPimpl->fGrab);
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleCopy()
{
   //

   if (!fPimpl->fGrab) return;

   TGMainFrame *tmp = new TGMainFrame(fClient->GetDefaultRoot(),
                                      fPimpl->fGrab->GetWidth(),
                                      fPimpl->fGrab->GetHeight());

   // save coordinates
   Int_t x0 = fPimpl->fGrab->GetX();
   Int_t y0 = fPimpl->fGrab->GetY();

   // save parent name
   TString name = fPimpl->fGrab->GetParent()->GetName();

   ((TGWindow*)fPimpl->fGrab->GetParent())->SetName(tmp->GetName());

   fPimpl->fGrab->SetX(0);
   fPimpl->fGrab->SetY(0);

   TGFrameElement *fe = fPimpl->fGrab->GetFrameElement();

   tmp->GetList()->Add(fe);
   tmp->SetLayoutBroken();
   tmp->SaveSource(fPasteFileName.Data(), "quiet");

   tmp->GetList()->Remove(fe);

   fPimpl->fGrab->SetX(x0);
   fPimpl->fGrab->SetY(y0);

   ((TGWindow*)fPimpl->fGrab->GetParent())->SetName(name.Data());

   if (fBuilder) {
      TString str = fPimpl->fGrab->ClassName();
      str += "::";
      str += fPimpl->fGrab->GetName();
      str += " copied to clipboard";
      fBuilder->UpdateStatusBar(str.Data());
   }

   delete tmp;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandlePaste()
{
   //

   Int_t xp = 0;
   Int_t yp = 0;
   TGFrame *frame = 0;

   if (gSystem->AccessPathName(fPasteFileName.Data())) return;

   fPasting = kTRUE;
   gROOT->Macro(fPasteFileName.Data());

   Window_t c;

   if (!fPimpl->fReplaceOn) {
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      fPimpl->fX0, fPimpl->fY0, xp, yp, c);
      ToGrid(xp, yp);
   }

   if (fPasteFrame) {
      TGCompositeFrame *comp = 0;
      comp = (TGCompositeFrame*)fPasteFrame;

      TList *list = comp->GetList();
      TGFrameElement *fe = 0;

      fe = (TGFrameElement*)list->First();

      comp = (TGCompositeFrame*)fClient->GetRoot();

      if (fe) {
         frame = fe->fFrame;
         if (frame) frame->ReparentWindow(fClient->GetRoot(), xp, yp);
         list->Remove(fe);
         comp->GetList()->Add(fe);
      }

      comp->RemoveFrame(fPasteFrame);
      fClient->UnregisterWindow(fPasteFrame);
      fPasteFrame->DestroyWindow();
      //delete fPasteFrame;
      fPasteFrame = frame;

      if (!fPimpl->fReplaceOn) {
         SelectFrame(frame);
      }
   }
   fPasting = kFALSE;

   if (fBuilder) {
      fBuilder->UpdateStatusBar("Paste action performed");
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::DoReplace(TGFrame *frame)
{
   //

   if (!frame || !fPimpl->fGrab || !fPimpl->fReplaceOn) return;

   Int_t w = fPimpl->fGrab->GetWidth();
   Int_t h = fPimpl->fGrab->GetHeight();
   Int_t x = fPimpl->fGrab->GetX();
   Int_t y = fPimpl->fGrab->GetY();

   if (fBuilder) {
      TString str = fPimpl->fGrab->ClassName();
      str += "::";
      str += fPimpl->fGrab->GetName();
      str += " replaced by ";
      str += frame->ClassName();
      str += "::";
      str += frame->GetName();
      fBuilder->UpdateStatusBar(str.Data());
   }

   TGFrameElement *fe = fPimpl->fGrab->GetFrameElement();

   fe->fFrame = 0;
   fPimpl->fGrab->DestroyWindow();
   delete fPimpl->fGrab;
   fPimpl->fGrab = 0;

   fe->fFrame = frame;
   frame->MoveResize(x, y, w, h);
   frame->MapRaised();
   frame->SetFrameElement(fe);

   SelectFrame(frame);
   fPimpl->fReplaceOn = kFALSE;

   TGWindow *root = (TGWindow *)fClient->GetRoot();
   root->SetEditable(kFALSE);
   DoRedraw();
   root->SetEditable(kTRUE);
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleReplace()
{
   //

   if (!fPimpl->fGrab) return;

   fPimpl->fReplaceOn = kTRUE;
   TGFrame *frame = 0;

   if (fBuilder && fBuilder->IsExecutalble())  {
      frame = (TGFrame *)fBuilder->ExecuteAction();
   } else {
      HandlePaste();
      frame = fPasteFrame;
   }
   DoReplace(frame);
   fPimpl->fReplaceOn = kFALSE;
}

//______________________________________________________________________________
void TGuiBldDragManager::CloneEditable()
{
   //

   TString tmpfile = gSystem->TempDirectory();
   tmpfile = gSystem->ConcatFileName(tmpfile.Data(),
                                     Form("tmp%d.C", gRandom->Integer(100)));
   Save(tmpfile.Data());
   gROOT->Macro(tmpfile.Data());
   gSystem->Unlink(tmpfile.Data());

   if (fClient->GetRoot()->InheritsFrom(TGFrame::Class())) {
      TGFrame *f = (TGFrame *)fClient->GetRoot();
      f->Resize(f->GetWidth() + 10, f->GetHeight() + 10);
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::Save(const char *file)
{
   //

   if (!fClient->GetRoot() || !fClient->IsEditable()) return;

   TGMainFrame *main = (TGMainFrame*)fClient->GetRoot()->GetMainFrame();
   Bool_t lbrk = main->IsLayoutBroken();
   main->SetLayoutBroken();

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   TString fname = file;

   root->SetEditable(kFALSE);

   if (!file || !strlen(file)) {
      static TString dir(".");
      static Bool_t overwr = kFALSE;
      static const char *gSaveMacroTypes[] = { "Macro files", "*.C",
                                               "All files",   "*",
                                               0,             0 };
      TGFileInfo fi;

      fi.fFileTypes = gSaveMacroTypes;
      fi.fIniDir    = StrDup(dir);
      fi.fOverwrite = overwr;
      new TGFileDialog(fClient->GetDefaultRoot(), this, kFDSave, &fi);

      if (!fi.fFilename) goto out;
      dir = fi.fIniDir;
      overwr = fi.fOverwrite;
      fname = gSystem->BaseName(gSystem->UnixPathName(fi.fFilename));
   }

   if (strstr(fname.Data(), ".C")) {
      main->SaveSource(fname.Data(), file ? "quiet" : "");
   } else {
      Int_t retval;
      TString msg = Form("file (%s) must have extension .C", fname.Data());

      new TGMsgBox(fClient->GetDefaultRoot(), main, "Error...", msg.Data(),
                   kMBIconExclamation, kMBRetry | kMBCancel, &retval);

      if (retval == kMBRetry) {
         Save();
      }
   }

out:
   main->SetLayoutBroken(lbrk);
   root->SetEditable(kTRUE);
}

//______________________________________________________________________________
void TGuiBldDragManager::DoResize()
{
   // handle resize

   if (!fPimpl->fGrab) return;

   TGFrame *fr = fPimpl->fGrab;

   Window_t c;
   Int_t x = fPimpl->fX;
   Int_t y = fPimpl->fY;

   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fr->GetId(), x, y, x, y, c);
   ToGrid(x, y);

   switch (fPimpl->fResizeType) {
      case kTopLeft:
         if ((((int)fr->GetWidth() > x) || (x < 0)) &&
             (((int)fr->GetHeight() > y) || (y < 0))) {
            fr->MoveResize(fr->GetX() + x, fr->GetY() + y,
                           fr->GetWidth() - x, fr->GetHeight() - y);
         }
         break;
      case kTopRight:
         if ((x > 0) && (((int)fr->GetHeight() > y) || (y < 0))) {
            fr->MoveResize(fr->GetX(), fr->GetY() + y,
                           x, fr->GetHeight() - y);
         }
         break;
      case kTopSide:
         if (((int)fr->GetHeight() > y) || (y < 0)) {
            fr->MoveResize(fr->GetX(), fr->GetY() + y,
                        fr->GetWidth(), fr->GetHeight() - y);
         }
         break;
      case kBottomLeft:
         if ((((int)fr->GetWidth() > x) || (x < 0)) && (y > 0)) {
            fr->MoveResize(fr->GetX() + x, fr->GetY(),
                        fr->GetWidth() - x, y);
         }
         break;
      case kBottomRight:
         if ((x > 0) && (y > 0)) {
            fr->Resize(x, y);
         }
         break;
      case kBottomSide:
         if (y > 0) {
            fr->Resize(fr->GetWidth(), y);
         }
         break;
      case kLeftSide:
         if ((int)fr->GetWidth() > x) {
         fr->MoveResize(fr->GetX() + x, fr->GetY(),
                        fr->GetWidth() - x, fr->GetHeight());
         }
         break;
      case kRightSide:
         if (x > 0) {
            fr->Resize(x, fr->GetHeight());
         }
         break;
      default:
         break;
   }
   gVirtualX->SetCursor(fClient->GetRoot()->GetId(),
                        gVirtualX->CreateCursor(fPimpl->fResizeType));
   fClient->NeedRedraw(fr, kTRUE);
   DoRedraw();
}

//______________________________________________________________________________
void TGuiBldDragManager::DoMove()
{
   //

   if (!fPimpl->fGrab ) return;

   Int_t x = fPimpl->fX - fPimpl->fXf;
   Int_t y = fPimpl->fY - fPimpl->fYf;

   fPimpl->fGrab->Move(x, y);
   CheckTargetUnderGrab();
}

//______________________________________________________________________________
void TGuiBldDragManager::CheckTargetUnderGrab()
{
   //

   Window_t c;
   TGWindow *win = 0;

   if (!fPimpl->fGrab ) return;

   Int_t x = fPimpl->fGrab->GetX();
   Int_t y = fPimpl->fGrab->GetY();

   Window_t w = GetWindowFromPoint(x - 1, y - 1);

   if (w && (w != gVirtualX->GetDefaultRootWindow())) {
      win = fClient->GetWindowById(w);
      TGCompositeFrame *comp = 0;

      if (!win) goto out;

      if (win->InheritsFrom(TGCompositeFrame::Class())) {
         comp = (TGCompositeFrame *)win;
      } else if (win->GetParent() != fClient->GetDefaultRoot()) {
         comp = (TGCompositeFrame *)win->GetParent();
      }

      if (comp) {
         gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                         comp->GetId(), x, y, x, y, c);

         if ((comp != fPimpl->fGrab) && (x >= 0) && (y >= 0) &&
             (x + fPimpl->fGrab->GetWidth() <= comp->GetWidth()) &&
             (y + fPimpl->fGrab->GetHeight() <= comp->GetHeight())) {

            if (comp != fTarget) {
               comp->HandleDragEnter(fPimpl->fGrab);
               if (fTarget) fTarget->HandleDragLeave(fPimpl->fGrab);
               else Snap2Grid();
            } else {
               if (fTarget) fTarget->HandleDragMotion(fPimpl->fGrab);
            }

            fTarget = comp;
            fTargetId = comp->GetId();

            return;
         } else {
            if (fTarget) fTarget->HandleDragLeave(fPimpl->fGrab);
            fTarget = 0;
            fTargetId = 0;
         }
      }
   }

out:
   if (fTarget) fTarget->HandleDragLeave(fPimpl->fGrab);
   if (!w || !win) {
      fTarget = 0;
      fTargetId = 0;
   }
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleMotion(Event_t *event)
{
   // handle motion event

   static Long_t was = gSystem->Now();
   static Int_t gy = event->fYRoot;
   static Int_t gx = event->fXRoot;

   Long_t now = (long)gSystem->Now();

   if ((now-was < 100) || !(event->fState & kButton1Mask) ||
       ((event->fYRoot == gy) && (event->fXRoot == gx))) {
      return kFALSE;
   }
   was = now;
   gy = event->fYRoot;
   gx = event->fXRoot;

   if (!fDragging) {
      if (fMoveWaiting && ((TMath::Abs(fPimpl->fX - event->fXRoot) > 10) ||
          (TMath::Abs(fPimpl->fY - event->fYRoot) > 10))) {
         return StartDrag(fSource, event->fXRoot, event->fYRoot);
      }
   } else {
      fPimpl->fX = event->fXRoot;
      fPimpl->fY = event->fYRoot;

      switch (fDragType) {
         case kDragLasso:
            DrawLasso();
            fSelectionIsOn = event->fState & kKeyShiftMask;
            break;
         case kDragMove:
         case kDragCopy:
         case kDragLink:
            DoMove();
            break;
         case kDragResize:
            DoResize();
            break;
         default:
            break;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGuiBldDragManager::PlaceFrame(TGFrame *frame, TGLayoutHints *hints)
{
   //

   Int_t x0, y0, x, y;
   Window_t c;

   if (!frame || !fClient->IsEditable()) return;

   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX0 , fPimpl->fY0, x0, y0, c);
   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX , fPimpl->fY, x, y, c);
   ToGrid(x, y);
   ToGrid(x0, y0);

   Int_t w = TMath::Abs(x - x0);
   Int_t h = TMath::Abs(y - y0);

   frame->Move(x > x0 ? x0 : x, y > y0 ? y0 : y);
   frame->Resize(w, h);
   frame->MapRaised();
   frame->SetCleanup(kDeepCleanup);
   frame->AddInput(kButtonPressMask);

   if (fClient->GetRoot()->InheritsFrom(TGCompositeFrame::Class())) {
      TGCompositeFrame *edit = (TGCompositeFrame*)fClient->GetRoot();
      edit->SetCleanup(kDeepCleanup);
      ReparentFrames(frame, edit);
      frame->MapRaised();
      edit->SetLayoutBroken();
      UInt_t g = GetGridStep()/2;
      edit->AddFrame(frame, hints ? hints : new TGLayoutHints(kLHintsNormal, g, g, g, g));
   }
   if (fBuilder) {
      TString str = frame->ClassName();
      str += "::";
      str += frame->GetName();
      str += " created";
      fBuilder->UpdateStatusBar(str.Data());
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::DrawLasso()
{
   // draw lasso for allocation new object

   if (!fClient->IsEditable()) return;

   UngrabFrame();

   Int_t x0, y0, x, y;
   Window_t c;

   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX0 , fPimpl->fY0, x0, y0, c);
   gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                   fClient->GetRoot()->GetId(),
                                   fPimpl->fX , fPimpl->fY, x, y, c);
   ToGrid(x, y);
   ToGrid(x0, y0);
   Int_t w = TMath::Abs(x - x0);
   Int_t h = TMath::Abs(y - y0);

   DoRedraw();

   gVirtualX->DrawRectangle(fClient->GetRoot()->GetId(),
                            GetBlackGC()(), x > x0 ? x0 : x,
                            y > y0 ? y0 : y, w, h);
   gVirtualX->DrawRectangle(fClient->GetRoot()->GetId(),
                            GetBlackGC()(), x > x0 ? x0-1 : x-1,
                            y > y0 ? y0-1 : y-1, w+2, h+2);

   gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kCross));
   gVirtualX->SetCursor(fClient->GetRoot()->GetId(), gVirtualX->CreateCursor(kCross));

   fLassoDrawn = kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleClientMessage(Event_t *event)
{
   //

   if ((event->fFormat == 32) && ((Atom_t)event->fUser[0] == gWM_DELETE_WINDOW) &&
       (event->fHandle != gROOT_MESSAGE)) {

      TGMainFrame *main = (TGMainFrame*)fClient->GetRoot()->GetMainFrame();

      if (event->fWindow == main->GetId()) {
         fClient->SetRoot(0);

         if (main != fBuilder) {
            main->Cleanup();
            if (fEditor && !fEditor->IsEmbedded()) {
               delete fEditor;
               fEditor = 0;
            }
            return kFALSE;
         }

         fBuilder->Hide();

         delete fFrameMenu;
         fFrameMenu =0;

         delete fLassoMenu;
         fLassoMenu = 0;

         //delete fPimpl->fGrid;
         fPimpl->fGrid = 0;
         Init();
         if (fPimpl) fPimpl->ResetParams();

      } else if (fBuilder && (event->fWindow == fBuilder->GetId())) {
         fBuilder->Hide();

      } else if (fEditor && (event->fWindow == fEditor->GetMainFrame()->GetId())) {
         TQObject::Disconnect(fEditor);
         fEditor = 0;
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleSelection(Event_t *)
{
   //

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::HandleSelectionRequest(Event_t *)
{
   //

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::StartDrag(TGFrame *src, Int_t x, Int_t y)
{
   // start dragging

   if (fDragging) return kFALSE;

   SetEditable(kTRUE);  // grab server

   fPimpl->fX = x;
   fPimpl->fY = y;
   fSelectionIsOn = kFALSE;

   fPimpl->fRepeatTimer->Reset();
   gSystem->AddTimer(fPimpl->fRepeatTimer);

   fMoveWaiting = kFALSE;
   fDragging = kTRUE;

   switch (fDragType) {
      case kDragCopy:
         fPimpl->fGrab = new TGDragGrabber(src, x, y);
         //GrabFrame(fPimpl->fGrab);
         break;
      case kDragMove:
         fPimpl->fGrab = src;
         GrabFrame(fPimpl->fGrab);
         break;
      default:
         //fPimpl->fGrab = 0;
         break;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::EndDrag()
{
   // end dragging

   TGFrame *frame = 0;
   Bool_t ret = kFALSE;

   if (fPimpl->fGrab && (fDragType >= kDragMove) && (fDragType <= kDragLink)) {

      ret = Drop();
   } else if (fBuilder && fBuilder->IsExecutalble() &&
              (fDragType == kDragLasso) && !fSelectionIsOn) {

      frame = (TGFrame*)fBuilder->ExecuteAction();
      PlaceFrame(frame, fBuilder->GetAction()->fHints);
      ret = kTRUE;
   } else if ((fDragType == kDragLasso) && fSelectionIsOn) {

      HandleReturn(kFALSE);
      ret = kTRUE;
   }

   if (!fLassoDrawn) DoRedraw();

   Init();
   if (fBuilder) {
      fBuilder->SetAction(0);
      fBuilder->Update();
      if (fLassoDrawn) {
         fBuilder->UpdateStatusBar("Lasso Drawn");
      }
   }

   return ret;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::Cancel(Bool_t /*delSrc*/)
{
   //

   fTarget = 0;
   EndDrag();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::Drop()
{
   //

   if (!fPimpl->fGrab && !fSelectionIsOn) return kFALSE;

   fDropStatus = kFALSE;
   TGFrame *frame = 0;
   TGFrame *parent = 0;
   Int_t x, y;
   Window_t c;

   switch (fDragType) {
      case kDragCopy:
         frame = (TGFrame*)fPimpl->fGrab->Clone();
         break;
      case kDragMove:
         frame = fPimpl->fGrab;
         break;
      default:
         break;
   }

   TGWindow *w = fClient->GetWindowById(fTargetId);

   if (fTarget && fPimpl->fGrab && (w == fTarget) &&  w &&
       (w != fClient->GetDefaultRoot())) {
      parent = fTarget;
      gVirtualX->TranslateCoordinates(fClient->GetDefaultRoot()->GetId(),
                                      fTarget->GetId(),
                                      fPimpl->fGrab->GetX(),
                                      fPimpl->fGrab->GetY(), x, y, c);

      fTarget->HandleDragLeave(fPimpl->fGrab);
   } else {
      parent = (TGFrame*)fPimpl->fGrabParent;
      x = fPimpl->fGrabX;
      y = fPimpl->fGrabY;
   }

   if (parent && frame) {
      ToGrid(x, y);
      fDropStatus = parent->HandleDragDrop(frame, x, y, fPimpl->fGrabLayout);

      if (!fDropStatus) {
         parent = (TGFrame*)fPimpl->fGrabParent;
         x = fPimpl->fGrabX;
         y = fPimpl->fGrabY;
         frame = fPimpl->fGrab;

         parent->HandleDragDrop(frame, x, y, fPimpl->fGrabLayout);
         fDropStatus = kTRUE;
      }
   }

   if (fDropStatus) {
      if (fBuilder) {
         TString str = frame->ClassName();
         str += "::";
         str += frame->GetName();
         str += " dropped into ";
         str += parent->ClassName();
         str += "::";
         str += parent->GetName();
         fBuilder->UpdateStatusBar(str.Data());
      }
      fTarget = 0;
      fTargetId = 0;

      if (parent && (parent == fPimpl->fGrabParent) && fPimpl->fGrabListPosition &&
          frame && parent->InheritsFrom(TGCompositeFrame::Class())) {

         TList *li = ((TGCompositeFrame*)parent)->GetList();
         li->Remove(frame->GetFrameElement());
         li->AddAfter(fPimpl->fGrabListPosition, frame->GetFrameElement());
      }
   }
   fPimpl->fGrabParent = 0;
   fPimpl->fGrabX = 0;
   fPimpl->fGrabY = 0;
   fPimpl->fGrabListPosition = 0;

   return fDropStatus;
}

//______________________________________________________________________________
Bool_t TGuiBldDragManager::IsMoveWaiting() const
{
   // Waits for either the mouse move from the given initial ButtonPress location
   // or for the mouse button to be released. If mouse moves away from the initial
   // ButtonPress location before the mouse button is released "IsMoveWaiting"
   // returns kTRUE. If the mouse button released before the mose moved from the
   // initial ButtonPress location, "IsMoveWaiting" returns kFALSE.

   return fMoveWaiting;
}

//______________________________________________________________________________
void TGuiBldDragManager::Compact(Bool_t global)
{
   // Layout and Resize frame.
   // If global is kFALSE - compact selected frame
   // If global is kFALSE - compact main frame of selected frame

   TGCompositeFrame *comp = 0;
   TGFrameElement *fe;

   if (!fClient || !fClient->IsEditable()) return;

   if (global) {
      if (!fBuilder) comp = (TGCompositeFrame*)fClient->GetRoot()->GetMainFrame();
      else comp = fBuilder->FindEditableMdiFrame(fClient->GetRoot());
   } else {
      if (fPimpl->fGrab &&
          fPimpl->fGrab->InheritsFrom(TGCompositeFrame::Class())) {
         comp = (TGCompositeFrame*)fPimpl->fGrab;
      } else {
         comp = (TGCompositeFrame*)fPimpl->fGrab->GetParent();
      }
   }
   if (!comp) return;

   TIter next(comp->GetList());

   TGFrame *root = (TGFrame *)fClient->GetRoot();
   root->SetEditable(kFALSE);

   if (global) {
      while ((fe = (TGFrameElement*)next())) {
         fe->fFrame->SetLayoutBroken(kFALSE);
         fe->fFrame->Resize();
      }
      root->SetLayoutBroken(kFALSE);
      fPimpl->fCompacted = kTRUE;
   }

   comp->SetLayoutBroken(kFALSE);
   comp->Resize();

   if (comp->GetParent()->InheritsFrom(TGMdiDecorFrame::Class())) {
      TGMdiDecorFrame *decor = (TGMdiDecorFrame *)comp->GetParent();
      Int_t b = 2 * decor->GetBorderWidth();
      decor->MoveResize(decor->GetX(), decor->GetY(), comp->GetWidth() + b,
                        comp->GetHeight() + b + decor->GetTitleBar()->GetDefaultHeight());
   }

   root->SetEditable(kTRUE);
   DoRedraw();
   DrawGrabRectangles();
}

//______________________________________________________________________________
void TGuiBldDragManager::SetEditable(Bool_t on)
{
   // grab server

   static Bool_t gon = kFALSE;
   static const TGWindow *gw = 0;

   HideGrabRectangles();

   if ((gon == on) && (fClient->GetRoot() == gw)) return;
   gon = on;  gw = fClient->GetRoot();

   Snap2Grid();

   if (on) {
      if (!fClient->GetRoot()->InheritsFrom(TGFrame::Class())) return;

      TGFrame *fr = (TGFrame*)fClient->GetRoot();
      fEventMask = fr->GetEventMask(); //saved event mask
      fr->AddInput(kKeyPressMask | kButtonPressMask);

      if (fPimpl->fRepeatTimer) {
         fPimpl->fRepeatTimer->Reset();
      } else {
         fPimpl->fRepeatTimer = new TGuiBldDragManagerRepeatTimer(this, 100);
      }
      gSystem->AddTimer(fPimpl->fRepeatTimer);

   } else if (fClient->GetRoot()->IsEditable()) {
      if (fPimpl->fRepeatTimer) fPimpl->fRepeatTimer->Remove();
      UngrabFrame();
      gVirtualX->SelectInput(fClient->GetRoot()->GetId(), fEventMask);
   }

   if (on && fClient->IsEditable()) {
      gVirtualX->SetCursor(fClient->GetRoot()->GetId(),
                           gVirtualX->CreateCursor(kPointer));
   }

   if (!on && !gSystem->AccessPathName(fPasteFileName.Data())) {
      gSystem->Unlink(fPasteFileName.Data());
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::ToGrid(Int_t &x, Int_t &y)
{
   //

   UInt_t step = GetGridStep();
   x = x - x%step;
   y = y - y%step;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleAction(Int_t act)
{
   //

   fPimpl->fLastPopupAction = act;

   if (fPimpl->fPlane) {
      fPimpl->fPlane->ChangeOptions(fPimpl->fPlane->GetOptions() & ~kRaisedFrame);
      fClient->NeedRedraw(fPimpl->fPlane, kTRUE);
   }

   switch ((EActionType)act) {
      case kPropertyAct:
         CreatePropertyEditor();
         break;
      case kEditableAct:
         if (fPimpl->fSaveGrab) fPimpl->fSaveGrab->SetEditable(kTRUE);
         break;
      case kCutAct:
         HandleCut();
         break;
      case kCopyAct:
         HandleCopy();
         break;
      case kPasteAct:
         HandlePaste();
         break;
      case kCropAct:
         HandleDelete(kTRUE);
         break;
      case kCompactAct:
         Compact(kFALSE);
         break;
      case kCompactGlobalAct:
         Compact(kTRUE);
         break;
      case kDropAct:
         HandleReturn(kTRUE);
         break;
      case kLayUpAct:
         HandleLayoutOrder(kFALSE);
         break;
      case kLayDownAct:
         HandleLayoutOrder(kTRUE);
         break;
      case kCloneAct:
         CloneEditable();
         break;
      case kGrabAct:
         HandleReturn(fBuilder && !fBuilder->IsGrabButtonDown());
         break;
      case kDeleteAct:
         HandleDelete(kFALSE);
         break;
      case kLeftAct:
         HandleAlignment(kKey_Left);
         break;
      case kRightAct:
         HandleAlignment(kKey_Right);
         break;
      case kUpAct:
         HandleAlignment(kKey_Up);
         break;
      case kDownAct:
         HandleAlignment(kKey_Down);
         break;
      case kEndEditAct:
         SetEditable(kFALSE);
         ((TGWindow*)fClient->GetRoot())->SetEditable(kFALSE);
         DeletePropertyEditor();
         break;
      case kReplaceAct:
         HandleReplace();
         break;
      case kGridAct:
         HandleGrid();
         break;
      case kBreakLayoutAct:
         BreakLayout();
         break;
      case kSwitchLayoutAct:
      case kLayoutVAct:
      case kLayoutHAct:
         SwitchLayout();
         break;
      case kNewAct:
         if (fBuilder) {
            fBuilder->NewProject();
         } else {
            TGMainFrame *main = new TGMainFrame(fClient->GetDefaultRoot(), 300, 300);
            main->MapRaised();
            main->SetEditable(kTRUE);
         }
         break;
      case kOpenAct:
         if (fBuilder) {
            fBuilder->OpenProject();
         } else {
            TGMainFrame *main = new TGMainFrame(fClient->GetDefaultRoot(), 300, 300);
            main->MapRaised();
            main->SetEditable(kTRUE);
         }
         break;
      case kSaveAct:
         if (fBuilder) {
            fBuilder->SaveProject();
         } else {
            Save();
         }
         break;
      default:
         break;
   }

   if (fBuilder) {
      fBuilder->SetAction(0);
      fBuilder->Update();
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::Menu4Frame(TGFrame *frame, Int_t x, Int_t y)
{
   // create and  place context menu for selected frame

   fPimpl->fSaveGrab = fPimpl->fGrab;
   fPimpl->fX0 = x;
   fPimpl->fY0 = y;

   fPimpl->fLastFrame = frame;

   Bool_t composite = frame->InheritsFrom(TGCompositeFrame::Class());
   Bool_t compar = frame->GetParent()->InheritsFrom(TGCompositeFrame::Class());

   TGCompositeFrame *cfr = 0;
   TGCompositeFrame *cfrp = 0;
   TGLayoutManager *lm = 0;

   if (composite)  {
      cfr = (TGCompositeFrame *)frame;
      lm = cfr->GetLayoutManager();
   }
   if (compar)  {
      cfrp = (TGCompositeFrame *)frame->GetParent();
   }

   delete fFrameMenu;

   fFrameMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fFrameMenu->SetEditDisabled();

   TString title = frame->ClassName();
   title += "::";
   title += frame->GetName();
   fFrameMenu->AddLabel(title.Data());
   fFrameMenu->AddSeparator();

   if (!fBuilder) fFrameMenu->AddEntry("Gui Builder", kPropertyAct);

   if (!frame->IsEditable() && !InEditable(frame->GetId())) {
      fPimpl->fSaveGrab = frame;
      goto out;
   }
   fFrameMenu->AddSeparator();

   if (!frame->IsEditable()) {
      if (composite) fFrameMenu->AddEntry("Drop               Ctrl+Return", kDropAct);
      fFrameMenu->AddSeparator();
      fFrameMenu->AddEntry("Cut                 Ctrl+X", kCutAct);
      fFrameMenu->AddEntry("Copy              Ctrl+C", kCopyAct);
      fFrameMenu->AddEntry("Delete             Del", kDeleteAct);
      if (composite) fFrameMenu->AddEntry("Crop               Shift+Del", kCropAct);
      fFrameMenu->AddEntry("Replace          Ctrl+R", kReplaceAct);
   } else {
      if (!gSystem->AccessPathName(fPasteFileName.Data())) {
         fFrameMenu->AddEntry("Paste               Ctrl+V", kPasteAct);
      }
      if (frame->GetMainFrame() == frame) {
         fFrameMenu->AddEntry("Clone               Ctrl+A", kCloneAct);
      }
   }

   fFrameMenu->AddSeparator();

   if (composite && !cfr->GetList()->IsEmpty()) {
      if (frame->IsLayoutBroken()) {
         if (!frame->IsEditable()) {
            fFrameMenu->AddEntry("Layout              Ctrl+L", kCompactAct);
         } else {
            fFrameMenu->AddEntry("Layout              Ctrl+L", kCompactGlobalAct);
         }
      } else {
         fFrameMenu->AddEntry("Break Layout    Ctrl+B", kBreakLayoutAct);
      }

      if (lm->InheritsFrom(TGVerticalLayout::Class()) ||
          lm->InheritsFrom(TGHorizontalLayout::Class())) {
         fFrameMenu->AddEntry("H/V Layout       Ctrl+H", kSwitchLayoutAct);
      }
   }
   if (compar && (cfrp->GetList()->GetSize() > 1) && !frame->IsEditable()) {
      if (cfrp->GetList()->First() != frame->GetFrameElement()) {
         fFrameMenu->AddEntry("Lay Up             Up/Left", kLayUpAct);
      }
      if (cfrp->GetList()->Last() != frame->GetFrameElement()) {
         fFrameMenu->AddEntry("Lay Down         Down/Right", kLayDownAct);
      }
   }
   fFrameMenu->AddSeparator();

   fFrameMenu->AddEntry("Grid On/Off       Ctrl+G", kGridAct);


   if (frame->IsEditable()) {
      fFrameMenu->AddSeparator();
      fFrameMenu->AddEntry("Save As ...         Ctrl+S", kSaveAct);
      fFrameMenu->AddEntry("End Edit         Ctrl+DblClick", kEndEditAct);
   } else if (composite) {
      fFrameMenu->AddSeparator();
      fFrameMenu->AddEntry("Edit              Ctrl+DblClick", kEditableAct);
   }

out:
   fFrameMenu->Connect("Activated(Int_t)", "TGuiBldDragManager", this, "HandleAction(Int_t)");

   fPimpl->fLastPopupAction = kNoneAct;
   fFrameMenu->PlaceMenu(x, y, kFALSE, kTRUE);

   if (fLassoDrawn) DrawLasso();
}

//______________________________________________________________________________
void TGuiBldDragManager::Menu4Lasso(Int_t x, Int_t y)
{
   //

   if (!fLassoDrawn) return;

   DrawLasso();

   if (!fLassoMenu) {
      fLassoMenu = new TGPopupMenu(fClient->GetDefaultRoot());
      fLassoMenu->SetEditDisabled();
      fLassoMenu->AddLabel("Edit actions");
      fLassoMenu->AddSeparator();
      fLassoMenu->AddEntry("Grab              Return", kGrabAct);
      fLassoMenu->AddSeparator();
      fLassoMenu->AddEntry("Delete            Delete", kDeleteAct);
      fLassoMenu->AddEntry("Crop              Shift+Delete", kCropAct);
      fLassoMenu->AddSeparator();
      fLassoMenu->AddEntry("Align Left        Left(Shift+Left)", kLeftAct);
      fLassoMenu->AddEntry("Align Right      Right(Shift+Right)", kRightAct);
      fLassoMenu->AddEntry("Align Up         Up(Shift+Up)", kUpAct);
      fLassoMenu->AddEntry("Align Down     Down(Shift+Down)", kDownAct);
      fLassoMenu->Connect("Activated(Int_t)", "TGuiBldDragManager", this, "HandleAction(Int_t)");
   }
   fPimpl->fLastPopupAction = kNoneAct;
   fLassoMenu->PlaceMenu(x, y, kFALSE, kTRUE);
}

//______________________________________________________________________________
void TGuiBldDragManager::CreatePropertyEditor()
{
   //

   if (!fPimpl->fLastFrame) return;

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   root->SetEditable(kFALSE);

   fBuilder = (TRootGuiBuilder*)TRootGuiBuilder::Instance();

   fBuilder->Move(fPimpl->fX0, fPimpl->fY0);
   fBuilder->SetWMPosition(fPimpl->fX0, fPimpl->fY0);
   SetPropertyEditor(fBuilder->GetEditor());

   root->SetEditable(kTRUE);
}

//______________________________________________________________________________
void TGuiBldDragManager::SetPropertyEditor(TGuiBldEditor *e)
{
   //

   fEditor = e;

   if (!fEditor) return;

   fEditor->ChangeSelected(fPimpl->fLastFrame);
   Connect("Selected(TGFrame*)", "TGuiBldEditor", fEditor, "ChangeSelected(TGFrame*)");
   fEditor->Connect("UpdateSelected(TGFrame*)", "TGuiBldDragManager", this,
                    "HandleUpdateSelected(TGFrame*)");
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleLayoutOrder(Bool_t forward)
{
   //

   if (!fPimpl->fGrab || !fPimpl->fGrab->GetFrameElement() ||
       !fPimpl->fGrab->GetParent()->InheritsFrom(TGCompositeFrame::Class())) {
      return;
   }

   TGCompositeFrame *comp = (TGCompositeFrame*)fPimpl->fGrab->GetParent();
   TList *li = comp->GetList();
   TGFrameElement *fe = fPimpl->fGrab->GetFrameElement();

   if (!fe) return;

   TGFrame *frame;
   TGFrameElement *el;

   if (forward) {
      el = (TGFrameElement *)li->After(fe);
      if (!el) return;
      frame = el->fFrame;

      el->fFrame = fPimpl->fGrab;
      fPimpl->fGrab->SetFrameElement(el);
      fe->fFrame = frame;
      frame->SetFrameElement(fe);
   } else {
      el = (TGFrameElement *)li->Before(fe);
      if (!el) return;
      frame = el->fFrame;

      el->fFrame = fPimpl->fGrab;
      fPimpl->fGrab->SetFrameElement(el);
      fe->fFrame = frame;
      frame->SetFrameElement(fe);
   }

   Bool_t sav = comp->IsLayoutBroken();
   comp->SetLayoutBroken(kFALSE);
   TGWindow *root = (TGWindow *)fClient->GetRoot();
   root->SetEditable(kFALSE);
   comp->Layout();
   DoRedraw();
   root->SetEditable(kTRUE);
   if (sav) comp->SetLayoutBroken(kTRUE);

   SelectFrame(el->fFrame);
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleGrid()
{
   //

   TGWindow *root = (TGWindow*)fClient->GetRoot();

   if (fPimpl->fGrid->fgStep > 1) {
      fPimpl->fGrid->SetStep(1);
      if (fBuilder) {
         fBuilder->UpdateStatusBar("Grid switched OFF");
      }
   } else {
      fPimpl->fGrid->SetStep(gGridStep);

      if (fBuilder) {
         fBuilder->UpdateStatusBar("Grid switched ON");
      }

      if (root->InheritsFrom(TGCompositeFrame::Class())) {
         TGCompositeFrame *comp = (TGCompositeFrame*)root;
         TIter next(comp->GetList());
         TGFrameElement *fe;
         Int_t x, y, w, h;

         while ((fe = (TGFrameElement*)next())) {
            x = fe->fFrame->GetX();
            y = fe->fFrame->GetY();
            w = fe->fFrame->GetWidth();
            h = fe->fFrame->GetHeight();
            ToGrid(x, y);
            ToGrid(w, h);
            fe->fFrame->MoveResize(x, y, w, h);
         }
      }
   }

   root->SetEditable(kFALSE);
   DoRedraw();
   root->SetEditable(kTRUE);

   DrawGrabRectangles();
}

//______________________________________________________________________________
TGCompositeFrame *TGuiBldDragManager::FindLayoutFrame(TGFrame *f)
{
   //

   if (!f) return 0;

   const TGWindow *parent = f->GetParent();
   TGCompositeFrame *ret = 0;

   while (parent && (parent != fClient->GetDefaultRoot())) {
      ret = (TGCompositeFrame*)parent;
      if (parent->InheritsFrom(TGMdiFrame::Class())) return ret;
      parent = parent->GetParent();
   }
   return ret;
}

//______________________________________________________________________________
void TGuiBldDragManager::HandleUpdateSelected(TGFrame *f)
{
   //

   if (!f) return;

   TGCompositeFrame *main = FindLayoutFrame(f);
   //Bool_t mainsav = main->IsLayoutBroken();
   TGWindow *root = (TGWindow*)fClient->GetRoot();
   root->SetEditable(kFALSE);

   TGFrame *parent = 0;
   if (f->GetParent()->InheritsFrom(TGFrame::Class())) {
      parent = (TGFrame*)f->GetParent();
   }

   Bool_t sav = kFALSE;
   if (parent && (parent != main)) {
      sav = parent->IsLayoutBroken();
      if (sav) {
         parent->SetLayoutBroken(kFALSE);
      }
   }
   main->SetLayoutBroken(kFALSE);
   main->Layout();
   fClient->NeedRedraw(f);
   fClient->NeedRedraw(main);

   if (sav) parent->SetLayoutBroken(kTRUE);
   //if (mainsav) main->SetLayoutBroken(kTRUE);

   root->SetEditable(kTRUE);
   DrawGrabRectangles();
}

//______________________________________________________________________________
void TGuiBldDragManager::HideGrabRectangles()
{
   //

   if (fPimpl->fGrabRectHidden) return;
   int i = 0;
   for (i = 0; i < 8; i++) fPimpl->fGrabRect[i]->UnmapWindow();
   for (i = 0; i < 4; i++) fPimpl->fAroundFrame[i]->UnmapWindow();
   fPimpl->fGrabRectHidden = kTRUE;
}

//______________________________________________________________________________
void TGuiBldDragManager::DeletePropertyEditor()
{

   if (!fEditor) return;

   TQObject::Disconnect(fEditor);

   delete fEditor;
   fEditor = 0;
}

//______________________________________________________________________________
Int_t TGuiBldDragManager::GetStrartDragX() const
{
   //

   return fPimpl->fX0;
}

//______________________________________________________________________________
Int_t TGuiBldDragManager::GetStrartDragY() const
{
   //

   return fPimpl->fY0;
}

//______________________________________________________________________________
Int_t TGuiBldDragManager::GetEndDragX() const
{
   //

   return fPimpl->fY;
}

//______________________________________________________________________________
Int_t TGuiBldDragManager::GetEndDragY() const
{
   //

   return fPimpl->fY;
}

//______________________________________________________________________________
void TGuiBldDragManager::ExecuteQuickAction(Event_t *event)
{
   //

   TGWindow *win = gClient->GetWindowById(event->fWindow);
   if (!win) return;

   if (!fQuickHandler) fQuickHandler = new TGuiBldQuickHandler();

   if (!fQuickHandler || !fQuickHandler->HandleEvent(win)) {
      if (win->IsEditable()) {
         TGFrame *f = (TGFrame*)win;
         TGDimension sz = f->GetDefaultSize();
         if ((sz.fWidth > 10) && (sz.fHeight > 10)) {
            //Compact(kTRUE);
         }
      }
   }
}

//______________________________________________________________________________
void TGuiBldDragManager::BreakLayout()
{
   //

   if (!fPimpl->fGrab) return;

   fPimpl->fGrab->SetLayoutBroken(kTRUE);

   if (fBuilder) {
      TString str = fPimpl->fGrab->ClassName();
      str += "::";
      str += fPimpl->fGrab->GetName();
      str += " Layout Broken";
      fBuilder->UpdateStatusBar(str.Data());
   }

   Int_t  x = fPimpl->fGrab->GetX();
   Int_t  y = fPimpl->fGrab->GetY();
   UInt_t w = fPimpl->fGrab->GetWidth();
   UInt_t h = fPimpl->fGrab->GetHeight();
   const TGGC *gc = fClient->GetResourcePool()->GetSelectedBckgndGC();

   DrawGrabRectangles();

   gVirtualX->DrawRectangle(fPimpl->fGrab->GetParent()->GetId(),
                            gc->GetGC(), x, y, w, h);
   gVirtualX->DrawRectangle(fPimpl->fGrab->GetParent()->GetId(),
                            gc->GetGC(), x-2, y-2, w+2, h+2);

   TTimer::SingleShot(500, "TGuiBldDragManager", this, "DoRedraw()");
}

//______________________________________________________________________________
void TGuiBldDragManager::SwitchLayout()
{
   //

   if (!fPimpl->fGrab || !fPimpl->fGrab->InheritsFrom(TGCompositeFrame::Class())) return;

   TGCompositeFrame *comp = (TGCompositeFrame*)fPimpl->fGrab;

   comp->SetLayoutBroken(kFALSE);

   UInt_t opt = comp->GetOptions();
   TGLayoutManager *m = comp->GetLayoutManager();

   if (m->InheritsFrom(TGHorizontalLayout::Class())) {
      opt &= ~kHorizontalFrame;
      opt |= kVerticalFrame;

      if (fBuilder) {
         TString str = comp->ClassName();
         str += "::";
         str += comp->GetName();
         str += " Vertical Layout ON";
         fBuilder->UpdateStatusBar(str.Data());
      }
   } else if (m->InheritsFrom(TGVerticalLayout::Class())) {
      opt &= ~kVerticalFrame;
      opt |= kHorizontalFrame;

      if (fBuilder) {
         TString str = comp->ClassName();
         str += "::";
         str += comp->GetName();
         str += " Horizontal Layout ON";
         fBuilder->UpdateStatusBar(str.Data());
      }
   }
   comp->ChangeOptions(opt);
   comp->Resize();
}

//______________________________________________________________________________
TGFrame *TGuiBldDragManager::GetSelected() const
{
   //

   return (fPimpl ?  fPimpl->fGrab : 0);
}

//______________________________________________________________________________
void TGuiBldDragManager::CloseMenus()
{
   //

   void *ud;
   if (fFrameMenu) fFrameMenu->EndMenu(ud);
   if (fLassoMenu) fLassoMenu->EndMenu(ud);
}
