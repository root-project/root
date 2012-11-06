// @(#)root/ged:$Id$
// Author: Marek Biskup, Ilka Antcheva   22/07/03

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedPatternFrame, TGedPatternSelector, TGedPatternPopup              //
// and TGedPatternColor                                                 //
//                                                                      //
// The TGedPatternFrame is a small frame with border showing            //
// a specific pattern (fill style.                                      //
//                                                                      //
// The TGedPatternSelector is a composite frame with TGedPatternFrames  //
// of all diferent styles                                               //
//                                                                      //
// The TGedPatternPopup is a popup containing a TGedPatternSelector.    //
//                                                                      //
// The TGedPatternSelect widget is a button with pattern area with      //
// a little down arrow. When clicked on the arrow the                   //
// TGedPatternPopup pops up.                                            //
//                                                                      //
// Selecting a pattern in this widget will generate the event:          //
// kC_PATTERNSEL, kPAT_SELCHANGED, widget id, style.                    //
//                                                                      //
// and the signal:                                                      //
// PatternSelected(Style_t pattern)                                     //
//                                                                      //
// TGedSelect is button that shows popup window when clicked.           //
// TGedPopup is a popup window.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGResourcePool.h"
#include "TGedPatternSelect.h"
#include "RConfigure.h"
#include "TGToolTip.h"
#include "TGButton.h"
#include "Riostream.h"
#include "RStipples.h"

ClassImp(TGedPopup)
ClassImp(TGedSelect)
ClassImp(TGedPatternFrame)
ClassImp(TGedPatternSelector)
ClassImp(TGedPatternPopup)
ClassImp(TGedPatternSelect)

TGGC* TGedPatternFrame::fgGC = 0;


//______________________________________________________________________________
TGedPatternFrame::TGedPatternFrame(const TGWindow *p, Style_t pattern,
                                   int width, int height)
   : TGFrame(p, width, height, kOwnBackground)
{
   // Pattern select ctor.

   Pixel_t white;
   gClient->GetColorByName("white", white); // white background
   SetBackgroundColor(white);

   // special case: solid
   if (pattern == 1001)
      SetBackgroundColor(0);     // if solid then black

   fPattern = pattern;

   AddInput(kButtonPressMask | kButtonReleaseMask);
   fMsgWindow  = p;
   fActive = kFALSE;
   snprintf(fTipText, 5, "%d", pattern);

   // solid and hollow must be treated separately
   if (pattern != 0 && pattern != 1001)
      fTip = new TGToolTip(fClient->GetDefaultRoot(), this, fTipText, 1000);
   else if (pattern == 0)
      fTip = new TGToolTip(fClient->GetDefaultRoot(), this, "0 - hollow", 1000);
   else // pattern == 1001
      fTip = new TGToolTip(fClient->GetDefaultRoot(), this, "1001 - solid", 1000);

   AddInput(kEnterWindowMask | kLeaveWindowMask);

   if (!fgGC) {
      GCValues_t gcv;
      gcv.fMask = kGCLineStyle  | kGCLineWidth  | kGCFillStyle |
                  kGCForeground | kGCBackground;
      gcv.fLineStyle  = kLineSolid;
      gcv.fLineWidth  = 0;
      gcv.fFillStyle  = 0;
      gcv.fBackground = white;
      gcv.fForeground = 0;    // black foreground
      fgGC = gClient->GetGC(&gcv, kTRUE);
   }
}

//______________________________________________________________________________
Bool_t TGedPatternFrame::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (fTip) {
      if (event->fType == kEnterNotify)
         fTip->Reset();
      else
         fTip->Hide();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGedPatternFrame::HandleButton(Event_t *event)
{
   // Handle mouse button event.

   if (event->fType == kButtonPress) {
      SendMessage(fMsgWindow, MK_MSG(kC_PATTERNSEL, kPAT_CLICK), event->fCode, fPattern);
   } else {    // kButtonRelease
      SendMessage(fMsgWindow, MK_MSG(kC_PATTERNSEL, kPAT_SELCHANGED), event->fCode, fPattern);
   }

   return kTRUE;
}

//______________________________________________________________________________
void TGedPatternFrame::DrawBorder()
{
   // Draw border.

   gVirtualX->DrawRectangle(fId, GetBckgndGC()(), 0, 0, fWidth, fHeight);
   Draw3dRectangle(kDoubleBorder | kSunkenFrame, 0, 0, fWidth, fHeight);
}

//______________________________________________________________________________
void TGedPatternFrame::DoRedraw()
{
   // Redraw selected pattern.

   TGFrame::DoRedraw();

   if (fPattern > 3000 && fPattern < 3026) {
      SetFillStyle(fgGC, fPattern);
      gVirtualX->FillRectangle(fId, fgGC->GetGC(), 0, 0, fWidth, fHeight);
   }
   DrawBorder();
}

//______________________________________________________________________________
void TGedPatternFrame::SetFillStyle(TGGC* gc, Style_t fstyle)
{
   // Set fill area style.
   // fstyle   : compound fill area interior style
   //    fstyle = 1000*interiorstyle + styleindex
   // this function should be in TGGC !!!

   Int_t style = fstyle/1000;
   Int_t fasi  = fstyle%1000;
   Int_t stn = (fasi >= 1 && fasi <=25) ? fasi : 2;

   static Pixmap_t fillPattern = 0;

   switch (style) {
      case 1:         // solid
         gc->SetFillStyle(kFillSolid);
         break;
      case 2:         // pattern
         break;
      case 3:         // hatch
         gc->SetFillStyle(kFillStippled);
         if (fillPattern != 0) {
            gVirtualX->DeletePixmap(fillPattern);
            fillPattern = 0;
         }
         fillPattern = gVirtualX->CreateBitmap(gClient->GetDefaultRoot()->GetId(),
                                               (const char*)gStipples[stn], 16, 16);
         gc->SetStipple(fillPattern);
         break;
      default:
        break;
   }
}

//______________________________________________________________________________
TGedPatternSelector::TGedPatternSelector(const TGWindow *p) :
   TGCompositeFrame(p, 124, 190)
{
   // Create pattern popup window.

   SetLayoutManager(new TGTileLayout(this, 1));

   Int_t i;
   for (i = 1; i <= 25; i++)
      fCe[i-1] = new TGedPatternFrame(this, 3000 + i);

   fCe[25] = new TGedPatternFrame(this, 0);
   fCe[26] = new TGedPatternFrame(this, 1001);

   for (i = 0; i < 27; i++)
      AddFrame(fCe[i], new TGLayoutHints(kLHintsNoHints));

   fMsgWindow  = p;
   fActive = -1;
}

//______________________________________________________________________________
TGedPatternSelector::~TGedPatternSelector()
{
   // Delete pattern popup window.

   Cleanup();
}

//______________________________________________________________________________
void TGedPatternSelector::SetActive(Int_t newat)
{
   // Set selected the current style.

   if (fActive != newat) {
      if ((fActive >= 0) && (fActive < 27)) {
         fCe[fActive]->SetActive(kFALSE);
      }
      fActive = newat;
      if ((fActive >= 0) && (fActive < 27)) {
         fCe[fActive]->SetActive(kTRUE);
      }
   }
}

//______________________________________________________________________________
Bool_t TGedPatternSelector::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Process message generated by pattern popup window.

   switch (GET_MSG(msg)) {
      case kC_PATTERNSEL:
         switch (GET_SUBMSG(msg)) {
            case kPAT_SELCHANGED:
               switch (parm1) {
                  case kButton1:
                     SendMessage(fMsgWindow, MK_MSG(kC_PATTERNSEL,
                                 kPAT_SELCHANGED), parm1, parm2);
                     break;
               }
               break;
            case kPAT_CLICK:
               switch (parm1) {
                  case kButton1:
                     SetActive(parm2);
                     break;
               }
               break;
         }
   }

   return kTRUE;
}

//______________________________________________________________________________
TGedPopup::TGedPopup(const TGWindow *p, const TGWindow *m, UInt_t w, UInt_t h,
                     UInt_t options, Pixel_t back)
   : TGCompositeFrame(p, w, h, options, back)
{
   // Create a popup frame.

   fMsgWindow = m;
   SetWindowAttributes_t wattr;

   wattr.fMask = kWAOverrideRedirect | kWASaveUnder ;
   wattr.fOverrideRedirect = kTRUE;
   wattr.fSaveUnder = kTRUE;
   gVirtualX->ChangeWindowAttributes(fId, &wattr);

   AddInput(kStructureNotifyMask);
}

//______________________________________________________________________________
void TGedPopup::EndPopup()
{
   // Ungrab pointer and unmap popup window.

   gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);
   UnmapWindow();
}

//______________________________________________________________________________
void TGedPopup::PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Place popup window at the specified place.

   Int_t rx, ry;
   UInt_t rw, rh;

   // Parent is root window for the popup:
   gVirtualX->GetWindowSize(fParent->GetId(), rx, ry, rw, rh);

   if (x < 0) x = 0;
   if (x + fWidth > rw) x = rw - fWidth;
   if (y < 0) y = 0;
   if (y + fHeight > rh) y = rh - fHeight;

   MoveResize(x, y, w, h);
   MapSubwindows();
   Layout();
   MapRaised();

   gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask | 
                          kPointerMotionMask, kNone,
                          fClient->GetResourcePool()->GetGrabCursor());
   gClient->WaitForUnmap(this);
   EndPopup();
}

//______________________________________________________________________________
Bool_t TGedPopup::HandleButton(Event_t *event)
{
   // Handle mouse button event in popup window.

   if ((event->fX < 0) || (event->fX >= (Int_t) fWidth) ||
       (event->fY < 0) || (event->fY >= (Int_t) fHeight))  {

      if (event->fType == kButtonRelease) EndPopup();

   } else {
      TGFrame *f = GetFrameFromPoint(event->fX, event->fY);
      if (f && f != this) {
         TranslateCoordinates(f, event->fX, event->fY, event->fX, event->fY);
         f->HandleButton(event);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGedPopup::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Process messages generated by popup window.

   switch (GET_MSG(msg)) {
      case kC_POPUP:
         switch (GET_SUBMSG(msg)) {
            case kPOP_HIDE:
               EndPopup();
               SendMessage(fMsgWindow, MK_MSG(kC_POPUP, kPOP_HIDE),
                           parm1, parm2);
               break;
            default:
               break;
         }
         break;
   }
   return kTRUE;
}

//______________________________________________________________________________
TGedPatternPopup::TGedPatternPopup(const TGWindow *p, const TGWindow *m, Style_t pattern)
   : TGedPopup(p, m, 10, 10, kDoubleBorder | kRaisedFrame | kOwnBackground,
               GetDefaultFrameBackground())
{
   // Pattern popup constructor.

   fCurrentPattern = pattern;

   TGedPatternSelector *ps = new TGedPatternSelector(this);
   AddFrame(ps, new TGLayoutHints(kLHintsCenterX, 1, 1, 1, 1));

   MapSubwindows();
   Resize(ps->GetDefaultWidth() + 6, ps->GetDefaultHeight());
}

//______________________________________________________________________________
TGedPatternPopup::~TGedPatternPopup()
{
   // Destructor of pattern popup window.

   Cleanup();
}

//______________________________________________________________________________
Bool_t TGedPatternPopup::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Process messages generated by pattern popup window.

   switch (GET_MSG(msg)) {
      case kC_PATTERNSEL:
         switch (GET_SUBMSG(msg)) {
            case kPAT_SELCHANGED:
               SendMessage(fMsgWindow, MK_MSG(kC_PATTERNSEL, kPAT_SELCHANGED),
                           parm1, parm2);
               UnmapWindow();
               break;

            default:
               break;
         }
         break;
   }
   return kTRUE;
}

//______________________________________________________________________________
TGedSelect::TGedSelect(const TGWindow *p, Int_t id)
   : TGCheckButton(p, "", id)
{
   // Create pattern select button.

   fPopup = 0;

   GCValues_t gcv;
   gcv.fMask = kGCLineStyle  | kGCLineWidth  | kGCFillStyle |
               kGCForeground | kGCBackground;
   gcv.fLineStyle  = kLineSolid;
   gcv.fLineWidth  = 0;
   gcv.fFillStyle  = 0;
   Pixel_t white;
   gClient->GetColorByName("white", white); // white background
   gcv.fBackground = white;
   gcv.fForeground = 0;    // black foreground
   fDrawGC = gClient->GetGC(&gcv, kTRUE);

   Enable();
   SetState(kButtonUp);
   AddInput(kButtonPressMask | kButtonReleaseMask);
}

//______________________________________________________________________________
TGedSelect::~TGedSelect()
{
   // Destructor of pattern select button.

   if (fPopup)
      delete fPopup;
   fClient->FreeGC(fDrawGC);
}

//______________________________________________________________________________
Bool_t TGedSelect::HandleButton(Event_t *event)
{
   // Handle mouse button events in pattern select button.

   TGFrame::HandleButton(event);

   if (!IsEnabled()) return kTRUE;

   if (event->fCode != kButton1) return kFALSE;

   if ((event->fType == kButtonPress) && HasFocus()) WantFocus();

   if (event->fType == kButtonPress) {
      if (fState != kButtonDown) {
         fPrevState = fState;
         SetState(kButtonDown);
      }
   } else {
      if (fState != kButtonUp) {
         SetState(kButtonUp);
         Window_t wdummy;
         Int_t ax, ay;
         if (fPopup) {
            gVirtualX->TranslateCoordinates(fId, gClient->GetDefaultRoot()->GetId(),
                                            0, fHeight, ax, ay, wdummy);
#ifdef R__HAS_COCOA
            gVirtualX->SetWMTransientHint(fPopup->GetId(), GetId());
#endif
            fPopup->PlacePopup(ax, ay, fPopup->GetDefaultWidth(),
                               fPopup->GetDefaultHeight());
         }
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGedSelect::Enable()
{
   // Set widget state flag (kTRUE=enabled, kFALSE=disabled).

   SetFlags(kWidgetIsEnabled);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGedSelect::Disable()
{
   // Clear widget state flag.

   ClearFlags(kWidgetIsEnabled);
   fClient->NeedRedraw(this);
}

//________________________________________________________________________________
void TGedSelect::DoRedraw()
{
   // Draw separator and arrow.

   Int_t  x, y;
   UInt_t h;

   TGButton::DoRedraw();

   if (IsEnabled()) {

      // separator
      x = fWidth - 6 - fBorderWidth - 6;
      y = fBorderWidth + 1;
      h = fHeight - fBorderWidth - 1;  // actually y1

      if (fState == kButtonDown) { ++x; ++y; }

      gVirtualX->DrawLine(fId, GetShadowGC()(),  x, y, x, h - 2);
      gVirtualX->DrawLine(fId, GetHilightGC()(), x + 1, y, x + 1, h - 1);
      gVirtualX->DrawLine(fId, GetHilightGC()(), x, h - 1, x + 1, h - 1);

      // arrow

      x = fWidth - 6 - fBorderWidth - 2;
      y = (fHeight - 4) / 2 + 1;

      if (fState == kButtonDown) { ++x; ++y; }

      DrawTriangle(GetBlackGC()(), x, y);

   } else {

      // separator
      x = fWidth - 6 - fBorderWidth - 6;
      y = fBorderWidth + 1;
      h = fHeight - fBorderWidth - 1;  // actually y1

      gVirtualX->DrawLine(fId, GetShadowGC()(),  x, y, x, h - 2);
      gVirtualX->DrawLine(fId, GetHilightGC()(), x + 1, y, x + 1, h - 1);
      gVirtualX->DrawLine(fId, GetHilightGC()(), x, h - 1, x + 1, h - 1);

      // sunken arrow

      x = fWidth - 6 - fBorderWidth - 2;
      y = (fHeight - 4) / 2 + 1;

      DrawTriangle(GetHilightGC()(), x + 1, y + 1);
      DrawTriangle(GetShadowGC()(), x, y);
   }
}

//______________________________________________________________________________
void TGedSelect::DrawTriangle(GContext_t gc, Int_t x, Int_t y)
{
   // Draw small triangle.

   Point_t points[3];
   
#ifdef R__HAS_COCOA
   points[0].fX = x;
   points[0].fY = y;
   points[1].fX = x + 6;
   points[1].fY = y;
   points[2].fX = x + 3;
   points[2].fY = y + 3;
#else
   points[0].fX = x;
   points[0].fY = y;
   points[1].fX = x + 5;
   points[1].fY = y;
   points[2].fX = x + 2;
   points[2].fY = y + 3;
#endif

   gVirtualX->FillPolygon(fId, gc, points, 3);
}


//______________________________________________________________________________
TGedPatternSelect::TGedPatternSelect(const TGWindow *p, Style_t pattern, Int_t id)
   : TGedSelect(p, id)
{
   // Create and pop up pattern select window.

   fPattern = pattern;
   SetPopup(new TGedPatternPopup(gClient->GetDefaultRoot(), this, fPattern));
   SetPattern(fPattern);
}

//______________________________________________________________________________
Bool_t TGedPatternSelect::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Process message according to the user input.

   if (GET_MSG(msg) == kC_PATTERNSEL && GET_SUBMSG(msg) == kPAT_SELCHANGED)
   {
      SetPattern(parm2);
      parm1 = (Long_t)fWidgetId;
      SendMessage(fMsgWindow, MK_MSG(kC_PATTERNSEL, kPAT_SELCHANGED),
                  parm1, parm2);
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGedPatternSelect::DoRedraw()
{
   // Draw selected pattern as current one.

   TGedSelect::DoRedraw();

   Int_t  x, y;
   UInt_t w, h;

   if (IsEnabled()) { // pattern rectangle

      x = fBorderWidth + 2;
      y = fBorderWidth + 2;  // 1;
      h = fHeight - (fBorderWidth * 2) - 4;  // -3;  // 14
      w = h * 2;
      if (fState == kButtonDown) { ++x; ++y; }

#ifdef R__HAS_COCOA
      TGedPatternFrame::SetFillStyle(fDrawGC, 1001);

      Pixel_t white;
      gClient->GetColorByName("white", white); // white background
      fDrawGC->SetForeground(white);
      gVirtualX->FillRectangle(fId, fDrawGC->GetGC(), x + 1, y + 1, w - 1, h - 1);

      if (fPattern != 0) {
         fDrawGC->SetForeground(0);
         TGedPatternFrame::SetFillStyle(fDrawGC, fPattern);
         gVirtualX->FillRectangle(fId, fDrawGC->GetGC(), x + 1, y + 1, w - 1, h - 1);
      }
      
      gVirtualX->DrawRectangle(fId, GetShadowGC()(), x + 1, y + 1, w - 1, h - 1);
#else
      gVirtualX->DrawRectangle(fId, GetShadowGC()(), x, y, w - 1, h - 1);

      TGedPatternFrame::SetFillStyle(fDrawGC, 1001);

      Pixel_t white;
      gClient->GetColorByName("white", white); // white background
      fDrawGC->SetForeground(white);
      gVirtualX->FillRectangle(fId, fDrawGC->GetGC(), x + 1, y + 1, w - 2, h - 2);

      if (fPattern != 0) {
         fDrawGC->SetForeground(0);
         TGedPatternFrame::SetFillStyle(fDrawGC, fPattern);
         gVirtualX->FillRectangle(fId, fDrawGC->GetGC(), x + 1, y + 1, w - 2, h - 2);
      }
#endif
   } else { // sunken rectangle

      x = fBorderWidth + 2;
      y = fBorderWidth + 2;  // 1;
      w = 42;
      h = fHeight - (fBorderWidth * 2) - 4;  // 3;
      Draw3dRectangle(kSunkenFrame, x, y, w, h);
   }
}

//______________________________________________________________________________
void TGedPatternSelect::SetPattern(Style_t pattern, Bool_t emit)
{
   // Set pattern.

   fPattern = pattern;
   gClient->NeedRedraw(this);
   if (emit)
      PatternSelected(fPattern);
}

//______________________________________________________________________________
void TGedPatternSelect::SavePrimitive(std::ostream &out, Option_t * /*= ""*/)
{
   // Save the pattern select widget as a C++ statement(s) on output stream out

   out <<"   TGedPatternSelect *";
   out << GetName() << " = new TGedPatternSelect(" << fParent->GetName()
       << "," << fPattern << "," << WidgetId() << ");" << std::endl;
}
