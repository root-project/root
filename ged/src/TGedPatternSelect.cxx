// @(#)root/ged:$Name:  $:$Id: TGedPatternSelect.cxx,v 1.0 2003/08/28 11:55:31 rdm Exp $
// Author: Marek Biskup, Ilka Antcheva   22/07/03
// ****It needs more fixes*****
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

#include "TGClient.h"
#include "TGedPatternSelect.h"
#include "TGResourcePool.h"
#include "TGToolTip.h"
#include "TGButton.h"

ClassImp(TGedPopup)
ClassImp(TGedSelect)
ClassImp(TGedPatternFrame)
ClassImp(TGedPatternSelector)
ClassImp(TGedPatternPopup)
ClassImp(TGedPatternSelect)

TGGC* TGedPatternFrame::fgGC = 0;
   
const char p_bits[26][32] = {
   {
      0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55,
      0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55,
      0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55
   },  //0
   {
      0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55,
      0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55,
      0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55
   },  //1
   {
      0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11,
      0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11,
      0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11
   },  //2
   {
      0x00, 0x00, 0x44, 0x44, 0x00, 0x00, 0x11, 0x11, 0x00, 0x00, 0x44, 0x44,
      0x00, 0x00, 0x11, 0x11, 0x00, 0x00, 0x44, 0x44, 0x00, 0x00, 0x11, 0x11,
      0x00, 0x00, 0x44, 0x44, 0x00, 0x00, 0x11, 0x11
   }, //3
   {
      0x80, 0x80, 0x40, 0x40, 0x20, 0x20, 0x10, 0x10, 0x08, 0x08, 0x04, 0x04,
      0x02, 0x02, 0x01, 0x01, 0x80, 0x80, 0x40, 0x40, 0x20, 0x20, 0x10, 0x10,
      0x08, 0x08, 0x04, 0x04, 0x02, 0x02, 0x01, 0x01
   }, //4
   {
      0x20, 0x20, 0x40, 0x40, 0x80, 0x80, 0x01, 0x01, 0x02, 0x02, 0x04, 0x04,
      0x08, 0x08, 0x10, 0x10, 0x20, 0x20, 0x40, 0x40, 0x80, 0x80, 0x01, 0x01,
      0x02, 0x02, 0x04, 0x04, 0x08, 0x08, 0x10, 0x10
   }, //5 
   {
      0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
      0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
      0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44
   }, //6
   {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff
   }, //7
   {
      0x11, 0x11, 0xb8, 0xb8, 0x7c, 0x7c, 0x3a, 0x3a, 0x11, 0x11, 0xa3, 0xa3,
      0xc7, 0xc7, 0x8b, 0x8b, 0x11, 0x11, 0xb8, 0xb8, 0x7c, 0x7c, 0x3a, 0x3a,
      0x11, 0x11, 0xa3, 0xa3, 0xc7, 0xc7, 0x8b, 0x8b
   }, //8
   {
      0x10, 0x10, 0x10, 0x10, 0x28, 0x28, 0xc7, 0xc7, 0x01, 0x01, 0x01, 0x01,
      0x82, 0x82, 0x7c, 0x7c, 0x10, 0x10, 0x10, 0x10, 0x28, 0x28, 0xc7, 0xc7,
      0x01, 0x01, 0x01, 0x01, 0x82, 0x82, 0x7c, 0x7c
   }, //9
   {
      0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0xff, 0xff, 0x01, 0x01, 0x01, 0x01,
      0x01, 0x01, 0xff, 0xff, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0xff, 0xff,
      0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0xff, 0xff
   }, //10
   {
      0x08, 0x08, 0x49, 0x49, 0x2a, 0x2a, 0x1c, 0x1c, 0x2a, 0x2a, 0x49, 0x49,
      0x08, 0x08, 0x00, 0x00, 0x80, 0x80, 0x94, 0x94, 0xa2, 0xa2, 0xc1, 0xc1,
      0xa2, 0xa2, 0x94, 0x94, 0x80, 0x80, 0x00, 0x00
   }, //11 
   {
      0x1c, 0x1c, 0x22, 0x22, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x22, 0x22,
      0x1c, 0x1c, 0x00, 0x00, 0xc1, 0xc1, 0x22, 0x22, 0x14, 0x14, 0x14, 0x14,
      0x14, 0x14, 0x22, 0x22, 0xc1, 0xc1, 0x00, 0x00
   }, //12
   {
      0x01, 0x01, 0x82, 0x82, 0x44, 0x44, 0x28, 0x28, 0x10, 0x10, 0x28, 0x28,
      0x44, 0x44, 0x82, 0x82, 0x01, 0x01, 0x82, 0x82, 0x44, 0x44, 0x28, 0x28,
      0x10, 0x10, 0x28, 0x28, 0x44, 0x44, 0x82, 0x82
   }, //13
   {
      0xff, 0xff, 0x11, 0x10, 0x11, 0x10, 0x11, 0x10, 0xf1, 0x1f, 0x11, 0x11,
      0x11, 0x11, 0x11, 0x11, 0xff, 0x11, 0x01, 0x11, 0x01, 0x11, 0x01, 0x11,
      0xff, 0xff, 0x01, 0x10, 0x01, 0x10, 0x01, 0x10
   }, //14
   {
      0x22, 0x22, 0x55, 0x55, 0x22, 0x22, 0x00, 0x00, 0x88, 0x88, 0x55, 0x55,
      0x88, 0x88, 0x00, 0x00, 0x22, 0x22, 0x55, 0x55, 0x22, 0x22, 0x00, 0x00,
      0x88, 0x88, 0x55, 0x55, 0x88, 0x88, 0x00, 0x00
   }, //15
   {
      0x0e, 0x0e, 0x11, 0x11, 0xe0, 0xe0, 0x00, 0x00, 0x0e, 0x0e, 0x11, 0x11,
      0xe0, 0xe0, 0x00, 0x00, 0x0e, 0x0e, 0x11, 0x11, 0xe0, 0xe0, 0x00, 0x00,
      0x0e, 0x0e, 0x11, 0x11, 0xe0, 0xe0, 0x00, 0x00
   }, //16
   {
      0x44, 0x44, 0x22, 0x22, 0x11, 0x11, 0x00, 0x00, 0x44, 0x44, 0x22, 0x22,
      0x11, 0x11, 0x00, 0x00, 0x44, 0x44, 0x22, 0x22, 0x11, 0x11, 0x00, 0x00,
      0x44, 0x44, 0x22, 0x22, 0x11, 0x11, 0x00, 0x00
   }, //17
   {
      0x11, 0x11, 0x22, 0x22, 0x44, 0x44, 0x00, 0x00, 0x11, 0x11, 0x22, 0x22,
      0x44, 0x44, 0x00, 0x00, 0x11, 0x11, 0x22, 0x22, 0x44, 0x44, 0x00, 0x00,
      0x11, 0x11, 0x22, 0x22, 0x44, 0x44, 0x00, 0x00
   }, //18
   {
      0xe0, 0x03, 0x98, 0x0c, 0x84, 0x10, 0x42, 0x21, 0x42, 0x21, 0x21, 0x42,
      0x19, 0x4c, 0x07, 0xf0, 0x19, 0x4c, 0x21, 0x42, 0x42, 0x21, 0x42, 0x21,
      0x84, 0x10, 0x98, 0x0c, 0xe0, 0x03, 0x80, 0x00
   }, //19
   {
      0x22, 0x22, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x22, 0x22, 0x44, 0x44,
      0x44, 0x44, 0x44, 0x44, 0x22, 0x22, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
      0x22, 0x22, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44
   }, //20
   {
      0xf1, 0xf1, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f, 0x1f, 0x01, 0x01,
      0x01, 0x01, 0x01, 0x01, 0xf1, 0xf1, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
      0x1f, 0x1f, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01
   }, //21
   {
      0x8f, 0x8f, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0xf8, 0xf8, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x8f, 0x8f, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
      0xf8, 0xf8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
   }, //22
   {
      0xAA, 0xAA, 0x55, 0x55, 0x6a, 0x6a, 0x74, 0x74, 0x78, 0x78, 0x74, 0x74,
      0x6a, 0x6a, 0x55, 0x55, 0xAA, 0xAA, 0x55, 0x55, 0x6a, 0x6a, 0x74, 0x74,
      0x78, 0x78, 0x74, 0x74, 0x6a, 0x6a, 0x55, 0x55
   }, //23
   {
      0x80, 0x00, 0xc0, 0x00, 0xea, 0xa8, 0xd5, 0x54, 0xea, 0xa8, 0xd5, 0x54,
      0xeb, 0xe8, 0xd5, 0xd4, 0xe8, 0xe8, 0xd4, 0xd4, 0xa8, 0xe8, 0x54, 0xd5,
      0xa8, 0xea, 0x54, 0xd5, 0xfc, 0xff, 0xfe, 0xff
   }, //24
   {
      0x80, 0x00, 0xc0, 0x00, 0xe0, 0x00, 0xf0, 0x00, 0xff, 0xf0, 0xff, 0xf0,
      0xfb, 0xf0, 0xf9, 0xf0, 0xf8, 0xf0, 0xf8, 0x70, 0xf8, 0x30, 0xff, 0xf0,
      0xff, 0xf8, 0xff, 0xfc, 0xff, 0xfe, 0xff, 0xff
   }, //25
};

//______________________________________________________________________________
TGedPatternFrame::TGedPatternFrame(const TGWindow *p, Style_t pattern, 
                                   int width, int height) 
   : TGFrame(p, width, height, kOwnBackground)
{
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
      fTip = new TGToolTip(fClient->GetRoot(), this, fTipText, 1000);
   else if (pattern == 0)
      fTip = new TGToolTip(fClient->GetRoot(), this, "0 - hollow", 1000);
   else // pattern == 1001
      fTip = new TGToolTip(fClient->GetRoot(), this, "1001 - solid", 1000);
  
   AddInput(kEnterWindowMask | kLeaveWindowMask); 

   if (!fgGC) {
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
   gVirtualX->DrawRectangle(fId, GetBckgndGC()(), 0, 0, fWidth - 1, fHeight - 1);
   Draw3dRectangle(kDoubleBorder | kSunkenFrame, 1, 1, fWidth - 2, fHeight - 2);
}

//______________________________________________________________________________
void TGedPatternFrame::DoRedraw()
{
   TGFrame::DoRedraw();

   if (fPattern > 3000 && fPattern < 3026) {
      SetFillStyle(fgGC, fPattern);
      gVirtualX->FillRectangle(fId, fgGC->GetGC(), 1, 1, fWidth - 3, fHeight - 3);
   }
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
         if (fasi >= 1 && fasi <=25)
            fillPattern = gVirtualX->CreateBitmap(gClient->GetRoot()->GetId(),
                                                  p_bits[fasi], 16, 16);
         else
            fillPattern = gVirtualX->CreateBitmap(gClient->GetRoot()->GetId(),
                                                  p_bits[2], 16, 16);
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
   SetLayoutManager(new TGTileLayout(this, 1));

   for (int i = 1; i <= 25; i++)
     fCe[i-1] = new TGedPatternFrame(this, 3000 + i);

   fCe[25] = new TGedPatternFrame(this, 0);
   fCe[26] = new TGedPatternFrame(this, 1001);

   for (int i = 0; i < 27; i++)
      AddFrame(fCe[i], new TGLayoutHints(kLHintsNoHints));

   fMsgWindow  = p;
   fActive = -1;
}

//______________________________________________________________________________
TGedPatternSelector::~TGedPatternSelector()
{
   Cleanup();
}

//______________________________________________________________________________
void TGedPatternSelector::SetActive(Int_t newat)
{
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
   gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab pointer
   UnmapWindow();
}

//______________________________________________________________________________
void TGedPopup::PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
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
                          kPointerMotionMask, kNone, kNone, 
                          fClient->GetResourcePool()->GetGrabCursor());
   gClient->WaitForUnmap(this);
//   EndPopup();
}

//______________________________________________________________________________
Bool_t TGedPopup::HandleButton(Event_t *event)
{

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
   fCurrentPattern = pattern;
   
   TGedPatternSelector *ps = new TGedPatternSelector(this);
   AddFrame(ps, new TGLayoutHints(kLHintsCenterX, 1, 1, 1, 1));

   MapSubwindows();
   Resize(ps->GetDefaultWidth() + 6, ps->GetDefaultHeight());
}

//______________________________________________________________________________
TGedPatternPopup::~TGedPatternPopup()
{
   Cleanup();
}

//______________________________________________________________________________
Bool_t TGedPatternPopup::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
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
   if (fPopup) 
      delete fPopup;
   delete fDrawGC;
}

//______________________________________________________________________________
Bool_t TGedSelect::HandleButton(Event_t *event)
{
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
            gVirtualX->TranslateCoordinates(fId, gClient->GetRoot()->GetId(),
                                            0, fHeight, ax, ay, wdummy);
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
   // Set state of widget. If kTRUE=enabled, kFALSE=disabled.

   SetFlags(kWidgetIsEnabled);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGedSelect::Disable()
{
   // Set state of widget. If kTRUE=enabled, kFALSE=disabled.

   ClearFlags(kWidgetIsEnabled);
   fClient->NeedRedraw(this);
}

//________________________________________________________________________________
void TGedSelect::DoRedraw()
{
   // Draws separator and arrow

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

      DrawTriangle(GetShadowGC()(), x, y);

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
   Point_t points[3];

   points[0].fX = x;
   points[0].fY = y;
   points[1].fX = x + 5;
   points[1].fY = y;
   points[2].fX = x + 2;
   points[2].fY = y + 3;

   gVirtualX->FillPolygon(fId, gc, points, 3);
}


//______________________________________________________________________________
TGedPatternSelect::TGedPatternSelect(const TGWindow *p, Style_t pattern, Int_t id) 
   : TGedSelect(p, id)
{
   fPattern = pattern;
   SetPopup(new TGedPatternPopup(gClient->GetRoot(), this, fPattern));
   SetPattern(fPattern);
}

//______________________________________________________________________________
Bool_t TGedPatternSelect::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   if (GET_MSG(msg) == kC_PATTERNSEL && GET_SUBMSG(msg) == kPAT_SELCHANGED)
   {
      SetPattern(parm2);
      SendMessage(fMsgWindow, MK_MSG(kC_PATTERNSEL, kPAT_SELCHANGED),
                 parm1, parm2);
      PatternSelected();
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGedPatternSelect::DoRedraw()
{
   TGedSelect::DoRedraw();

   Int_t  x, y;
   UInt_t w, h;

   if (IsEnabled()) { // pattern rectangle

      x = fBorderWidth + 2;
      y = fBorderWidth + 2;  // 1;
      h = fHeight - (fBorderWidth * 2) - 4;  // -3;  // 14
      w = h * 2;
      if (fState == kButtonDown) { ++x; ++y; }

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
   } else { // sunken rectangle

      x = fBorderWidth + 2;
      y = fBorderWidth + 2;  // 1;
      w = 42;
      h = fHeight - (fBorderWidth * 2) - 4;  // 3;
      Draw3dRectangle(kSunkenFrame, x, y, w, h);
   }
}

//______________________________________________________________________________
void TGedPatternSelect::SetPattern(Style_t pattern)
{
   // Set pattern.

   fPattern = pattern;
   gClient->NeedRedraw(this);
}
