// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGPack.h"
#include "TGSplitter.h"
#include "TMath.h"


/** \class TGPack
    \ingroup guiwidgets

Stack of frames in horizontal (default) or vertical stack.
The splitters are placed between the neighbouring frames so that
they can be resized by the user.
When the whole pack is resized, frames are scaled proportionally to
their previous size.

When frames are left in pack at destruction time, they will be
deleted via local-cleanup.

*/


ClassImp(TGPack);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGPack::TGPack(const TGWindow *p, UInt_t w, UInt_t h, UInt_t options, Pixel_t back) :
   TGCompositeFrame(p, w, h, options, back),
   fVertical     (kTRUE),
   fUseSplitters (kTRUE),
   fSplitterLen  (4),
   fDragOverflow (0),
   fWeightSum(0),
   fNVisible(0)
{
   SetCleanup(kLocalCleanup);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGPack::TGPack(TGClient *c, Window_t id, const TGWindow *parent) :
   TGCompositeFrame(c, id, parent),
   fVertical     (kTRUE),
   fUseSplitters (kTRUE),
   fSplitterLen  (4),
   fDragOverflow (0),
   fWeightSum    (0.0),
   fNVisible     (0)
{
   SetCleanup(kLocalCleanup);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGPack::~TGPack()
{
}

//------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
/// Return length of entire frame without splitters.

Int_t TGPack::GetAvailableLength() const
{
   Int_t len = fVertical ? GetHeight() : GetWidth();
   len -= fSplitterLen * (fNVisible - 1);

   return len;
}

////////////////////////////////////////////////////////////////////////////////
/// Set pack-wise length of frame f.

void TGPack::SetFrameLength(TGFrame* f, Int_t len)
{
   if (fVertical)
      f->Resize(GetWidth(), len);
   else
      f->Resize(len, GetHeight());
}

////////////////////////////////////////////////////////////////////////////////
/// Set pack-wise position of frame f.

void TGPack::SetFramePosition(TGFrame* f, Int_t pos)
{
   if (fVertical)
      f->Move(0, pos);
   else
      f->Move(pos, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if splitter of first visible frame is hidden.
/// Check if the next following visible splitter is visible.

void TGPack::CheckSplitterVisibility()
{
   TGFrameElementPack *el;
   TIter next(fList);
   Int_t rvf = 0;
   while ((el = (TGFrameElementPack*) next())) {
      if (el->fState && el->fSplitFE) {
         if (rvf) {
            // unmap first slider if necessary
            if ( el->fSplitFE->fState == 0 ) {
               el->fSplitFE->fState = 1;
               el->fSplitFE->fFrame->MapWindow();
            }
         } else {
            // show slider in next visible frame
            if (el->fSplitFE->fState) {
               el->fSplitFE->fState = 0;
               el->fSplitFE->fFrame->UnmapWindow();
            }
         }
         ++rvf;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Resize (shrink or expand) existing frames by amount in total.

void TGPack::ResizeExistingFrames()
{
   if (fList->IsEmpty())
      return;

   // get unitsize
   Int_t nflen  = GetAvailableLength();
   Float_t unit = Float_t(nflen)/fWeightSum;

   // set frame sizes
   Int_t sumFrames = 0;
   Int_t frameLength = 0;
   {
      TGFrameElementPack *el;
      TIter next(fList);
      while ((el = (TGFrameElementPack*) next())) {
         if (el->fState) {
            frameLength = TMath::Nint( unit*(el->fWeight));
            SetFrameLength(el->fFrame, frameLength);
            sumFrames += frameLength;
         }
      }
   }

   // redistribute the remain
   {
      // printf("available %d total %d \n", nflen, sumFrames);
      Int_t remain =  nflen-sumFrames;
      Int_t step = TMath::Sign(1, remain);
      TGFrameElementPack *el;
      TIter next(fList);
      while ((el = (TGFrameElementPack*) next()) && remain) {
         if (el->fState) {
            Int_t l = GetFrameLength(el->fFrame) + step;
            if (l > 0) {
               SetFrameLength(el->fFrame, l);
               remain -= step;
            }
         }
      }
   }
   RefitFramesToPack();
}

////////////////////////////////////////////////////////////////////////////////
/// Refit existing frames to pack size.

void TGPack::RefitFramesToPack()
{
   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      if (fVertical)
         el->fFrame->Resize(GetWidth(), el->fFrame->GetHeight());
      else
         el->fFrame->Resize(el->fFrame->GetWidth(), GetHeight());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Find frames around splitter and return them f0 (previous) and f1 (next).

void TGPack::FindFrames(TGFrame* splitter, TGFrameElementPack*& f0, TGFrameElementPack*& f1) const
{
   TGFrameElementPack *el;
   TIter next(fList);

   while ((el = (TGFrameElementPack *) next())) {
      if ( ! (el->fState & kIsVisible) )
         continue;

      if (el->fFrame == splitter)
         break;
      f0 = el;
   }
   f1 = (TGFrameElementPack *) next();
}

////////////////////////////////////////////////////////////////////////////////
/// Add frame f at the end.
/// LayoutHints are ignored in TGPack.

void TGPack::AddFrameInternal(TGFrame* f, TGLayoutHints* l, Float_t weight)
{
   // add splitter
   TGFrameElementPack *sf = 0;
   if (fUseSplitters) {
      TGSplitter* s = 0;
      if (fVertical)
         s = new TGHSplitter(this, GetWidth(), fSplitterLen, kTRUE);
      else
         s = new TGVSplitter(this, fSplitterLen, GetHeight(), kTRUE);
      s->Connect("Moved(Int_t)",  "TGPack", this, "HandleSplitterResize(Int_t)");
      s->Connect("DragStarted()", "TGPack", this, "HandleSplitterStart()");

      sf = new TGFrameElementPack(s, l ? l : fgDefaultHints, 0);
      fList->Add(sf);
      // in case of recursive cleanup, propagate cleanup setting to all
      // child composite frames
      if (fMustCleanup == kDeepCleanup)
         s->SetCleanup(kDeepCleanup);
      s->MapWindow();
   }

   // instead TGCopositeFrame::AddFrame
   TGFrameElementPack *el = new TGFrameElementPack(f, l ? l : fgDefaultHints, weight);
   el->fSplitFE = sf;
   fList->Add(el);

   // in case of recursive cleanup, propagate cleanup setting to all
   // child composite frames
   if (fMustCleanup == kDeepCleanup)
      f->SetCleanup(kDeepCleanup);
   f->MapWindow();

   fNVisible ++;
   fWeightSum += weight;

   CheckSplitterVisibility();
   ResizeExistingFrames();
}

////////////////////////////////////////////////////////////////////////////////
/// Add frame f at the end with given weight.
/// LayoutHints are ignored in TGPack.

void TGPack::AddFrameWithWeight(TGFrame* f, TGLayoutHints *l, Float_t weight)
{
   AddFrameInternal(f, l, weight);
   Layout();
}

////////////////////////////////////////////////////////////////////////////////
/// Add frame f at the end with default weight.
/// LayoutHints are ignored in TGPack.

void TGPack::AddFrame(TGFrame* f, TGLayoutHints *l)
{
   AddFrameInternal(f, l, 1);
   Layout();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove frame f.

void TGPack::RemoveFrameInternal(TGFrame* f)
{
   TGFrameElementPack *el = (TGFrameElementPack*)FindFrameElement(f);

   if (!el) return;

   if (fUseSplitters) {
      TGFrame* splitter = el->fSplitFE->fFrame;
      splitter->UnmapWindow();
      TGCompositeFrame::RemoveFrame(splitter);
      // This is needed so that splitter window gets destroyed on server.
      splitter->ReparentWindow(fClient->GetDefaultRoot());
      delete splitter;
   }
   if (el->fState & kIsVisible) {
      f->UnmapWindow();
      fWeightSum -= el->fWeight;
      --fNVisible;
   }
   TGCompositeFrame::RemoveFrame(f);

   CheckSplitterVisibility();
   ResizeExistingFrames();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove frame f and refit existing frames to pack size.
/// Frame is deleted.

void TGPack::DeleteFrame(TGFrame* f)
{
   RemoveFrameInternal(f);
   delete f;
   Layout();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove frame f and refit existing frames to pack size.
/// Frame is not deleted.

void TGPack::RemoveFrame(TGFrame* f)
{
   RemoveFrameInternal(f);
   Layout();
}

////////////////////////////////////////////////////////////////////////////////
/// Print sub frame info.

void TGPack::Dump() const
{
   printf("--------------------------------------------------------------\n");
   Int_t cnt = 0;
   TGFrameElementPack *el;
   TIter next(fList);
   while ((el = (TGFrameElementPack *) next())) {
      printf("idx[%d] visible(%d) %s  \n",cnt, el->fState, el->fFrame->GetName());
      cnt++;
   }
   printf("--------------------------------------------------------------\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Show sub frame.
/// Virtual from TGCompositeFrame.

void TGPack::ShowFrame(TGFrame* f)
{
   TGFrameElementPack *el = (TGFrameElementPack*)FindFrameElement(f);
   if (el) {
      //show
      el->fState = 1;
      el->fFrame->MapWindow();

      // show splitter
      if (fUseSplitters) {
         el->fSplitFE->fFrame->MapWindow();
         el->fSplitFE->fState = 1;
      }

      // Dump();
      fNVisible++;
      fWeightSum += el->fWeight;

      CheckSplitterVisibility();
      ResizeExistingFrames();
      Layout();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Hide sub frame.
/// Virtual from TGCompositeFrame.

void TGPack::HideFrame(TGFrame* f)
{
   TGFrameElementPack *el = (TGFrameElementPack*) FindFrameElement(f);
   if (el) {
      // hide real frame
      el->fState = 0;
      el->fFrame->UnmapWindow();

      // hide splitter
      if (fUseSplitters) {
         el->fSplitFE->fFrame->UnmapWindow();
         el->fSplitFE->fState = 0;
      }

      // Dump();
      fNVisible--;
      fWeightSum -= el->fWeight;

      CheckSplitterVisibility();
      ResizeExistingFrames();
      Layout();
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Virtual method of TGcompositeFrame.
/// Map all sub windows that are part of the composite frame.

void TGPack::MapSubwindows()
{
   if (!fMapSubwindows) {
      return;
   }

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      if (el->fFrame && el->fState) {
         el->fFrame->MapWindow();
         el->fFrame->MapSubwindows();
         TGFrameElement *fe = el->fFrame->GetFrameElement();
         if (fe) fe->fState |= kIsVisible;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Resize the pack.
/// Contents is resized proportionally.

void TGPack::Resize(UInt_t w, UInt_t h)
{
   if (w == fWidth && h == fHeight) return;

   fWidth  = w;
   fHeight = h;
   TGWindow::Resize(fWidth, fHeight);

   ResizeExistingFrames();

   Layout();
}

////////////////////////////////////////////////////////////////////////////////
/// Move and resize the pack.

void TGPack::MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   TGCompositeFrame::Move(x, y);
   Resize(w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Reposition the frames so that they fit correctly.
/// LayoutHints are ignored.

void TGPack::Layout()
{
   Int_t pos = 0;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      if (el->fState) {
         SetFramePosition(el->fFrame, pos);
         pos += GetFrameLength(el->fFrame);
         el->fFrame->Layout();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Refit existing frames so that their lengths are equal.

void TGPack::EqualizeFrames()
{
   if (fList->IsEmpty())
      return;

   fWeightSum = 0;
   TGFrameElementPack *el;
   TIter next(fList);
   while ((el = (TGFrameElementPack *) next())) {
      el->fWeight = 1;
      if (el->fState)
         fWeightSum ++;
   }

   ResizeExistingFrames();
   Layout();
}

////////////////////////////////////////////////////////////////////////////////
/// Called when splitter drag starts.

void TGPack::HandleSplitterStart()
{
   fDragOverflow = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle resize events from splitters.

void TGPack::HandleSplitterResize(Int_t delta)
{
   Int_t available = GetAvailableLength();
   Int_t min_dec = - (available + fNVisible*2 -1);
   if (delta <  min_dec)
      delta = min_dec;

   TGSplitter *s = dynamic_cast<TGSplitter*>((TGFrame*) gTQSender);

   TGFrameElementPack *f0 = nullptr, *f1 = nullptr;
   FindFrames(s, f0, f1);
   if (!f0 || !f1)
      return;

   if (fDragOverflow < 0) {
      fDragOverflow += delta;
      if (fDragOverflow > 0) {
         delta = fDragOverflow;
         fDragOverflow = 0;
      } else {
         return;
      }
   } else if (fDragOverflow > 0) {
      fDragOverflow += delta;
      if (fDragOverflow < 0) {
         delta = fDragOverflow;
         fDragOverflow = 0;
      } else {
         return;
      }
   }

   Int_t l0 = GetFrameLength(f0->fFrame);
   Int_t l1 = GetFrameLength(f1->fFrame);
   if (delta < 0) {
      if (l0 - 1 < -delta) {
         fDragOverflow += delta + l0 - 1;
         delta = -l0 + 1;
      }
   } else {
      if (l1 - 1 < delta) {
         fDragOverflow += delta - l1 + 1;
         delta = l1 - 1;
      }
   }
   l0 += delta;
   l1 -= delta;
   SetFrameLength(f0->fFrame, l0);
   SetFrameLength(f1->fFrame, l1);
   Float_t weightDelta = Float_t(delta)/available;
   weightDelta *= fWeightSum;
   f0->fWeight += weightDelta;
   f1->fWeight -= weightDelta;

   ResizeExistingFrames();
   Layout();
}


////////////////////////////////////////////////////////////////////////////////
/// Sets the vertical flag and reformats the back to new stacking
/// direction.

void TGPack::SetVertical(Bool_t x)
{
   if (x == fVertical)
      return;

   TList list;
   while ( ! fList->IsEmpty()) {
      TGFrameElement *el = (TGFrameElement*) fList->At(1);
      TGFrame        *f  = el->fFrame;
      if ( ! (el->fState & kIsVisible) )
         f->SetBit(kTempFrame);
      RemoveFrameInternal(f);
      list.Add(f);
   }
   fVertical = x;
   while ( ! list.IsEmpty()) {
      TGFrame* f = (TGFrame*) list.First();
      AddFrameInternal(f);
      if (f->TestBit(kTempFrame)) {
         f->ResetBit(kTempFrame);
         HideFrame(f);
      }
      list.RemoveFirst();
   }
   Layout();
}
