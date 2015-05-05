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

//______________________________________________________________________________
//
// Stack of frames in horizontal (default) or vertical stack.
// The splitters are placed between the neighbouring frames so that
// they can be resized by the user.
// When the whole pack is resized, frames are scaled proportionally to
// their previous size.
//
// When frames are left in pack at destruction time, they will be
// deleted via local-cleanup.

ClassImp(TGPack);

//______________________________________________________________________________
TGPack::TGPack(const TGWindow *p, UInt_t w, UInt_t h, UInt_t options, Pixel_t back) :
   TGCompositeFrame(p, w, h, options, back),
   fVertical     (kTRUE),
   fUseSplitters (kTRUE),
   fSplitterLen  (4),
   fDragOverflow (0),
   fWeightSum(0),
   fNVisible(0)
{
   // Constructor.

   SetCleanup(kLocalCleanup);
}

//______________________________________________________________________________
TGPack::TGPack(TGClient *c, Window_t id, const TGWindow *parent) :
   TGCompositeFrame(c, id, parent),
   fVertical     (kTRUE),
   fUseSplitters (kTRUE),
   fSplitterLen  (4),
   fDragOverflow (0),
   fWeightSum    (0.0),
   fNVisible     (0)
{
   // Constructor.

   SetCleanup(kLocalCleanup);
}

//______________________________________________________________________________
TGPack::~TGPack()
{
   // Destructor.
}

//------------------------------------------------------------------------------

//______________________________________________________________________________
Int_t TGPack::GetAvailableLength() const
{
   // Return length of entire frame without splitters.

   Int_t len = fVertical ? GetHeight() : GetWidth();
   len -= fSplitterLen * (fNVisible - 1);

   return len;
}

//______________________________________________________________________________
void TGPack::SetFrameLength(TGFrame* f, Int_t len)
{
   // Set pack-wise length of frame f.

   if (fVertical)
      f->Resize(GetWidth(), len);
   else
      f->Resize(len, GetHeight());
}

//______________________________________________________________________________
void TGPack::SetFramePosition(TGFrame* f, Int_t pos)
{
   // Set pack-wise position of frame f.

   if (fVertical)
      f->Move(0, pos);
   else
      f->Move(pos, 0);
}

//______________________________________________________________________________
void TGPack::CheckSplitterVisibility()
{
   // Check if splitter of first visible frame is hidden.
   // Check if the next following visible splitter is visible.

   TGFrameElementPack *el;
   TIter next(fList);
   Int_t rvf = 0;
   while ((el = (TGFrameElementPack*) next()))
   {
      if (el->fState && el->fSplitFE)
      {
         if (rvf)
         {
            // unmap first slider if necessary
            if ( el->fSplitFE->fState == 0 ) { 
               el->fSplitFE->fState = 1;
               el->fSplitFE->fFrame->MapWindow();
            }
         }
         else
         {
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

//______________________________________________________________________________
void TGPack::ResizeExistingFrames()
{
   // Resize (shrink or expand) existing frames by amount in total.

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
      while ((el = (TGFrameElementPack*) next()))
      {
         if (el->fState && el->fWeight)
         {
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
      while ((el = (TGFrameElementPack*) next()) && remain)
      {
         if (el->fState &&  el->fWeight)
         {
            Int_t l = GetFrameLength(el->fFrame) + step;
            if (l > 0)
            {
               SetFrameLength(el->fFrame, l);
               remain -= step;
            }
         }
      }
   }
   RefitFramesToPack();
}

//______________________________________________________________________________
void TGPack::RefitFramesToPack()
{
   // Refit existing frames to pack size.

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
   {
      if (fVertical)
         el->fFrame->Resize(GetWidth(), el->fFrame->GetHeight());
      else
         el->fFrame->Resize(el->fFrame->GetWidth(), GetHeight());
   }
}

//______________________________________________________________________________
void TGPack::FindFrames(TGFrame* splitter, TGFrameElementPack*& f0, TGFrameElementPack*& f1) const
{
   // Find frames around splitter and return them f0 (previous) and f1 (next).

   TGFrameElementPack *el;
   TIter next(fList);

   while ((el = (TGFrameElementPack *) next()))
   {
      if ( ! (el->fState & kIsVisible) )
         continue;

      if (el->fFrame == splitter)
         break;
      f0 = el;
   }
   f1 = (TGFrameElementPack *) next();
}


//------------------------------------------------------------------------------

//______________________________________________________________________________
void TGPack::AddFrameInternal(TGFrame* f, TGLayoutHints* l, Float_t weight)
{
   // Add frame f at the end.
   // LayoutHints are ignored in TGPack.

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
      // in case of recusive cleanup, propagate cleanup setting to all
      // child composite frames
      if (fMustCleanup == kDeepCleanup)
         s->SetCleanup(kDeepCleanup);
      s->MapWindow();
   }

   // instread TGCopositeFrame::AddFrame
   TGFrameElementPack *el = new TGFrameElementPack(f, l ? l : fgDefaultHints, weight);
   el->fSplitFE = sf;
   fList->Add(el);

   // in case of recusive cleanup, propagate cleanup setting to all
   // child composite frames
   if (fMustCleanup == kDeepCleanup)
      f->SetCleanup(kDeepCleanup);
   f->MapWindow();

   fNVisible ++;
   fWeightSum += weight;

   CheckSplitterVisibility();
   ResizeExistingFrames();
}

//______________________________________________________________________________
void TGPack::AddFrameWithWeight(TGFrame* f, TGLayoutHints *l, Float_t weight)
{
   // Add frame f at the end with given weight.
   // LayoutHints are ignored in TGPack.

   AddFrameInternal(f, l, weight);
   Layout();
}

//______________________________________________________________________________
void TGPack::AddFrame(TGFrame* f, TGLayoutHints *l)
{
   // Add frame f at the end with default weight.
   // LayoutHints are ignored in TGPack.

   AddFrameInternal(f, l, 1);
   Layout();
}

//______________________________________________________________________________
void TGPack::RemoveFrameInternal(TGFrame* f)
{
   // Remove frame f.

   TGFrameElementPack *el = (TGFrameElementPack*)FindFrameElement(f);

   if (!el) return;

   if (fUseSplitters)
   {
      TGFrame* splitter = el->fSplitFE->fFrame;
      splitter->UnmapWindow();
      TGCompositeFrame::RemoveFrame(splitter);
      // This is needed so that splitter window gets destroyed on server.
      splitter->ReparentWindow(fClient->GetDefaultRoot());
      delete splitter;
   }
   if (el->fState & kIsVisible)
   {
      f->UnmapWindow();
      fWeightSum -= el->fWeight;
      --fNVisible;
   }
   TGCompositeFrame::RemoveFrame(f);

   CheckSplitterVisibility();
   ResizeExistingFrames();
}

//______________________________________________________________________________
void TGPack::DeleteFrame(TGFrame* f)
{
   // Remove frame f and refit existing frames to pack size.
   // Frame is deleted.

   RemoveFrameInternal(f);
   delete f;
   Layout();
}

//______________________________________________________________________________
void TGPack::RemoveFrame(TGFrame* f)
{
   // Remove frame f and refit existing frames to pack size.
   // Frame is not deleted.

   RemoveFrameInternal(f);
   Layout();
}

//______________________________________________________________________________
void TGPack::Dump() const
{
   // Print sub frame info.

   printf("--------------------------------------------------------------\n");
   Int_t cnt = 0;
   TGFrameElementPack *el;
   TIter next(fList);
   while ((el = (TGFrameElementPack *) next()))
   {
      printf("idx[%d] visible(%d) %s  \n",cnt, el->fState, el->fFrame->GetName());
      cnt++;
   }
   printf("--------------------------------------------------------------\n");
}

//______________________________________________________________________________
void TGPack::ShowFrame(TGFrame* f)
{
   // Show sub frame.
   // Virtual from TGCompositeFrame.

   TGFrameElementPack *el = (TGFrameElementPack*)FindFrameElement(f);
   if (el)
   {
      //show
      el->fState = 1;
      el->fFrame->MapWindow();

      // show splitter
      if (fUseSplitters)
      {
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

//______________________________________________________________________________
void TGPack::HideFrame(TGFrame* f)
{
   // Hide sub frame.
   // Virtual from TGCompositeFrame.

   TGFrameElementPack *el = (TGFrameElementPack*) FindFrameElement(f);
   if (el)
   {
      // hide real frame
      el->fState = 0;
      el->fFrame->UnmapWindow();

      // hide splitter
      if (fUseSplitters)
      {
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

//------------------------------------------------------------------------------

//______________________________________________________________________________
void TGPack::MapSubwindows()
{
   // Virtual method of TGcompositeFrame.
   // Map all sub windows that are part of the composite frame.

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

//______________________________________________________________________________
void TGPack::Resize(UInt_t w, UInt_t h)
{
   // Resize the pack.
   // Contents is resized proportionally.

   if (w == fWidth && h == fHeight) return;

   fWidth  = w;
   fHeight = h;
   TGWindow::Resize(fWidth, fHeight);

   ResizeExistingFrames();

   Layout();
}

//______________________________________________________________________________
void TGPack::MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move and resize the pack.

   TGCompositeFrame::Move(x, y);
   Resize(w, h);
}

//______________________________________________________________________________
void TGPack::Layout()
{
   // Reposition the frames so that they fit correctly.
   // LayoutHints are ignored.

   Int_t pos = 0;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
   {
      if (el->fState)
      {
         SetFramePosition(el->fFrame, pos);
         pos += GetFrameLength(el->fFrame);
         el->fFrame->Layout();
      }
   }
}

//______________________________________________________________________________
void TGPack::EqualizeFrames()
{
   // Refit existing frames so that their lengths are equal.

   if (fList->IsEmpty())
      return;

   fWeightSum = 0;
   TGFrameElementPack *el;
   TIter next(fList);
   while ((el = (TGFrameElementPack *) next()))
   {
      el->fWeight = 1;
      if (el->fState)
         fWeightSum ++;
   }

   ResizeExistingFrames();
   Layout();
}

//______________________________________________________________________________
void TGPack::HandleSplitterStart()
{
   // Called when splitter drag starts.

   fDragOverflow = 0;
}

//______________________________________________________________________________
void TGPack::HandleSplitterResize(Int_t delta)
{
   // Handle resize events from splitters.

   Int_t available = GetAvailableLength();
   Int_t min_dec = - (available + fNVisible*2 -1);
   if (delta <  min_dec)
      delta = min_dec;

   TGSplitter *s = dynamic_cast<TGSplitter*>((TGFrame*) gTQSender);

   TGFrameElementPack *f0=0, *f1=0;
   FindFrames(s, f0, f1);

   if (fDragOverflow < 0)
   {
      fDragOverflow += delta;
      if (fDragOverflow > 0) {
         delta = fDragOverflow;
         fDragOverflow = 0;
      } else {
         return;
      }
   }
   else if (fDragOverflow > 0)
   {
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
   if (delta < 0)
   {
      if (l0 - 1 < -delta)
      {
         fDragOverflow += delta + l0 - 1;
         delta = -l0 + 1;
      }
   }
   else
   {
      if (l1 - 1 < delta)
      {
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

//------------------------------------------------------------------------------

//______________________________________________________________________________
void TGPack::SetVertical(Bool_t x)
{
   // Sets the vertical flag and reformats the back to new stacking
   // direction.

   if (x == fVertical)
      return;

   TList list;
   while ( ! fList->IsEmpty())
   {
      TGFrameElement *el = (TGFrameElement*) fList->At(1);
      TGFrame        *f  = el->fFrame;
      if ( ! (el->fState & kIsVisible) )
         f->SetBit(kTempFrame);
      RemoveFrameInternal(f);
      list.Add(f);
   }
   fVertical = x;
   while ( ! list.IsEmpty())
   {
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
