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

#include <algorithm>
#include <vector>

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
   fDragOverflow (0)
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
   fDragOverflow (0)
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
   if (fUseSplitters)
      len -= fSplitterLen * (fList->GetSize() - 1) / 2;
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
Int_t TGPack::NumberOfRealFrames() const
{
   // Returns number of frames in pack excluding the splitters.

   if (fUseSplitters)
      return (fList->GetSize() + 1) / 2;
   else
      return  fList->GetSize();
}

//______________________________________________________________________________
Int_t TGPack::LengthOfRealFrames() const
{
   // Returns length of frames in pack excluding the splitters.

   Int_t l = 0;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
   {
      l += GetFrameLength(el->fFrame);

      if (fUseSplitters)
         next();
   }

   return l;
}

//______________________________________________________________________________
void TGPack::ResizeExistingFrames(Int_t amount)
{
   // Resize (shrink or expand) existing frames by amount in total.

   if (amount > 0)
      ExpandExistingFrames(amount);
   else if (amount < 0)
      ShrinkExistingFrames(-amount);

   RefitFramesToPack();
}

//______________________________________________________________________________
void TGPack::ExpandExistingFrames(Int_t amount)
{
   // Expand existing frames by amount in total.

   if (fList->IsEmpty())
      return;

   Int_t length    = LengthOfRealFrames();
   Int_t remainder = amount;

   std::vector<TGFrame*> frame_vec;
   {
      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next()))
      {
         Int_t l = GetFrameLength(el->fFrame);
         Int_t d = (l * amount) / length;
         SetFrameLength(el->fFrame, l + d);
         remainder -= d;

         frame_vec.push_back(el->fFrame);

         if (fUseSplitters)
            next();
      }
   }

   std::random_shuffle(frame_vec.begin(), frame_vec.end());

   while (remainder > 0)
   {
      std::vector<TGFrame*>::iterator fi = frame_vec.begin();
      while (fi != frame_vec.end() && remainder > 0)
      {
         Int_t l = GetFrameLength(*fi);
         if (l > 0)
         {
            SetFrameLength(*fi, l + 1);
            --remainder;
         }
         ++fi;
      }
   }
}

//______________________________________________________________________________
void TGPack::ShrinkExistingFrames(Int_t amount)
{
   // Shrink existing frames by amount in total.

   Int_t length    = LengthOfRealFrames();
   Int_t remainder = amount;

   std::vector<TGFrame*> frame_vec;
   {
      TIter next(fList);
      TGFrameElement *el;
      while ((el = (TGFrameElement *) next()))
      {
         Int_t l = GetFrameLength(el->fFrame);
         Int_t d = (l * amount) / length;
         SetFrameLength(el->fFrame, l - d);
         remainder -= d;

         frame_vec.push_back(el->fFrame);

         if (fUseSplitters)
            next();
      }
   }

   std::random_shuffle(frame_vec.begin(), frame_vec.end());

   Bool_t all_one = kFALSE;
   while (remainder > 0 && ! all_one)
   {
      all_one = kTRUE;

      std::vector<TGFrame*>::iterator fi = frame_vec.begin();
      while (fi != frame_vec.end() && remainder > 0)
      {
         Int_t l = GetFrameLength(*fi);
         if (l > 1)
         {
            all_one = kFALSE;
            SetFrameLength(*fi, l - 1);
            --remainder;
         }
         ++fi;
      }
   }
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
void TGPack::FindFrames(TGFrame* splitter, TGFrame*& f0, TGFrame*& f1)
{
   // Find frames around splitter and return them f0 (previous) and f1 (next).

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
   {
      if (el->fFrame == splitter)
         break;
      f0 = el->fFrame;
   }
   el = (TGFrameElement *) next();
   f1 = el->fFrame;
}

//------------------------------------------------------------------------------

//______________________________________________________________________________
void TGPack::AddFrameInternal(TGFrame* f, TGLayoutHints* l)
{
   // Add frame f at the end.
   // LayoutHints are ignored in TGPack.

   Int_t n     = NumberOfRealFrames();
   Int_t nflen = (GetLength() - (fUseSplitters ? fSplitterLen*n : 0)) / (n + 1);

   // printf("New frame, n=%d, new_frame_len=%d\n", n, nflen);

   if (n > 0)
   {
      ShrinkExistingFrames(nflen);

      if (fUseSplitters) {
         nflen -= fSplitterLen;
         TGSplitter* s = 0;
         if (fVertical)
            s = new TGHSplitter(this, GetWidth(), fSplitterLen, kTRUE);
         else
            s = new TGVSplitter(this, fSplitterLen, GetHeight(), kTRUE);
         s->Connect("Moved(Int_t)",  "TGPack", this, "HandleSplitterResize(Int_t)");
         s->Connect("DragStarted()", "TGPack", this, "HandleSplitterStart()");
         TGCompositeFrame::AddFrame(s);
         s->MapWindow();
      }
   }
   SetFrameLength(f, nflen);
   TGCompositeFrame::AddFrame(f, l);
   f->MapWindow();
}

//______________________________________________________________________________
void TGPack::AddFrame(TGFrame* f, TGLayoutHints* l)
{
   // Add frame f at the end.
   // LayoutHints are ignored in TGPack.

   AddFrameInternal(f, l);

   Layout();
}

//______________________________________________________________________________
Int_t TGPack::RemoveFrameInternal(TGFrame* f)
{
   // Remove frame f.

   TGFrameElement *el = FindFrameElement(f);

   if (!el) return 0;

   Int_t space_freed = 0;

   if (fUseSplitters && NumberOfRealFrames() > 1)
   {
      TGFrameElement *splitter_el = 0;
      if (el == fList->First())
         splitter_el = (TGFrameElement*) fList->After(el);
      else
         splitter_el = (TGFrameElement*) fList->Before(el);
      TGFrame* splitter = splitter_el->fFrame;
      space_freed += fSplitterLen;
      splitter->UnmapWindow();
      TGCompositeFrame::RemoveFrame(splitter);
      // This is needed so that splitter window gets destroyed on server.
      splitter->ReparentWindow(fClient->GetDefaultRoot());
      delete splitter;
   }

   space_freed += GetFrameLength(f);
   f->UnmapWindow();
   TGCompositeFrame::RemoveFrame(f);

   // printf("Removed frame, n=%d, space_freed=%d\n", NumberOfRealFrames(), space_freed);

   return space_freed;
}

//______________________________________________________________________________
void TGPack::DeleteFrame(TGFrame* f)
{
   // Remove frame f and refit existing frames to pack size.
   // Frame is deleted.

   Int_t space_freed = RemoveFrameInternal(f);
   if (space_freed)
   {
      delete f;
      ResizeExistingFrames(space_freed);
      Layout();
   }
}

//______________________________________________________________________________
void TGPack::RemoveFrame(TGFrame* f)
{
   // Remove frame f and refit existing frames to pack size.
   // Frame is not deleted.

   Int_t space_freed = RemoveFrameInternal(f);
   if (space_freed)
   {
      ResizeExistingFrames(space_freed);
      Layout();
   }
}

//______________________________________________________________________________
void TGPack::ShowFrame(TGFrame* /*f*/)
{
   // Virtual from TGCompositeFrame.
   // This operation not supported by pack.

   Error("ShowFrame", "not yet supported.");
}

//______________________________________________________________________________
void TGPack::HideFrame(TGFrame* /*f*/)
{
   // Virtual from TGCompositeFrame.
   // This operation not supported by pack.

   Error("HideFrame", "not yet supported.");
}

//------------------------------------------------------------------------------

//______________________________________________________________________________
void TGPack::Resize(UInt_t w, UInt_t h)
{
   // Resize the pack.
   // Contents is resized proportionally.

   if (w == fWidth && h == fHeight) return;

   Int_t delta = fVertical ? h - GetHeight() : w - GetWidth();

   fWidth  = w;
   fHeight = h;
   TGWindow::Resize(fWidth, fHeight);

   ResizeExistingFrames(delta);

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
      SetFramePosition(el->fFrame, pos);
      pos += GetFrameLength(el->fFrame);
      el->fFrame->Layout();
   }
}

//______________________________________________________________________________
void TGPack::EqualizeFrames()
{
   // Refit existing frames so that their lengths are equal.

   if (fList->IsEmpty())
      return;

   Int_t length = GetAvailableLength();
   Int_t nf     = NumberOfRealFrames();
   Int_t lpf    = (Int_t) (((Double_t) length) / nf);
   Int_t extra  = length - nf*lpf;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
   {
      SetFrameLength(el->fFrame, lpf + extra);
      extra = 0;

      if (fUseSplitters)
         next();
   }
}

//------------------------------------------------------------------------------

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

   Int_t min_dec = - (GetAvailableLength() + NumberOfRealFrames());
   if (delta <  min_dec)
      delta = min_dec;

   TGSplitter *s = dynamic_cast<TGSplitter*>((TGFrame*) gTQSender);

   TGFrame *f0=0, *f1=0;
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

   if (delta < 0)
   {
      Int_t l = GetFrameLength(f0);
      if (l - 1 < -delta)
      {
         fDragOverflow += delta + l - 1;
         delta = -l + 1;
      }
      SetFrameLength(f0, l + delta);
      SetFrameLength(f1, GetFrameLength(f1) - delta);
   }
   else
   {
      Int_t l = GetFrameLength(f1);
      if (l - 1 < delta)
      {
         fDragOverflow += delta - l + 1;
         delta = l - 1;
      }
      SetFrameLength(f0, GetFrameLength(f0) + delta);
      SetFrameLength(f1, l - delta);
   }
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
      TGFrame* f = ((TGFrameElement*) fList->First())->fFrame;
      RemoveFrameInternal(f);
      list.Add(f);
   }
   fVertical = x;
   while ( ! list.IsEmpty())
   {
      TGFrame* f = (TGFrame*) list.First();
      AddFrameInternal(f);
      list.RemoveFirst();
   }
   Layout();
}
