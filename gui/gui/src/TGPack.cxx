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

//______________________________________________________________________________
// Description of TGPack
//

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
}

//______________________________________________________________________________
TGPack::~TGPack()
{
   // Destructor.
}

//------------------------------------------------------------------------------

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

   Int_t len = LengthOfRealFrames();

   Int_t remainder = amount;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
   {
      Int_t l = GetFrameLength(el->fFrame);
      Int_t d = (l * amount) / len;
      SetFrameLength(el->fFrame, l + d);
      remainder -= d;
      
      if (fUseSplitters)
         next();
   }

   while (remainder > 0)
   {
      next.Reset();
      while ((el = (TGFrameElement *) next()) && remainder > 0)
      {
         Int_t l = GetFrameLength(el->fFrame);
         if (l > 0)
         {
            SetFrameLength(el->fFrame, l + 1);
            --remainder;
         }

         if (fUseSplitters)
            next();
      }
   }
}

//______________________________________________________________________________
void TGPack::ShrinkExistingFrames(Int_t amount)
{
   // Shrink existing frames by amount in total.

   Int_t len = LengthOfRealFrames();

   Int_t remainder = amount;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
   {
      Int_t l = GetFrameLength(el->fFrame);
      Int_t d = (l * amount) / len;
      SetFrameLength(el->fFrame, l - d);
      remainder -= d;
      
      if (fUseSplitters)
         next();
   }

   Bool_t all_one = kFALSE;
   while (remainder > 0 && ! all_one)
   {
      next.Reset();
      all_one = kTRUE;
      while ((el = (TGFrameElement *) next()) && remainder > 0)
      {
         Int_t l = GetFrameLength(el->fFrame);
         if (l > 1)
         {
            all_one = kFALSE;
            SetFrameLength(el->fFrame, l - 1);
            --remainder;
         }

         if (fUseSplitters)
            next();
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
void TGPack::AddFrame(TGFrame* f, TGLayoutHints* l)
{
   // Add frame f at the end.
   // LayoutHints are ignored in TGPack.

   Int_t n     = NumberOfRealFrames();
   Int_t nflen = (GetLength() - (fUseSplitters ? fSplitterLen*n : 0)) / (n + 1);

   printf("New frame, n=%d, new_frame_len=%d\n", n, nflen);

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
      }
   }
   SetFrameLength(f, nflen);
   TGCompositeFrame::AddFrame(f, l);

   Layout();
   MapSubwindows();
}

//______________________________________________________________________________
void TGPack::RemoveFrame(TGFrame* f)
{
   // Remove frame f and refit existing frames to pack size.

//   Error("RemoveFrame", "not yet supported.");

   TGFrameElement *el = FindFrameElement(f);

   if (!el) return;

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
      delete splitter;
   }

   space_freed += GetFrameLength(f);
   f->UnmapWindow();
   TGCompositeFrame::RemoveFrame(f);
   delete f;

   printf("Removed frame, n=%d, space_freed=%d\n", NumberOfRealFrames(), space_freed);

   ResizeExistingFrames(space_freed);
   Layout();
}

//______________________________________________________________________________
void TGPack::ShowFrame(TGFrame* /*f*/)
{
   // Blabla blu.

   Error("ShowFrame", "not yet supported.");
}

//______________________________________________________________________________
void TGPack::HideFrame(TGFrame* /*f*/)
{
   // Blabla blu.

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
void TGPack::HandleSplitterStart()
{
   // Called when splitter drag starts.

   fDragOverflow = 0;
}

//______________________________________________________________________________
void TGPack::HandleSplitterResize(Int_t delta)
{
   // Handle resize events from splitters.

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
