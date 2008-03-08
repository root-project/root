// @(#)root/gui:$Id$
// Author: Bertrand Bellenot 23/01/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGFrame.h"
#include "TGLayout.h"
#include "TGSplitter.h"
#include "TGSplitFrame.h"
#include "TString.h"
#include "Riostream.h"

ClassImp(TGSplitFrame)

//______________________________________________________________________________
TGSplitFrame::TGSplitFrame(const TGWindow *p, UInt_t w, UInt_t h,
        UInt_t options) : TGCompositeFrame(p, w, h, options), 
        fFrame(0), fSplitter(0), fFirst(0), fSecond(0)
{
   // Default constructor.

}

//______________________________________________________________________________
TGSplitFrame::~TGSplitFrame()
{
   // Destructor. Make cleanup.

   Cleanup();
}

//______________________________________________________________________________
void TGSplitFrame::AddFrame(TGFrame *f, TGLayoutHints *l)
{
   TGCompositeFrame::AddFrame(f, l);
   fFrame = f;
}

//______________________________________________________________________________
void TGSplitFrame::Cleanup()
{
   // recursively cleanup child frames.

   if (fFirst) {
      fFirst->Cleanup();
      delete fFirst;
      fFirst = 0;
   }
   if (fSecond) {
      fSecond->Cleanup();
      delete fSecond;
      fSecond = 0;
   }
   if (fSplitter) {
      delete fSplitter;
      fSplitter = 0;
   }
}

//______________________________________________________________________________
void TGSplitFrame::HSplit(UInt_t h)
{
   // Horizontally split the frame.

   // return if already splitted
   if ((fSplitter != 0) || (fFirst != 0) || (fSecond != 0) || (fFrame != 0))
      return;
   UInt_t height = (h > 0) ? h : fHeight/2;
   // set correct option (vertical frame)
   ChangeOptions((GetOptions() & ~kHorizontalFrame) | kVerticalFrame);
   // create first split frame with fixed height - required for the splitter
   fFirst = new TGSplitFrame(this, fWidth, height, kSunkenFrame | kFixedHeight);
   // create second split frame
   fSecond = new TGSplitFrame(this, fWidth, height, kSunkenFrame);
   // create horizontal splitter
   fSplitter = new TGHSplitter(this, 4, 4);
   // set the splitter's frame to the first one
   fSplitter->SetFrame(fFirst, kTRUE);
   // add all frames
   AddFrame(fFirst, new TGLayoutHints(kLHintsExpandX));
   AddFrame(fSplitter, new TGLayoutHints(kLHintsLeft | kLHintsTop | 
            kLHintsExpandX));
   AddFrame(fSecond, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
}

//______________________________________________________________________________
void TGSplitFrame::VSplit(UInt_t w)
{
   // Vertically split the frame.

   // return if already splitted
   if ((fSplitter != 0) || (fFirst != 0) || (fSecond != 0) || (fFrame != 0))
      return;
   UInt_t width = (w > 0) ? w : fWidth/2;
   // set correct option (horizontal frame)
   ChangeOptions((GetOptions() & ~kVerticalFrame) | kHorizontalFrame);
   // create first split frame with fixed width - required for the splitter
   fFirst = new TGSplitFrame(this, width, fHeight, kSunkenFrame | kFixedWidth);
   // create second split frame
   fSecond = new TGSplitFrame(this, width, fHeight, kSunkenFrame);
   // create vertical splitter
   fSplitter = new TGVSplitter(this, 4, 4);
   // set the splitter's frame to the first one
   fSplitter->SetFrame(fFirst, kTRUE);
   // add all frames
   AddFrame(fFirst, new TGLayoutHints(kLHintsExpandY));
   AddFrame(fSplitter, new TGLayoutHints(kLHintsLeft | kLHintsTop | 
            kLHintsExpandY));
   AddFrame(fSecond, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
}

//______________________________________________________________________________
void TGSplitFrame::SwitchFrames(TGFrame *frame, TGCompositeFrame *dest,
                                TGFrame *prev)
{
   // Switch (exchange) two frames.
   // frame is the source, dest is the destination (the new parent)
   // prev is the frame that has to be exchanged with the source 
   // (the one actually in the destination)

   // get parent of the source (its container)
   TGCompositeFrame *parent = (TGCompositeFrame *)frame->GetParent();

   // unmap the window (to avoid flickering)
   prev->UnmapWindow();
   // remove it from the destination frame
   dest->RemoveFrame(prev);
   // temporary reparent it to root (desktop window)
   prev->ReparentWindow(gClient->GetDefaultRoot());

   // now unmap the source window (still to avoid flickering)
   frame->UnmapWindow();
   // remove it from its parent (its container)
   parent->RemoveFrame(frame);
   // reparent it to the target location
   frame->ReparentWindow(dest);
   // add it to its new parent (for layout managment)
   dest->AddFrame(frame, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   // Layout...
   frame->Resize(dest->GetDefaultSize());
   dest->MapSubwindows();
   dest->Layout();

   // now put back the previous one in the previous source parent
   // reparent to the previous source container
   prev->ReparentWindow(parent);
   // add it to the frame (for layout managment)
   parent->AddFrame(prev, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   // Layout...
   prev->Resize(parent->GetDefaultSize());
   parent->MapSubwindows();
   parent->Layout();
}

//______________________________________________________________________________
void TGSplitFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a splittable frame as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << endl << "   // splittable frame" << endl;
   out << "   TGSplitFrame *";
   out << GetName() << " = new TGSplitFrame(" << fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out << ");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }

   // setting layout manager if it differs from the main frame type
   TGLayoutManager * lm = GetLayoutManager();
   if ((GetOptions() & kHorizontalFrame) &&
       (lm->InheritsFrom(TGHorizontalLayout::Class()))) {
      ;
   } else if ((GetOptions() & kVerticalFrame) &&
              (lm->InheritsFrom(TGVerticalLayout::Class()))) {
      ;
   } else {
      out << "   " << GetName() <<"->SetLayoutManager(";
      lm->SavePrimitive(out, option);
      out << ");"<< endl;
   }

   SavePrimitiveSubframes(out, option);
}
