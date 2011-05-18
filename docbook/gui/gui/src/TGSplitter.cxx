// @(#)root/gui:$Id$
// Author: Fons Rademakers   6/09/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGSplitter, TGVSplitter and TGHSplitter                              //
//                                                                      //
// A splitter allows the frames left and right or above and below of    //
// it to be resized. The frame to be resized must have the kFixedWidth  //
// or kFixedHeight property set.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGSplitter.h"
#include "TGPicture.h"
#include "Riostream.h"


ClassImp(TGSplitter)
ClassImp(TGVSplitter)
ClassImp(TGHSplitter)
ClassImp(TGVFileSplitter)


//______________________________________________________________________________
TGSplitter::TGSplitter(const TGWindow *p, UInt_t w, UInt_t h,
                       UInt_t options, ULong_t back) :
   TGFrame(p, w, h, options, back),
   fDragging        (kFALSE),
   fExternalHandler (kFALSE),
   fSplitterPic     (0)
{
   // Create a splitter.

   fSplitCursor = kNone;
   fEditDisabled = kTRUE;
}

//______________________________________________________________________________
void TGSplitter::DragStarted()
{
   // Emit DragStarted signal.

   Emit("DragStarted()");
}

//______________________________________________________________________________
void TGSplitter::Moved(Int_t delta)
{
   // Emit Moved signal.

   Emit("Moved(Int_t)", delta);
}

//______________________________________________________________________________
TGVSplitter::TGVSplitter(const TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options, ULong_t back) : TGSplitter(p, w, h, options, back)
{
   // Create a vertical splitter.

   fSplitCursor = kNone;
   fSplitterPic = fClient->GetPicture("splitterv.xpm");
   fFrameHeight = h;
   fFrameWidth = w;
   fLeft = kTRUE;
   fMax = fMin = 0;
   fStartX = 0;

   if (!fSplitterPic)
      Error("TGVSplitter", "splitterv.xpm not found");

   if (p && !p->InheritsFrom(TGCompositeFrame::Class())) {
      Error("TGVSplitter", "parent must inherit from a TGCompositeFrame");
      return;
   }
   if (p && !(((TGCompositeFrame*)p)->GetOptions() & kHorizontalFrame)) {
      Error("TGVSplitter", "parent must have a horizontal layout manager");
      return;
   }

   fSplitCursor = gVirtualX->CreateCursor(kArrowHor);
   fFrame = 0;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);

   AddInput(kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
TGVSplitter::TGVSplitter(const TGWindow *p, UInt_t w, UInt_t h, Bool_t external) :
   TGSplitter(p, w, h, kChildFrame, GetDefaultFrameBackground())
{
   // Create a vertical splitter.

   fExternalHandler = external;

   fSplitCursor = kNone;
   fSplitterPic = fClient->GetPicture("splitterv.xpm");

   if (!fSplitterPic)
      Error("TGVSplitter", "splitterv.xpm not found");

   fSplitCursor = gVirtualX->CreateCursor(kArrowHor);
   fFrame = 0;
   fFrameHeight = h;
   fFrameWidth = w;
   fLeft = kTRUE;
   fMax = fMin = 0;
   fStartX = 0;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);

   AddInput(kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
TGVSplitter::~TGVSplitter()
{
   // Delete vertical splitter widget.

   if (fSplitterPic) fClient->FreePicture(fSplitterPic);
}

//______________________________________________________________________________
void TGVSplitter::SetFrame(TGFrame *frame, Bool_t left)
{
   // Set frame to be resized. If frame is on the left of the splitter
   // set left to true.

   fFrame = frame;
   fLeft  = left;

   if (!fExternalHandler && !(fFrame->GetOptions() & kFixedWidth))
      Error("SetFrame", "resize frame must have kFixedWidth option set");
}

//______________________________________________________________________________
Bool_t TGVSplitter::HandleButton(Event_t *event)
{
   // Handle mouse button event in vertical splitter.

   if (fSplitCursor == kNone) return kTRUE;

   if (!fExternalHandler && !fFrame) {
      Error("HandleButton", "frame to be resized not set");
      return kTRUE;
   }

   if (event->fType == kButtonPress) {
      fStartX   = event->fXRoot;
      fDragging = kTRUE;

      if (fExternalHandler) {
         fMin = 0;
         fMax = 99999;
         DragStarted();
      } else {
         Int_t  x, y;
         gVirtualX->GetWindowSize(fFrame->GetId(), x, y, fFrameWidth, fFrameHeight);

         // get fMin and fMax in root coordinates
         Int_t    xroot, yroot;
         UInt_t   w, h;
         Window_t wdum;
         gVirtualX->GetWindowSize(fParent->GetId(), x, y, w, h);
         gVirtualX->TranslateCoordinates(fParent->GetParent()->GetId(),
                                         fClient->GetDefaultRoot()->GetId(),
                                         x, y, xroot, yroot, wdum);
         fMin = xroot;
         fMax = xroot + w - 2;
      }

      // last argument kFALSE forces all specified events to this window
      gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask |
                             kPointerMotionMask, kNone, fSplitCursor,
                             kTRUE, kFALSE);
   } else {
      fDragging = kFALSE;
      gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab pointer
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGVSplitter::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in vertical splitter.

   if (fDragging) {
      Int_t xr = event->fXRoot;
      if (xr > fMax) xr = fMax;
      if (xr < fMin) xr = fMin;
      Int_t delta = xr - fStartX;
      if (fExternalHandler) {
         if (delta != 0) {
            Moved(delta);
            fStartX = xr;
         }
      } else {
         Int_t w = (Int_t) fFrameWidth;
         if (fLeft)
            w += delta;
         else
            w -= delta;
         if (w < 0) w = 0;
         fStartX = xr;

         if (delta != 0) {
            fFrameWidth = w;
            fFrame->Resize(fFrameWidth, fFrameHeight);

            TGCompositeFrame *p = (TGCompositeFrame *) GetParent();
            p->Layout();
         }
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGVSplitter::HandleCrossing(Event_t *event)
{
   // Handle mouse motion event in vertical splitter.

   if (event->fType == kEnterNotify)
      gVirtualX->SetCursor(fId, fSplitCursor);
   else
      gVirtualX->SetCursor(fId, kNone);

   return kTRUE;
}

//______________________________________________________________________________
void TGVSplitter::DrawBorder()
{
   // Draw vertical splitter.

   if (fSplitterPic) {
      Int_t posx = (fWidth/2)-(fSplitterPic->GetWidth()/2);
      Int_t posy = (fHeight/2)-(fSplitterPic->GetHeight()/2);
      fSplitterPic->Draw(fId, GetBckgndGC()(), posx, posy);
   }
}


//______________________________________________________________________________
TGHSplitter::TGHSplitter(const TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options, ULong_t back) : TGSplitter(p, w, h, options, back)
{
   // Create a horizontal splitter.

   fSplitCursor = kNone;

   fSplitterPic = fClient->GetPicture("splitterh.xpm");

   if (!fSplitterPic)
      Error("TGHSplitter", "splitterh.xpm not found");

   if (p && !p->InheritsFrom(TGCompositeFrame::Class())) {
      Error("TGHSplitter", "parent must inherit from a TGCompositeFrame");
      return;
   }
   if (p && !(((TGCompositeFrame*)p)->GetOptions() & kVerticalFrame)) {
      Error("TGHSplitter", "parent must have a vertical layout manager");
      return;
   }

   fSplitCursor = gVirtualX->CreateCursor(kArrowVer);
   fFrame = 0;
   fFrameHeight = h;
   fFrameWidth = w;
   fAbove = kTRUE;
   fMax = fMin = 0;
   fStartY = 0;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);

   AddInput(kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
TGHSplitter::TGHSplitter(const TGWindow *p, UInt_t w, UInt_t h, Bool_t external) :
   TGSplitter(p, w, h, kChildFrame, GetDefaultFrameBackground())
{
   // Create a horizontal splitter.

   fExternalHandler = external;

   fSplitCursor = kNone;

   fSplitterPic = fClient->GetPicture("splitterh.xpm");

   if (!fSplitterPic)
      Error("TGHSplitter", "splitterh.xpm not found");

   fSplitCursor = gVirtualX->CreateCursor(kArrowVer);
   fFrame = 0;
   fFrameHeight = h;
   fFrameWidth = w;
   fAbove = kTRUE;
   fMax = fMin = 0;
   fStartY = 0;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);

   AddInput(kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
TGHSplitter::~TGHSplitter()
{
   // Delete horizontal splitter widget.

   if (fSplitterPic) fClient->FreePicture(fSplitterPic);
}

//______________________________________________________________________________
void TGHSplitter::SetFrame(TGFrame *frame, Bool_t above)
{
   // Set frame to be resized. If frame is above the splitter
   // set above to true.

   fFrame = frame;
   fAbove = above;

   if (!fExternalHandler && !(fFrame->GetOptions() & kFixedHeight))
      Error("SetFrame", "resize frame must have kFixedHeight option set");
}

//______________________________________________________________________________
Bool_t TGHSplitter::HandleButton(Event_t *event)
{
   // Handle mouse button event in horizontal splitter.

   if (fSplitCursor == kNone) return kTRUE;

   if (!fExternalHandler && !fFrame) {
      Error("HandleButton", "frame to be resized not set");
      return kTRUE;
   }

   if (event->fType == kButtonPress) {
      fStartY   = event->fYRoot;
      fDragging = kTRUE;

      if (fExternalHandler) {
         fMin = 0;
         fMax = 99999;
         DragStarted();
      } else {
         Int_t  x, y;
         gVirtualX->GetWindowSize(fFrame->GetId(), x, y, fFrameWidth, fFrameHeight);

         // get fMin and fMax in root coordinates
         Int_t    xroot, yroot;
         UInt_t   w, h;
         Window_t wdum;
         gVirtualX->GetWindowSize(fParent->GetId(), x, y, w, h);
         gVirtualX->TranslateCoordinates(fParent->GetParent()->GetId(),
                                         fClient->GetDefaultRoot()->GetId(),
                                         x, y, xroot, yroot, wdum);
         fMin = yroot;
         fMax = yroot + h - 2;
      }

      // last argument kFALSE forces all specified events to this window
      gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask |
                             kPointerMotionMask, kNone, fSplitCursor,
                             kTRUE, kFALSE);
   } else {
      fDragging = kFALSE;
      gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab pointer
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGHSplitter::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in horizontal splitter.

   if (fDragging) {
      Int_t yr = event->fYRoot;
      if (yr > fMax) yr = fMax;
      if (yr < fMin) yr = fMin;
      Int_t delta = yr - fStartY;
      if (fExternalHandler) {
         if (delta != 0) {
            Moved(delta);
            fStartY = yr;
         }
      } else {
         Int_t h = (Int_t) fFrameHeight;
         if (fAbove)
            h += delta;
         else
            h -= delta;
         if (h < 0) h = 0;
         fStartY = yr;

         if (delta != 0) {
            fFrameHeight = h;
            fFrame->Resize(fFrameWidth, fFrameHeight);

            TGCompositeFrame *p = (TGCompositeFrame *) GetParent();
            p->Layout();
         }
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGHSplitter::HandleCrossing(Event_t *event)
{
   // Handle mouse motion event in horizontal splitter.

   if (event->fType == kEnterNotify)
      gVirtualX->SetCursor(fId, fSplitCursor);
   else
      gVirtualX->SetCursor(fId, kNone);

   return kTRUE;
}

//______________________________________________________________________________
void TGHSplitter::DrawBorder()
{
   // Draw horizontal splitter.

   if (fSplitterPic) {
      Int_t posx = (fWidth/2)-(fSplitterPic->GetWidth()/2);
      Int_t posy = (fHeight/2)-(fSplitterPic->GetHeight()/2);
      fSplitterPic->Draw(fId, GetBckgndGC()(), posx, posy);
   }
}

//______________________________________________________________________________
TGVFileSplitter::TGVFileSplitter(const TGWindow *p, UInt_t w, UInt_t h,
                                 UInt_t options, Pixel_t back):
  TGVSplitter(p, w, h, options, back)
{
//    fSplitterPic = fClient->GetPicture("filesplitterv.xpm");

//    if (!fSplitterPic)
//       Error("TGVFileSplitter", "filesplitterv.xpm not found");
}

//______________________________________________________________________________
TGVFileSplitter::~TGVFileSplitter()
{
//    if (fSplitterPic) fClient->FreePicture(fSplitterPic);
}

//______________________________________________________________________________
Bool_t TGVFileSplitter::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in vertical splitter.

   fMin = 30;

   if (fDragging) {
      Int_t xr = event->fXRoot;
      if (xr > fMax) xr = fMax;
      if (xr < fMin) xr = fMin;
      Int_t delta = xr - fStartX;
      Int_t w = (Int_t) fFrameWidth;
      if (fLeft)
         w += delta;
      else
         w -= delta;

      if (w < 0) w = 0;
      fStartX = xr;

      if (delta != 0) {
         delta = w - fFrameWidth;
         fFrameWidth = w;

         TGCompositeFrame *p = (TGCompositeFrame *) GetParent();
         p->Resize( p->GetWidth() + delta, p->GetHeight() );

         fFrame->Resize(fFrameWidth, fFrameHeight);

         p->Layout();
         LayoutHeader((TGFrame *)fFrame);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGVFileSplitter::HandleButton(Event_t *event)
{
   // Handle mouse button event in vertical splitter.

   if ( event->fType == kButtonPress) {
      ButtonPressed();
   } else if ( event->fType == kButtonRelease) {
      LayoutHeader(0);
      LayoutListView();
      ButtonReleased();
   } else if ( event->fType == kButtonDoubleClick ) {
      DoubleClicked(this);
   }
   return TGVSplitter::HandleButton(event);
}

//______________________________________________________________________________
void TGVFileSplitter::LayoutHeader(TGFrame *f)
{
   // Emit LayoutFeader() signal.

   Emit("LayoutHeader(TGFrame*)", (Long_t)f);
}

//______________________________________________________________________________
void TGVFileSplitter::LayoutListView()
{
   // Emit LayoutListView() signal.

   Emit("LayoutListView()");
}

//______________________________________________________________________________
void TGVFileSplitter::ButtonPressed()
{
   // Emit ButtonPressed() signal.

   Emit("ButtonPressed()");
}

//______________________________________________________________________________
void TGVFileSplitter::ButtonReleased()
{
   // Emit ButtonReleased() signal.

   Emit("ButtonReleased()");
}

//______________________________________________________________________________
void TGVFileSplitter::DoubleClicked(TGVFileSplitter* splitter)
{
   // Emit DoubleClicked() signal.

   Emit("DoubleClicked(TGVFileSplitter*)", (Long_t) splitter);
}

//______________________________________________________________________________
Bool_t TGVFileSplitter::HandleDoubleClick(Event_t *)
{
   // Handle double click mouse event in splitter.

   DoubleClicked(this);
   return kTRUE;
}

//______________________________________________________________________________
void TGVSplitter::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
    // Save a splitter widget as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << "   TGVSplitter *";
   out << GetName() <<" = new TGVSplitter("<< fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;
   // TGVSplitter->SetFrame( theframe ) can only be saved here
   // if fFrame is the frame on the left (since the frame on the
   // right will only be saved afterwards)... The other case is
   // handled in TGCompositeFrame::SavePrimitiveSubframes()
   if (GetLeft()) {
      out << "   " << GetName() << "->SetFrame(" << GetFrame()->GetName();
      if (GetLeft()) out << ",kTRUE);" << endl;
      else           out << ",kFALSE);"<< endl;
   }
}

//______________________________________________________________________________
void TGHSplitter::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
    // Save a splitter widget as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << "   TGHSplitter *";
   out << GetName() <<" = new TGHSplitter("<< fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;
   // TGHSplitter->SetFrame( theframe ) can only be saved here
   // if fFrame is the frame above (since the frame below will
   // only be saved afterwards)... The other case is handled in
   // TGCompositeFrame::SavePrimitiveSubframes()
   if (GetAbove()) {
      out << "   " << GetName() << "->SetFrame(" << GetFrame()->GetName();
      if (GetAbove()) out << ",kTRUE);" << endl;
      else            out << ",kFALSE);"<< endl;
   }
}

//______________________________________________________________________________
void TGVFileSplitter::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
    // Save a splitter widget as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << "   TGVFileSplitter *";
   out << GetName() <<" = new TGVFileSplitter("<< fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   out << "   " << GetName() << "->SetFrame(" << GetFrame()->GetName();
   if (GetLeft()) out << ",kTRUE);" << endl;
   else           out << ",kFALSE);"<< endl;
}

