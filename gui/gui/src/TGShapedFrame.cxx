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
#include "TString.h"
#include "TGPicture.h"
#include "TImage.h"
#include "TGShapedFrame.h"
#include "Riostream.h"

ClassImp(TGShapedFrame)

//______________________________________________________________________________
TGShapedFrame::TGShapedFrame(const char *pname, const TGWindow *p, UInt_t w,
                             UInt_t h, UInt_t options) :
      TGCompositeFrame(p, w, h, options), fBgnd(0), fImage(0)
{
   // Shaped window default constructor

   TString picName;
   // set a few attributes
   if (options & kTempFrame) {
      SetWindowAttributes_t attr;
      attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
      attr.fOverrideRedirect = kTRUE;
      attr.fSaveUnder        = kTRUE;
      gVirtualX->ChangeWindowAttributes(fId, &attr);
   }
   // open the image file used as shape & background
   if (pname)
      picName = pname;
   else
      picName = "Default.png";
   fImage = TImage::Open(picName.Data());
   if (!fImage || !fImage->IsValid())
      Error("TGShapedFrame", "%s not found", picName.Data());
   fBgnd = fClient->GetPicturePool()->GetPicture(picName.Data(),
           fImage->GetPixmap(), fImage->GetMask());
   // shape the window with the picture mask
   gVirtualX->ShapeCombineMask(fId, 0, 0, fBgnd->GetMask());
   // and finally set the background picture
   SetBackgroundPixmap(fBgnd->GetPicture());

   MapSubwindows();
   Resize();
   Resize(fBgnd->GetWidth(), fBgnd->GetHeight());
}

//______________________________________________________________________________
TGShapedFrame::~TGShapedFrame()
{
   // Destructor.

   delete fImage;
   fClient->FreePicture(fBgnd);
}

//______________________________________________________________________________
void TGShapedFrame::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   // Save a shaped frame as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << std::endl << "   // shaped frame" << std::endl;
   out << "   TGShapedFrame *";
   out << GetName() << " = new TGShapedFrame(" << fImage->GetName()
       << "," << fParent->GetName() << "," << GetWidth() << ","
       << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out << ");" << std::endl;
      } else {
         out << "," << GetOptionString() <<");" << std::endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << std::endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << std::endl;

   // setting layout manager if it differs from the main frame type
   // coverity[returned_null]
   // coverity[dereference]
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
      out << ");"<< std::endl;
   }

   SavePrimitiveSubframes(out, option);
}
