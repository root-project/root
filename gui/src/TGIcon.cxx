// @(#)root/gui:$Name:  $:$Id: TGIcon.cxx,v 1.11 2006/04/24 13:51:06 antcheva Exp $
// Author: Fons Rademakers   05/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGIcon                                                               //
//                                                                      //
// This class handles GUI icons.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGIcon.h"
#include "TGPicture.h"
#include "TSystem.h"
#include "TImage.h"
#include "Riostream.h"
#include "TMath.h"
#include "TGFileDialog.h"
#include "TGMsgBox.h"
#include "TVirtualDragManager.h"


ClassImp(TGIcon)

//______________________________________________________________________________
TGIcon::TGIcon(const TGWindow *p, const char *image) : TGFrame(p, 1, 1)
{
   // Create icon.

   fPic = 0;
   char *path;

   if (!image) {
      const char *rootsys = gSystem->ExpandPathName("$ROOTSYS");
      path = gSystem->ConcatFileName(rootsys, "icons/bld_rgb.xpm");
   }
   fPath = gSystem->DirName(path);

   fImage = TImage::Open(path);
   fPic = fClient->GetPicturePool()->GetPicture(Form("%s_%dx%d", gSystem->BaseName(path), 
                                                fImage->GetWidth(), fImage->GetHeight()),
                                                fImage->GetPixmap(), fImage->GetMask());
   TGFrame::Resize(fImage->GetWidth(), fImage->GetHeight());
   SetWindowName();
   delete path;
}

//______________________________________________________________________________
TGIcon::~TGIcon()
{
   // Delete icon and free picture.

   if (fPic) fClient->FreePicture(fPic);
}

//______________________________________________________________________________
void TGIcon::SetPicture(const TGPicture *pic)
{
   // Set icon picture.

   fPic = pic;
   gVirtualX->ClearWindow(fId);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGIcon::SetImage(const char *img)
{
   // Set icon image.

   //delete fImage;
   TImage *i = TImage::Open(img);
   fPath = gSystem->DirName(img);

   SetImage(i);
}

//______________________________________________________________________________
void TGIcon::SetImage(TImage *img)
{
   // Change icon image.

   if (!img) {
      return;
   }

   //delete fImage;  !! mem.leak!!
   fImage = img;
   Resize(fImage->GetWidth(), fImage->GetHeight());
}

//______________________________________________________________________________
TGDimension TGIcon::GetDefaultSize() const
{
   // Return size of icon.

   return TGDimension((fPic) ? fPic->GetWidth()  : fWidth,
                      (fPic) ? fPic->GetHeight() : fHeight);
}

//______________________________________________________________________________
void TGIcon::DoRedraw()
{
   // Redraw picture.

   Bool_t border = (GetOptions() & kRaisedFrame) || 
                   (GetOptions() & kSunkenFrame) ||
                   (GetOptions() & kDoubleBorder);

   if (fPic) fPic->Draw(fId, GetBckgndGC()(), border, border);
   DrawBorder();
}

//______________________________________________________________________________
void TGIcon::Resize(UInt_t w, UInt_t h)
{
   // Resize.

   //if (fImage && (TMath::Abs(Int_t(fImage->GetWidth() - w)) < 5) && 
   //    (TMath::Abs(Int_t(fImage->GetHeight() - h)) < 5)) {
   //   return;
   //}

   gVirtualX->ClearWindow(fId);
   TGFrame::Resize(w, h);

   if (!fImage) {
      fImage = TImage::Create();
      if (fPic) fImage->SetImage(fPic->GetPicture(), fPic->GetMask());
   }
   if (fPic) {
      fClient->FreePicture(fPic);
   }
   Bool_t border = (GetOptions() & kRaisedFrame) || 
                   (GetOptions() & kSunkenFrame) ||
                   (GetOptions() & kDoubleBorder);

   fImage->Scale(w - 2*border, h - 2*border);
   fPic = fClient->GetPicturePool()->GetPicture(Form("%s_%dx%d", fImage->GetName(), 
                                                fImage->GetWidth(), fImage->GetHeight()),
                                                fImage->GetPixmap(), fImage->GetMask());
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGIcon::MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move icon to (x,y) and resize it to (w,h). 

   Move(x, y);
   Resize(w, h);
}

//______________________________________________________________________________
void TGIcon::Reset()
{
   // Reset icon to original image. It can be used only via context menu.

   if (!fImage || !fClient->IsEditable()) return;

   TString name = fImage->GetName();
   name.Chop();
   char *path = gSystem->ConcatFileName(fPath.Data(), name.Data());
   SetImage(path);

   delete path;
}

//______________________________________________________________________________
void TGIcon::ChangeImage()
{
   // Invoke file dialog to assign a new image.

   static const char *gImageTypes[] = {"XPM",     "*.xpm", 
                                       "GIF",     "*.gif",
                                       "PNG",     "*.png", 
                                       "JPEG",    "*.jpg",
                                       "TARGA",   "*.tga", 
                                       "ICO",     "*.ico", 
                                       "XCF",     "*.xcf", 
                                       "CURSORS", "*.cur",
                                       "PPM",     "*.ppm", 
                                       "PNM",     "*.pnm", 
                                       "XBM",     "*.xbm", 
                                       "TIFF",    "*.tiff", 
                                       "BMP",     "*.bmp",
                                       "Enacapsulated PostScript", "*.eps", 
                                       "PostScript", "*.ps", 
                                       "PDF",        "*.pdf", 
                                       "ASImage XML","*.xml",
                                       "All files",  "*",
                                        0,             0 };

   TGFileInfo fi;
   static TString dir(".");
   static Bool_t overwr = kFALSE;
   const char *fname;

   fi.fFileTypes = gImageTypes;
   fi.fIniDir    = StrDup(dir);
   fi.fOverwrite = overwr;

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   gDragManager->SetEditable(kFALSE);

   new TGFileDialog(fClient->GetDefaultRoot(), this, kFDOpen, &fi);

   if (!fi.fFilename) {
      root->SetEditable(kTRUE);
      gDragManager->SetEditable(kTRUE);
      return;
   }

   dir    = fi.fIniDir;
   overwr = fi.fOverwrite;
   fname  = fi.fFilename;

   fImage = TImage::Open(fname);

   if (!fImage) {
      Int_t retval;
      new TGMsgBox(fClient->GetDefaultRoot(), this, "Error...",
                   Form("Cannot read image file (%s)", fname),
                   kMBIconExclamation, kMBRetry | kMBCancel, &retval);

      if (retval == kMBRetry) {
         ChangeImage();
      }
   } else {
      SetImage(fImage);
   }
   root->SetEditable(kTRUE);
   gDragManager->SetEditable(kTRUE);
}

//______________________________________________________________________________
void TGIcon::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save an icon widget as a C++ statement(s) on output stream out.

   char quote = '"';

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   if (!fPic) {
      Error("SavePrimitive()", "icon pixmap not found ");
      return;
   }

   const char *picname = fPic->GetName();

   out <<"   TGIcon *";
   out << GetName() << " = new TGIcon(" << fParent->GetName()
       << ",gClient->GetPicture(" << quote
       << gSystem->ExpandPathName(gSystem->UnixPathName(picname))                       // if no path
       << quote << ")" << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
}
