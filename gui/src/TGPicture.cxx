// @(#)root/gui:$Name:  $:$Id: TGPicture.cxx,v 1.13 2004/10/18 12:47:28 brun Exp $
// Author: Fons Rademakers   01/01/98

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
// TGPicture & TGPicturePool                                            //
//                                                                      //
// The TGPicture class implements pictures and icons used in the        //
// different GUI elements and widgets. The TGPicturePool class          //
// implements a TGPicture cache. TGPictures are created, managed and    //
// destroyed by the TGPicturePool.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGPicture.h"
#include "TGResourcePool.h"
#include "THashTable.h"
#include "TSystem.h"
#include "TGWindow.h"
#include "TVirtualX.h"
#include "TImage.h"

TGGC *TGSelectedPicture::fgSelectedGC = 0;

ClassImp(TGPicture)
ClassImp(TGSelectedPicture)
ClassImp(TGPicturePool)


//______________________________________________________________________________
const TGPicture *TGPicturePool::GetPicture(const char *name)
{
   // Get a picture from the picture pool. Picture must be freed using
   // TGPicturePool::FreePicture(). If picture is not found 0 is returned.

   TGPicture *pic;

   if (!fPicList)
      fPicList = new THashTable(50);

   TString pname = name;
   pname.Strip();
   TString ext = strrchr(pname.Data(), '.');
   ext.ToLower();

   if (ext.Length()) { // ".xpm", ".gif" etc
      char *pxname = gSystem->ExpandPathName(gSystem->UnixPathName(pname));
      pname = pxname;
      delete [] pxname;
   }

   pic = (TGPicture *)fPicList->FindObject(pname);
   if (pic && !pic->IsScaled()) {
      if (pic->fPic == kNone)
         return 0;
      pic->AddReference();
      return pic;
   }

   char *picnam = gSystem->Which(fPath, pname, kReadPermission);
   if (!picnam) {
      return 0;
   }

   if (ext != ".xpm") {
      TImage *img = TImage::Open(picnam);
      if (!img) return 0;

      pic = new TGPicture(pname, img->GetPixmap(), img->GetMask());
      delete [] picnam;
      delete img;
      fPicList->Add(pic);
      return pic;
   }

   pic = new TGPicture(pname);
   pic->fAttributes.fColormap  = fClient->GetDefaultColormap();
   pic->fAttributes.fCloseness = 40000; // Allow for "similar" colors
   pic->fAttributes.fMask      = kPASize | kPAColormap | kPACloseness;

   if (gVirtualX->CreatePictureFromFile(fClient->GetDefaultRoot()->GetId(), picnam,
                                        pic->fPic, pic->fMask,
                                        pic->fAttributes)) {
      fPicList->Add(pic);
   } else {
      delete pic;
      pic = new TGPicture(name);
      fPicList->Add(pic);
      pic = 0;
   }

   delete [] picnam;
   return pic;
}

//______________________________________________________________________________
const TGPicture *TGPicturePool::GetPicture(const char *name,
                                           UInt_t new_width, UInt_t new_height)
{
   // Get picture with specified size from pool (picture will be scaled if
   // necessary). Picture must be freed using TGPicturePool::FreePicture(). If
   // picture is not found 0 is returned.

   if (!fPicList)
      fPicList = new THashTable(50);

   TString pname = name;

   pname.Strip();
   TString ext = strrchr(pname.Data(), '.');
   ext.ToLower();

   if (ext.Length()) { // ".xpm", ".gif" etc
      char *pxname = gSystem->ExpandPathName(gSystem->UnixPathName(pname));
      pname = pxname;
      delete [] pxname;
   }

   TGPicture *pic;

   const char *hname = TGPicture::HashName(pname, new_width, new_height);
   pic = (TGPicture *)fPicList->FindObject(hname);
   if (pic && pic->GetWidth() == new_width && pic->GetHeight() == new_height) {
      if (pic->fPic == kNone)
         return 0;
      pic->AddReference();
      return pic;
   }

   char *picnam = gSystem->Which(fPath, pname, kReadPermission);
   if (!picnam) {
      return 0;
   }

   if (ext != ".xpm") {
      TImage *img = TImage::Open(picnam);
      if (!img) return 0;

      img->Scale(new_width, new_height);
      pic = new TGPicture(hname, img->GetPixmap(), img->GetMask());
      delete [] picnam;
      delete img;
      fPicList->Add(pic);
      return pic;
   }

   pic = new TGPicture(hname, kTRUE);
   pic->fAttributes.fColormap  = fClient->GetDefaultColormap();
   pic->fAttributes.fCloseness = 40000; // Allow for "similar" colors
   pic->fAttributes.fMask      = kPASize | kPAColormap | kPACloseness;

   Bool_t retc = kFALSE;
   if (!gVirtualX->InheritsFrom("TGX11")) {
      // case of win32gdk and qt drivers
      retc = gVirtualX->CreatePictureFromFile(fClient->GetDefaultRoot()->GetId(),
                                              picnam, pic->fPic, pic->fMask,
                                              pic->fAttributes);
      delete [] picnam;
   } else {
      // case supported by X11 driver
      char **data;
      if (!gVirtualX->ReadPictureDataFromFile(picnam, &data)) {
         delete pic;
         delete [] picnam;
         return 0;
      }
      delete [] picnam;

      Int_t    colors, chars, headersize, totalheight;
      UInt_t   width, height;
      Double_t xscale, yscale;

      sscanf(data[0], "%u %u %d %d", &width, &height, &colors, &chars);
      headersize = colors + 1;
      yscale = (Double_t) new_height / (Double_t) height;
      xscale = (Double_t) new_width / (Double_t) width;
      totalheight = (colors + new_height + 1);

      if ((width != new_width) || (height != new_height)) {
         char **smalldata;
         Int_t    i, x1, y1, pixels;
         Double_t x, y;

         smalldata = new char* [totalheight + 1];

         smalldata[0] = new char[30];
         for (i = 1; i < headersize; i++)
            smalldata[i] = new char [strlen(data[i]) + 1];

         for (i = headersize; i < totalheight + 1; i++)
            smalldata[i] = new char[(new_width * chars) + 1];

         sprintf(smalldata[0], "%u %u %d %d", new_width, new_height, colors, chars);

         for (i = 1; i < headersize; i++) strcpy(smalldata[i], data[i]);

         y = headersize;
         for (y1 = headersize; y1 < (Int_t)new_height + headersize; y1++) {
            x = 0;
            for (x1 = 0; x1 < (Int_t)new_width; x1++) {
               for (pixels = 0; pixels < chars; pixels++)
                  smalldata[y1][x1+pixels] = data[(Int_t)y][(Int_t)x + pixels];
               x += 1.0 / xscale;
            }
            smalldata[y1][x1] = '\0';
            y += 1.0 / yscale;
         }

         retc = gVirtualX->CreatePictureFromData(fClient->GetDefaultRoot()->GetId(), smalldata,
                                                 pic->fPic, pic->fMask,
                                                 pic->fAttributes);

         for (i = 0; i < totalheight + 1; i++)
            delete [] smalldata[i];
         delete [] smalldata;

      } else {
         retc = gVirtualX->CreatePictureFromData(fClient->GetDefaultRoot()->GetId(), data,
                                                 pic->fPic, pic->fMask,
                                                 pic->fAttributes);
      }

      gVirtualX->DeletePictureData(data);
   }

   if (!retc) {
      delete pic;
      return 0;
   }

   fPicList->Add(pic);
   return pic;
}

//______________________________________________________________________________
const TGPicture *TGPicturePool::GetPicture(const char *name, Pixmap_t pxmap, 
                                           Pixmap_t mask)
{
   // ctor

   if (!fPicList) {
      fPicList = new THashTable(50);
   }

   Int_t xy;
   UInt_t w, h;

   gVirtualX->GetWindowSize(pxmap, xy, xy, w, h);

   const char *hname = TGPicture::HashName(name, w, h);
   TGPicture *pic = (TGPicture *)fPicList->FindObject(hname);

   if (pic) {
      pic->AddReference();
      return pic;
   }

   pic = new TGPicture(hname, pxmap, mask);
   fPicList->Add(pic);

   return pic;
}

//______________________________________________________________________________
void TGPicturePool::FreePicture(const TGPicture *fpic)
{
   // Remove picture from cache if nobody is using it anymore.

   if (!fPicList) return;

   TGPicture *pic = (TGPicture *)fPicList->FindObject(fpic);
   if (pic) {
      if (pic->RemoveReference() == 0) {
         fPicList->Remove(pic);
         delete pic;
      }
   }
}

//______________________________________________________________________________
TGPicturePool::~TGPicturePool()
{
   // Delete picture cache.

   if (fPicList) {
      fPicList->Delete();
      delete fPicList;
   }
}

//______________________________________________________________________________
void TGPicturePool::Print(Option_t *) const
{
   // List all pictures in the pool.

   if (fPicList)
      fPicList->Print();
   else
      Info("Print", "no pictures in picture pool");
}

//______________________________________________________________________________
TGPicture::TGPicture(const char *name, Pixmap_t pxmap, Pixmap_t mask)
{
   // ctor

   fName   = name;
   fScaled = kFALSE;
   fPic    = pxmap;
   fMask   = mask;
   Int_t xy;

   fAttributes.fColormap  = gClient->GetDefaultColormap();
   fAttributes.fCloseness = 40000; // Allow for "similar" colors
   fAttributes.fMask      = kPASize | kPAColormap | kPACloseness;
   fAttributes.fPixels    = 0;

   gVirtualX->GetWindowSize(fPic, xy, xy, fAttributes.fWidth, fAttributes.fHeight);
   SetRefCount(1);   
}

//______________________________________________________________________________
void TGPicture::Draw(Handle_t id, GContext_t gc, Int_t x, Int_t y) const
{
   // Draw a picture.

   GCValues_t gcv;

   gcv.fMask = kGCClipMask | kGCClipXOrigin | kGCClipYOrigin;
   gcv.fClipMask = fMask;
   gcv.fClipXOrigin = x;
   gcv.fClipYOrigin = y;
   gVirtualX->ChangeGC(gc, &gcv);
   gVirtualX->CopyArea(fPic, id, gc, 0, 0, fAttributes.fWidth, fAttributes.fHeight,
                  x, y);
   gcv.fMask = kGCClipMask;
   gcv.fClipMask = kNone;
   gVirtualX->ChangeGC(gc, &gcv);
}

//______________________________________________________________________________
TGPicture::~TGPicture()
{
   // Delete picture object.

   if (fPic != kNone)
      gVirtualX->DeletePixmap(fPic);
   if (fMask != kNone)
      gVirtualX->DeletePixmap(fMask);
   if (fAttributes.fPixels)
      delete [] fAttributes.fPixels;
}

//______________________________________________________________________________
const char *TGPicture::HashName(const char *name, Int_t width, Int_t height)
{
   // Static function returning a unique name used to look up a picture.
   // The unique name has the form "name__widthxheight".

   static char hashName[256];

   sprintf(hashName, "%s__%dx%d", name, width, height);

   return hashName;
}

//______________________________________________________________________________
void TGPicture::Print(Option_t *) const
{
   // Print picture info.

   Printf("TGPicture: %s,%sref cnt = %u", GetName(),
          fScaled ? " scaled, " : " ", References());
}


//______________________________________________________________________________
TGSelectedPicture::TGSelectedPicture(const TGClient *client, const TGPicture *p) :
   TGPicture("")
{
   // Create a "selected" looking picture based on the original TGPicture.

   GCValues_t gcv;
   UInt_t     w, h;

   fClient = client;
   Window_t root  = fClient->GetDefaultRoot()->GetId();

   w = p->GetWidth();
   h = p->GetHeight();

   fPic  = gVirtualX->CreatePixmap(root, w, h);
   fMask = p->GetMask();

   fAttributes.fWidth  = w;
   fAttributes.fHeight = h;

   gVirtualX->CopyArea(p->GetPicture(), fPic, GetSelectedGC()(), 0, 0, w, h, 0, 0);

   gcv.fMask = kGCClipMask | kGCClipXOrigin | kGCClipYOrigin;
   gcv.fClipMask = p->GetMask();
   gcv.fClipXOrigin = 0;
   gcv.fClipYOrigin = 0;
   GetSelectedGC().SetAttributes(&gcv);

   gVirtualX->FillRectangle(fPic, GetSelectedGC()(), 0, 0, w, h);

   GetSelectedGC().SetClipMask(kNone);
}

//______________________________________________________________________________
TGSelectedPicture::~TGSelectedPicture()
{
   // Delete selected picture.

   // fMask was borrowed so should not be deleted by ~TGPicture.
   fMask = kNone;
}

//______________________________________________________________________________
TGGC &TGSelectedPicture::GetSelectedGC()
{
   if (!fgSelectedGC) {
      fgSelectedGC = new TGGC(*gClient->GetResourcePool()->GetFrameGC());
      fgSelectedGC->SetForeground(gClient->GetResourcePool()->GetSelectedBgndColor());
      fgSelectedGC->SetBackground(gClient->GetResourcePool()->GetBlackColor());
      fgSelectedGC->SetFillStyle(kFillStippled);
      fgSelectedGC->SetStipple(gClient->GetResourcePool()->GetCheckeredBitmap());
   }
   return *fgSelectedGC;
}
