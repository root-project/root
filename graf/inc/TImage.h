// @(#)root/graf:$Name:$:$Id:$
// Author: Fons Rademakers, Reiner Rohlfs   15/10/2001

/*************************************************************************
 * Copyright (C) 2001-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TImage
#define ROOT_TImage


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TImage                                                               //
//                                                                      //
// Abstract interface to image processing library.                      //
// It allows for the reading and writing of images in different         //
// formats, several image manipulations (scaling, tiling, merging,      //
// etc.) and displaying in pads.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TAttImage
#include "TAttImage.h"
#endif

class TVirtualPad;
class TImage;
class TArrayD;
class TVectorD;


class TImage : public TNamed, public TAttImage {

public:
   // Defines image file types
   enum EImageFileTypes {
      kXpm = 0,
      kZCompressedXpm,
      kGZCompressedXpm,
      kPng,
      kJpeg,
      kXcf,
      kPpm,
      kPnm,
      kBmp,
      kIco,
      kCur,
      kGif,
      kTiff,
      kXbm,
      kFits,
      kUnknown
   };

protected:
   TImage() { }
   TImage(const char *file) : TNamed(file, "") { }

public:
   TImage(const TImage &img) : TNamed(img), TAttImage(img) { }
   TImage &operator=(const TImage &img)
            { TNamed::operator=(img); TAttImage::operator=(img); return *this; }
   virtual ~TImage() { }

   // Input / output
   virtual void ReadImage(const char *file, EImageFileTypes type = kUnknown) = 0;
   virtual void WriteImage(const char *file, EImageFileTypes type = kUnknown) = 0;
   virtual void SetImage(const Double_t *imageData, UInt_t width, UInt_t height, TImagePalette *palette = 0) = 0;
   virtual void SetImage(const TArrayD &imageData, UInt_t width, TImagePalette *palette = 0) = 0;
   virtual void SetImage(const TVectorD &imageData, UInt_t width, TImagePalette *palette = 0) = 0;

   // Pad conversions
   virtual void FromPad(TVirtualPad *pad, Int_t x = 0, Int_t y = 0, UInt_t w = 0, UInt_t h = 0) = 0;

   // Transformations
   virtual void Scale(UInt_t toWidth, UInt_t toHeight) = 0;
   virtual void Zoom(UInt_t offX, UInt_t offY, UInt_t width, UInt_t height) = 0; //*MENU*
   virtual void UnZoom() = 0;                        //*MENU*
   virtual void Flip(Int_t flip = 180) = 0;          //*MENU*
   virtual void Mirror(Bool_t vert = kTRUE) = 0;     //*MENU*

   // Utilities
   virtual UInt_t GetWidth() const = 0;
   virtual UInt_t GetHeight() const = 0;
   virtual Bool_t IsValid() const = 0;

   static TImage *Create();
   static TImage *Open(const char *file, EImageFileTypes type = kUnknown);
   static TImage *Open(const char *name, const Double_t *imageData, UInt_t width, UInt_t height, TImagePalette *palette);
   static TImage *Open(const char *name, const TArrayD &imageData, UInt_t width, TImagePalette *palette = 0);
   static TImage *Open(const char *name, const TVectorD &imageData, UInt_t width, TImagePalette *palette = 0);

   ClassDef(TImage,1)  // Abstract image class
};

#endif
