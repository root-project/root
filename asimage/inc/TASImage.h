// @(#)root/asimage:$Name:$:$Id:$
// Author: Fons Rademakers, Reiner Rohlfs 28/11/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TASImage
#define ROOT_TASImage


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TASImage                                                             //
//                                                                      //
// Interface to image processing library using libAfterImage.           //
// It allows for the reading and writing of images in different         //
// formats, several image manipulations (scaling, tiling, merging,      //
// etc.) and displaying in pads.                                        //
// The size of the image on the screen does not depend on the original  //
// size of the image but on the size of the pad. Therefore it is very   //
// easy to resize the image on the screen by resizing the pad.          //
//                                                                      //
// Beside reading an image from a file an image can be defined by a     //
// two dimensional array of values. A palette defines the color of      //
// each value.                                                          //
//                                                                      //
// The image can be zoomed by defining a rectangle with the mouse.      //
// The color palette can be modified with a GUI, just select            //
// StartPaletteEditor() from the context meny.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TImage
#include "TImage.h"
#endif

struct ASImage;
struct ASVisual;


class TASImage : public TImage {

protected:
   ASImage  *fImage;        //! pointer to image structure of original image
   ASImage  *fScaledImage;  //! pointer to scaled and zoomed image structure
   Double_t  fMaxValue;     // max value in image
   Double_t  fMinValue;     // min value in image
   UInt_t    fZoomOffX;     // X - offset for zooming
   UInt_t    fZoomOffY;     // Y - offset for zooming
   UInt_t    fZoomWidth;    // width of zoomed image
   UInt_t    fZoomHeight;   // hight of zoomed image
   Bool_t    fZoomUpdate;   // kTRUE: new zooming required

   static ASVisual *fgVisual;  // pointer to visual structure
   static Bool_t    fgInit;    // global flag to init afterimage only once

   EImageFileTypes GetFileType(const char *ext);
   void MapFileTypes(EImageFileTypes &type, UInt_t &astype,
                     Bool_t toas = kTRUE);
   void MapQuality(EImageQuality &quality, UInt_t &asquality,
                   Bool_t toas = kTRUE);

   static Bool_t InitVisual();

public:
   TASImage();
   TASImage(const char *file, EImageFileTypes type = kUnknown);
   TASImage(const char *name, const Double_t *imageData, UInt_t width, UInt_t height, TImagePalette *palette = 0);
   TASImage(const char *name, const TArrayD &imageData, UInt_t width, TImagePalette *palette = 0);
   TASImage(const char *name, const TVectorD &imageData, UInt_t width, TImagePalette *palette = 0);
   TASImage(const TASImage &img);
   TASImage &operator=(const TASImage &img);
   virtual ~TASImage();

   // Input / output
   void  ReadImage(const char *file, EImageFileTypes type = kUnknown);
   void  WriteImage(const char *file, EImageFileTypes type = kUnknown);
   void  SetImage(const Double_t *imageData, UInt_t width, UInt_t height, TImagePalette *palette = 0);
   void  SetImage(const TArrayD &imageData, UInt_t width, TImagePalette *palette = 0);
   void  SetImage(const TVectorD &imageData, UInt_t width, TImagePalette *palette = 0);

   // Pad conversions
   void  FromPad(TVirtualPad *pad, Int_t x = 0, Int_t y = 0,
                 UInt_t w = 0, UInt_t h = 0);
   void  Draw(Option_t *option = "");
   void  Paint(Option_t *option = "");
   Int_t DistancetoPrimitive(Int_t px, Int_t py);
   void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
   char *GetObjectInfo(Int_t px, Int_t py) const;

   // Transformations
   void  StartPaletteEditor(); // *MENU*
   void  SetPalette(const TImagePalette *palette);
   void  Scale(UInt_t toWidth, UInt_t toHeight);
   void  Zoom(UInt_t offX, UInt_t offY, UInt_t width, UInt_t height);  //*MENU*
   void  UnZoom();                     //*MENU*
   void  Flip(Int_t flip = 180);       //*MENU*
   void  Mirror(Bool_t vert = kTRUE);  //*MENU*

   // Utilities
   UInt_t GetWidth() const;
   UInt_t GetHeight() const;
   Bool_t IsValid() const { return fImage ? kTRUE : kFALSE; }
   const ASImage *GetImage() const { return fImage; }

   static const ASVisual *GetVisual() { return fgVisual; }

   ClassDef(TASImage,1)  // Image display class
};

#endif
