// @(#)root/asimage:$Name:  $:$Id: TASImage.cxx,v 1.13 2004/12/07 17:01:39 brun Exp $
// Author: Fons Rademakers, Reiner Rohlfs   28/11/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**************************************************************************
 * The part of this source is based on AfterStep-2.00.00 code
 *          (http://www.afterstep.org/)
 *
 * Copyright (c) 2002 Sasha Vasko <sasha@aftercode.net>
 * Copyright (c) 1998, 1999 Ethan Fischer <allanon@crystaltokyo.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
**************************************************************************/

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
// Besides reading an image from a file an image can be defined by a    //
// two dimensional array of values. A palette defines the color of      //
// each value.                                                          //
//                                                                      //
// The image can be zoomed by defining a rectangle with the mouse.      //
// The color palette can be modified with a GUI, just select            //
// StartPaletteEditor() from the context menu.                          //
//                                                                      //
// There are two methods for polygon filling:                           //
//    FillPolygon  - fill convex  polygon                               //
//    DrawFillArea - general polygon filling according "Even-Odd Rule". //
//    Both methods has correspondent TVirtualX ones.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TASImage.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TArrayD.h"
#include "TVectorD.h"
#include "TVirtualPS.h"
#include "TGaxis.h"
#include "TColor.h"
#include "TObjArray.h"
#include "TASPaletteEditor.h"
#include "TArrayL.h"
#include "TPoint.h"


#ifndef WIN32
#   include <X11/Xlib.h>
#else
#   include "Windows4root.h"
#endif
extern "C" {
#ifndef WIN32
#   include <afterbase.h>
#   include <afterimage.h>
#else
#   include <win32/config.h>
#   include <win32/afterbase.h>
#   include <afterimage.h>
#   include <bmp.h>
#   define X_DISPLAY_MISSING 1
#endif
    extern Display *dpy;    // defined in afterbase.c
}


ASVisual *TASImage::fgVisual;
Bool_t TASImage::fgInit = kFALSE;

static ASFontManager *gFontManager = 0;

typedef struct {
   unsigned char b;
   unsigned char g;
   unsigned char r;
   unsigned char a;
} __argb32__;

static __argb32__ *t;
static __argb32__ *b;

//______________________________________________________________________________
inline void _alphaBlend(ARGB32 *bot, ARGB32 *top)
{
   // auxilary function providing alphablending for vector graphics

   t = (__argb32__*)top;
   b = (__argb32__*)bot;

   b->a = b->a*(255-t->a)>>8 + t->a;
   b->r = (b->r*(255-t->a)+t->r*t->a)>>8;
   b->g = (b->g*(255-t->a)+t->g*t->a)>>8;
   b->b = (b->b*(255-t->a)+t->b*t->a)>>8;
}


ClassImp(TASImage)


//______________________________________________________________________________
TASImage::TASImage()
{
   // Default image ctor.

   fImage       = 0;
   fScaledImage = 0;
   fEditable    = kFALSE;
   fPaintMode   = kTRUE;

   if (!fgInit) {
      set_application_name((char*)(gProgName ? gProgName : "ROOT"));
      fgInit = kTRUE;
   }
}

//______________________________________________________________________________
TASImage::TASImage(const char *file, EImageFileTypes) : TImage(file)
{
   // Create an image object and read from specified file.
   // For more information see description of function ReadImage()
   // which is called by this constructor.

   fImage       = 0;
   fScaledImage = 0;
   fEditable    = kFALSE;
   fPaintMode   = kTRUE;

   if (!fgInit) {
      set_application_name((char*)(gProgName ? gProgName : "ROOT"));
      fgInit = kTRUE;
   }

   ReadImage(file);
}

//______________________________________________________________________________
TASImage::TASImage(const char *name, const Double_t *imageData, UInt_t width,
                   UInt_t height, TImagePalette *palette) : TImage(name)
{
   // Creates an image depending on the values of imageData.
   // For more information see function SetImage() which is called
   // by this constructor.

   fImage       = 0;
   fScaledImage = 0;
   fEditable    = kFALSE;
   fPaintMode   = kTRUE;

   if (!fgInit) {
      set_application_name((char*)(gProgName ? gProgName : "ROOT"));
      fgInit = kTRUE;
   }

   SetImage(imageData, width, height, palette);
}

//______________________________________________________________________________
TASImage::TASImage(const char *name, const TArrayD &imageData, UInt_t width,
                   TImagePalette *palette) : TImage(name)
{
   // Creates an image depending on the values of imageData. The size
   // of the image is width X (imageData.fN / width).
   // For more information see function SetImage() which is called by
   // this constructor.

   fImage       = 0;
   fScaledImage = 0;
   fEditable    = kFALSE; 
   fPaintMode   = kTRUE;

   if (!fgInit) {
      set_application_name((char*)(gProgName ? gProgName : "ROOT"));
      fgInit = kTRUE;
   }

   SetImage(imageData, width, palette);
}

//______________________________________________________________________________
TASImage::TASImage(const char *name, const TVectorD &imageData, UInt_t width,
                   TImagePalette *palette) : TImage(name)
{
   // Creates an image depending on the values of imageData. The size
   // of the image is width X (imageData.fN / width).
   // For more information see function SetImage() which is called by
   // this constructor.

   fImage       = 0;
   fScaledImage = 0;
   fEditable    = kFALSE;
   fPaintMode   = kTRUE;

   if (!fgInit) {
      set_application_name((char*)(gProgName ? gProgName : "ROOT"));
      fgInit = kTRUE;
   }

   SetImage(imageData, width, palette);
}

//______________________________________________________________________________
TASImage::TASImage(const TASImage &img) : TImage(img)
{
   // Image copy ctor.

   fImage       = 0;
   fScaledImage = fScaledImage ? (TASImage*)fScaledImage->Clone("") : 0;
   fPaintMode   = kTRUE;

   if (img.IsValid()) {
      fImage = clone_asimage(img.fImage, SCL_DO_ALL);
      if (img.fImage->alt.vector) {
         Int_t size = img.fImage->width * img.fImage->height * sizeof(double);
         fImage->alt.vector = (double*)malloc(size);
         memcpy(fImage->alt.vector, img.fImage->alt.vector, size);
      }

      fZoomUpdate = kZoom;
      fZoomOffX   = img.fZoomOffX;
      fZoomOffY   = img.fZoomOffY;
      fZoomWidth  = img.fZoomWidth;
      fZoomHeight = img.fZoomHeight;
      fEditable   = img.fEditable;
   }
}

//______________________________________________________________________________
TASImage &TASImage::operator=(const TASImage &img)
{
   // Image assignment operator.

   if (this != &img && img.IsValid()) {
      TImage::operator=(img);
      if (fImage)
         destroy_asimage(&fImage);
      fImage = clone_asimage(img.fImage, SCL_DO_ALL);
      if (img.fImage->alt.vector) {
         Int_t size = img.fImage->width * img.fImage->height * sizeof(double);
         fImage->alt.vector = (double*)malloc(size);
         memcpy(fImage->alt.vector, img.fImage->alt.vector, size);
      }

      fScaledImage = img.fScaledImage ? (TASImage*)img.fScaledImage->Clone("") : 0;
      fZoomUpdate = kZoom;
      fZoomOffX   = img.fZoomOffX;
      fZoomOffY   = img.fZoomOffY;
      fZoomWidth  = img.fZoomWidth;
      fZoomHeight = img.fZoomHeight;
      fEditable   = img.fEditable;
   }
   fPaintMode   = kTRUE;

   return *this;
}

//______________________________________________________________________________
TASImage::~TASImage()
{
   // Image dtor, clean up image and visual.

   if (fImage)
      destroy_asimage(&fImage);

   delete fScaledImage;
   fScaledImage = 0;
}

//______________________________________________________________________________
void TASImage::ReadImage(const char *file, EImageFileTypes /*type*/)
{
   // Read specified image file. The file type is determined by
   // the file extension (the type argument is ignored). It will
   // attempt to append .gz and then .Z to the filename and find such
   // a file. If the filename ends with extension consisting of digits
   // only, it will attempt to find the file with this extension stripped
   // off. On success this extension will be used to load subimage from
   // the file with that number. Subimage is supported only for GIF files.

   if (fImage) {
      destroy_asimage(&fImage);
   }

   delete fScaledImage;
   fScaledImage = 0;

   fImage = file2ASImage(file, 0, SCREEN_GAMMA, GetImageCompression(), 0);

   fZoomUpdate = kNoZoom;
   fEditable   = kFALSE;
   fZoomOffX   = 0;
   fZoomOffY   = 0;
   fZoomWidth  = fImage->width;
   fZoomHeight = fImage->height;

   SetName(file);
}

//______________________________________________________________________________
void TASImage::WriteImage(const char *file, EImageFileTypes type)
{
   // Write image to specified file. If there is no file extension or
   // if the file extension is unknown, the type argument will be used
   // to determine the file type. The quality and compression is derived from
   // the TAttImage values.
   // The size of the image in the file is independent of the actually
   // displayed size and zooming factor on the screen. This function
   // writes always the original image with its size in the file

   if (!IsValid()) {
      Error("WriteImage", "no image loaded");
      return;
   }

   if (!file || !*file) {
      Error("WriteImage", "no file name specified");
      return;
   }

   const char *s;
   if ((s = strrchr(file, '.'))) {
      s++;
      EImageFileTypes t = GetFileType(s);
      if (t == kUnknown && type == kUnknown) {
         Error("WriteImage", "cannot determine a valid file type");
         return;
      }
      if (t != kUnknown)
         type = t;
   }

   if (type == kUnknown) {
      Error("WriteImage", "not a valid file type was specified");
      return;
   }

   ASImageFileTypes atype;
   MapFileTypes(type, (UInt_t&)atype);

   UInt_t aquality;
   EImageQuality quality = GetImageQuality();
   MapQuality(quality, aquality);

   ASImageExportParams parms;
   switch (type) {
   case kXpm:
      parms.xpm.type = atype;
      parms.xpm.flags = EXPORT_ALPHA;
      parms.xpm.dither = 4;
      parms.xpm.opaque_threshold = 127;
      parms.xpm.max_colors = 512;
      break;
   case kPng:
      parms.png.type = atype;
      parms.png.flags = EXPORT_ALPHA;
      parms.png.compression = !GetImageCompression() ? -1 : int(GetImageCompression());
      break;
   case kJpeg:
      parms.jpeg.type = atype;
      parms.jpeg.flags = 0;
      parms.jpeg.quality = aquality;
      break;
   case kGif:
      parms.gif.type = atype;
      parms.gif.flags = EXPORT_ALPHA;
      parms.gif.dither = 0;
      parms.gif.opaque_threshold = 0;
      break;
   case kTiff:
      parms.tiff.type = atype;
      parms.tiff.flags = EXPORT_ALPHA;
      parms.tiff.rows_per_strip = 0;
      parms.tiff.compression_type = aquality <= 50 ? TIFF_COMPRESSION_JPEG :
                                                     TIFF_COMPRESSION_NONE;
      parms.tiff.jpeg_quality = 100;
      parms.tiff.opaque_threshold = 0;
      break;
   default:
      Error("WriteImage", "file type %s not yet supported", s);
      return;
   }

   if (!ASImage2file(fScaledImage ? fScaledImage->fImage : fImage, 0, file, atype, &parms))
      Error("WriteImage", "error writing file %s", file);
}

//______________________________________________________________________________
TImage::EImageFileTypes TASImage::GetFileType(const char *ext)
{
   // Return file type depending on specified extension.
   // Protected method.

   TString s(ext);
   s.Strip();
   s.ToLower();

   if (s == "xpm")
      return kXpm;
   if (s == "png")
      return kPng;
   if (s == "jpg" || s == "jpeg")
      return kJpeg;
   if (s == "xcf")
      return kXcf;
   if (s == "ppm")
      return kPpm;
   if (s == "pnm")
      return kPnm;
   if (s == "bmp")
      return kBmp;
   if (s == "ico")
      return kIco;
   if (s == "cur")
      return kCur;
   if (s == "gif")
      return kGif;
   if (s == "tiff")
      return kTiff;
   if (s == "xbm")
      return kXbm;
   if (s == "tga")
      return kTga;
   if (s == "xml")
      return kXml;

   return kUnknown;
}

//______________________________________________________________________________
void TASImage::MapFileTypes(EImageFileTypes &type, UInt_t &astype, Bool_t toas)
{
   // Map file type to/from AfterImage types.
   // Protected method.

   if (toas) {
      switch (type) {
         case kXpm:
            astype = ASIT_Xpm; break;
         case kZCompressedXpm:
            astype = ASIT_ZCompressedXpm; break;
         case kGZCompressedXpm:
            astype = ASIT_GZCompressedXpm; break;
         case kPng:
            astype = ASIT_Png; break;
         case kJpeg:
            astype = ASIT_Jpeg; break;
         case kXcf:
            astype = ASIT_Xcf; break;
         case kPpm:
            astype = ASIT_Ppm; break;
         case kPnm:
            astype = ASIT_Pnm; break;
         case kBmp:
            astype = ASIT_Bmp; break;
         case kIco:
            astype = ASIT_Ico; break;
         case kCur:
            astype = ASIT_Cur; break;
         case kGif:
            astype = ASIT_Gif; break;
         case kTiff:
            astype = ASIT_Tiff; break;
         case kXbm:
            astype = ASIT_Xbm; break;
         case kTga:
            astype = ASIT_Targa; break;
         case kXml:
            astype = ASIT_XMLScript; break;
         default:
            astype = ASIT_Unknown;
      }
   } else {
      switch (astype) {
         case ASIT_Xpm:
            type = kXpm; break;
         case ASIT_ZCompressedXpm:
            type = kZCompressedXpm; break;
         case ASIT_GZCompressedXpm:
            type = kGZCompressedXpm; break;
         case ASIT_Png:
            type = kPng; break;
         case ASIT_Jpeg:
            type = kJpeg; break;
         case ASIT_Xcf:
            type = kXcf; break;
         case ASIT_Ppm:
            type = kPpm; break;
         case ASIT_Pnm:
            type = kPnm; break;
         case ASIT_Bmp:
            type = kBmp; break;
         case ASIT_Ico:
            type = kIco; break;
         case ASIT_Cur:
            type = kCur; break;
         case ASIT_Gif:
            type = kGif; break;
         case ASIT_Tiff:
            type = kTiff; break;
         case ASIT_Xbm:
            type = kXbm; break;
         case ASIT_XMLScript:
            type = kXml; break;
         case ASIT_Targa:
            type = kTga; break;
         default:
            type = kUnknown;
      }
   }
}

//______________________________________________________________________________
void TASImage::MapQuality(EImageQuality &quality, UInt_t &asquality, Bool_t toas)
{
   // Map quality to/from AfterImage quality.
   // Protected method.

   if (toas) {
      switch (quality) {
         case kImgPoor:
            asquality = 25; break;
         case kImgFast:
            asquality = 75; break;
         case kImgGood:
            asquality = 50; break;
         case kImgBest:
            asquality = 100; break;
         default:
            asquality = 0;
      }
   } else {
      quality = kImgDefault;
      if (asquality > 0  && asquality <= 25)
         quality = kImgPoor;
      if (asquality > 26 && asquality <= 50)
         quality = kImgFast;
      if (asquality > 51 && asquality <= 75)
         quality = kImgGood;
      if (asquality > 76 && asquality <= 100)
         quality = kImgBest;
   }
}

//______________________________________________________________________________
void TASImage::SetImage(const Double_t *imageData, UInt_t width, UInt_t height,
                        TImagePalette *palette)
{
   // Deletes the old image and creates a new image depending on the values
   // of imageData. The size of the image is width X height.
   // The color of each pixel depends on the imageData of the corresponding
   // pixel. The palette is used to convert an image value into its color.
   // If palette is not defined (palette = 0) a default palette is used.
   // Any previously defined zooming is reset.

   TAttImage::SetPalette(palette);

   if (!InitVisual()) {
      Warning("SetImage", "Visual not initiated");
      return;
   }

   if (fImage)
      destroy_asimage(&fImage);

   delete fScaledImage;
   fScaledImage = 0;

   // get min and max value of image
   fMinValue = fMaxValue = *imageData;
   for (Int_t pixel = 1; pixel < Int_t(width * height); pixel++) {
      if (fMinValue > *(imageData + pixel)) fMinValue = *(imageData + pixel);
      if (fMaxValue < *(imageData + pixel)) fMaxValue = *(imageData + pixel);
   }

   // copy ROOT palette to asImage palette
   const TImagePalette &pal = GetPalette();

   ASVectorPalette asPalette;

   asPalette.npoints = pal.fNumPoints;
   Int_t col;
   for (col = 0; col < 4; col++)
      asPalette.channels[col] = new UShort_t[asPalette.npoints];

   memcpy(asPalette.channels[0], pal.fColorBlue,  pal.fNumPoints * sizeof(UShort_t));
   memcpy(asPalette.channels[1], pal.fColorGreen, pal.fNumPoints * sizeof(UShort_t));
   memcpy(asPalette.channels[2], pal.fColorRed,   pal.fNumPoints * sizeof(UShort_t));
   memcpy(asPalette.channels[3], pal.fColorAlpha, pal.fNumPoints * sizeof(UShort_t));

   asPalette.points = new Double_t[asPalette.npoints];
   for (Int_t point = 0; point < Int_t(asPalette.npoints); point++)
      asPalette.points[point] = fMinValue + (fMaxValue - fMinValue) * pal.fPoints[point];

   fImage = create_asimage_from_vector(fgVisual, (Double_t*)imageData, width,
                                       height, &asPalette, ASA_ASImage,
                                       GetImageCompression(), GetImageQuality());

   delete [] asPalette.points;
   for (col = 0; col < 4; col++)
      delete [] asPalette.channels[col];

   fZoomUpdate = 0;
   fZoomOffX   = 0;
   fZoomOffY   = 0;
   fZoomWidth  = width;
   fZoomHeight = height;
}

//______________________________________________________________________________
void TASImage::SetImage(const TArrayD &imageData, UInt_t width, TImagePalette *palette)
{
   // Deletes the old image and creates a new image depending on the values
   // of imageData. The size of the image is width X (imageData.fN / width).
   // The color of each pixel depends on the imageData of the corresponding
   // pixel. The palette is used to convert an image value into its color.
   // If palette is not defined (palette = 0) a default palette is used.
   // Any previously defined zooming is reset.

   SetImage(imageData.GetArray(), width, imageData.GetSize() / width, palette);
}

//______________________________________________________________________________
void TASImage::SetImage(const TVectorD &imageData, UInt_t width, TImagePalette *palette)
{
   // Deletes the old image and creates a new image depending on the values
   // of imageData. The size of the image is width X (imageData.fN / width).
   // The color of each pixel depends on the imageData of the corresponding
   // pixel. The palette is used to convert an image value into its color.
   // If palette is not defined (palette = 0) a default palette is used.
   // Any previously defined zooming is reset.

   SetImage(imageData.GetMatrixArray(), width,
            imageData.GetNoElements() / width, palette);
}

//______________________________________________________________________________
void TASImage::FromPad(TVirtualPad *pad, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Create an image from the given pad, afterwards this image can be
   // saved in any of the supported image formats.

   if (!pad) {
      Error("FromPad", "pad cannot be 0");
      return;
   }

   if (!InitVisual()) {
      Warning("FromPad", "Visual not initiated");
      return;
   }

   if (fImage)
      destroy_asimage(&fImage);

   delete fScaledImage;
   fScaledImage = 0;

   SetName(pad->GetName());

   if (w == 0)
      w = pad->UtoPixel(1.);
   if (h == 0)
      h = pad->VtoPixel(0.);

   Int_t wid = (pad == pad->GetCanvas()) ? pad->GetCanvas()->GetCanvasID()
                                         : pad->GetPixmapID();
   Window wd = (Window) gVirtualX->GetWindowID(wid);

#ifndef WIN32
   fImage = pixmap2asimage(fgVisual, wd, x, y, w, h, AllPlanes, 0, 0);
#else
   unsigned char *bits = (gGetBmBits != 0) ? gGetBmBits(wd, w, h) : 0;
   fImage = bitmap2asimage (bits, w, h, 0);
#endif
}

//______________________________________________________________________________
void TASImage::Draw(Option_t *option)
{
   // Draw image. Support the following drawing options:
   // "T[x,y[,tint]]" - tile image (use specified offset and tint),
   //                   e.g. "T100,100,#556655"
   //                   with this option the zooming is not possible
   //                   and disabled
   // "N"             - display in new canvas (of original image size)
   // "xxx"           - image is drawn exapnded to pad size
   //
   // The default is to display the image in the current gPad.

   static Bool_t calcBorder = kTRUE;
   static UInt_t bw = 0;
   static UInt_t bh = 0;

   TString opt = option;
   opt.ToLower();
   if (opt.Contains("n") || !gPad || !gPad->IsEditable()) {
      TCanvas *c = new TCanvas(GetName(), Form("%s (%d x %d)", GetName(),
                               fImage->width, fImage->height),
                               fImage->width+bw, fImage->height+bh);
      if (calcBorder) {
         bw = c->GetWindowWidth() - c->GetWw();
         bh = c->GetWindowHeight() - c->GetWh();
         c->SetWindowSize(fImage->width+bw, fImage->height+bh);
         calcBorder = kFALSE;
      }
   }

   Double_t left =    gPad->GetLeftMargin();
   Double_t right =   gPad->GetRightMargin();
   Double_t top =     gPad->GetTopMargin();
   Double_t bottom =  gPad->GetBottomMargin();
   gPad->Range(-left / (1.0 - left - right), 
               -bottom / (1.0 - top - bottom), 
               1 + right / (1.0 - left - right),
               1 + top / ( 1.0 - top - bottom));
   gPad->RangeAxis(0,0,1,1);
   TObject::Draw(option);
}

//______________________________________________________________________________
void TASImage::Paint(Option_t *option)
{
   // Paint image in current pad. See Draw() function for drawing options.
   //
   // options:
   // "T[x,y[,tint]]" - tile image (use specified offset and tint),
   //                   e.g. "T100,100,#556655"
   //                   with this option the zooming is not possible
   //                   and disabled
   // "xxx" - image expanded to pad size

#ifdef WIN32
   void *bmbits = NULL ;
   BITMAPINFO *bmi = NULL ;
#endif

   if (!fImage) {
      Error("Paint", "no image set");
      return;
   }

   if (!InitVisual()) {
      Warning("Paint", "Visual not initiated");
      return;
   }

   Int_t   tile_x = 0, tile_y = 0;
   ULong_t tile_tint = 0;
   Bool_t  tile = kFALSE;
   Bool_t  expand = kFALSE;

   TString opt = option;
   opt.ToLower();
   if (opt.Contains("t")) {
      char stint[64];
      if (sscanf(opt.Data()+opt.Index("t"), "t%d,%d,%s", &tile_x, &tile_y,
                 stint) <= 3) {
         tile = kTRUE;
         if (parse_argb_color(stint, (CARD32*) &tile_tint) == stint)
            tile_tint = 0;
      } else
         Error("Paint", "tile option error");
   } else if (opt.Contains("xxx")) {
      expand = kTRUE;
   }

   ASImage *image = fImage;

   // Get geometry of pad
   Int_t to_w = gPad->UtoPixel(1.);
   Int_t to_h = gPad->VtoPixel(0.);

   Int_t mw = 0;
   Int_t mh = 0;

   // remove the size by the margin of the pad
   if (!expand) {
      mw = int(gPad->UtoPixel(1.) * gPad->GetLeftMargin() + 0.5);
      mh = int(gPad->VtoPixel(0.) * gPad->GetTopMargin() + 0.5);
      to_h  -= 2*mh;
      to_w  -= 2*mw;
   }

   // upper left corner and size of the palette in pixels
   Int_t pal_Ax = gPad->XtoAbsPixel(1.0) + 5;
   Int_t pal_Ay = gPad->YtoAbsPixel(1.0) + 20;
   Int_t pal_x = gPad->XtoPixel(1.0) + 5;
   Int_t pal_y = gPad->YtoPixel(1.0) + 20;
   Int_t pal_w = gPad->UtoPixel(gPad->GetRightMargin()) / 3;
   Int_t pal_h = to_h - 20;

   if ((to_w < 25 || to_h < 25) && !expand) {
      Error("Paint", "pad too small to display an image");
      return;
   }

   if (GetConstRatio()) {
      if ((Double_t)to_w / (Double_t)fZoomWidth <
          (Double_t)to_h / (Double_t)fZoomHeight)
         to_h = Int_t(Double_t(fZoomHeight) * to_w / fZoomWidth);
      else
         to_w = Int_t(Double_t(fZoomWidth) * to_h / fZoomHeight);
   }

   ASImage *grad_im = 0;
   if (fImage->alt.vector) {
      // draw the palette
      ASGradient grad;
      const TImagePalette &pal = GetPalette();

      grad.npoints = pal.fNumPoints;
      grad.type    = GRADIENT_Top2Bottom;
      grad.color   = new ARGB32[grad.npoints];
      grad.offset  = new double[grad.npoints];

      for (Int_t pt = 0; pt < grad.npoints; pt++) {
         Int_t oldPt = grad.npoints - pt -1;
         grad.offset[pt] = 1 - pal.fPoints[oldPt];
         grad.color[pt] = (((ARGB32)(pal.fColorBlue[oldPt]  & 0xff00)) >>  8) |
                          (((ARGB32)(pal.fColorGreen[oldPt] & 0xff00))      ) |
                          (((ARGB32)(pal.fColorRed[oldPt]   & 0xff00)) <<  8) |
                          (((ARGB32)(pal.fColorAlpha[oldPt] & 0xff00)) << 16);
      }

      grad_im = make_gradient(fgVisual, &grad , UInt_t(pal_w),
                              pal_h, SCL_DO_COLOR,
                              ASA_ASImage, GetImageCompression(), GetImageQuality());

      delete [] grad.color;
      delete [] grad.offset;
   }

   if (tile) {
      delete fScaledImage;
      fScaledImage = (TASImage*)TImage::Create();
      fScaledImage->fImage = tile_asimage(fgVisual, fImage, tile_x, tile_y,
                                          to_w, to_h, tile_tint, ASA_ASImage,
                                          GetImageCompression(), GetImageQuality());
      image = fScaledImage->fImage;
   } else if (fZoomUpdate == kZoomOps) {
      image = fImage;

   } else {

      // Scale and zoom image if needed
      if (Int_t(fImage->width) != to_w || Int_t(fImage->height) != to_h ||
          fImage->width != fZoomWidth || fImage->height != fZoomHeight) {

         if (fScaledImage && (fZoomUpdate == kZoom)) {
            delete fScaledImage;
            fScaledImage = 0;
         }

         if (!fScaledImage) {
            fScaledImage = (TASImage*)TImage::Create();

            if (fImage->width != fZoomWidth || fImage->height != fZoomHeight) {
               // zoom and scale image
               ASImage *tmpImage = 0;
               tmpImage = tile_asimage(fgVisual, fImage, fZoomOffX,
                                       fImage->height - fZoomHeight - fZoomOffY,
                                       fZoomWidth, fZoomHeight, 0, ASA_ASImage,
                                       GetImageCompression(), GetImageQuality());

               fScaledImage->fImage = scale_asimage(fgVisual, tmpImage, to_w, to_h,
                                                   ASA_ASImage, GetImageCompression(),
                                                   GetImageQuality());
               destroy_asimage(&tmpImage);
            } else {
               // scale image, no zooming
               fScaledImage->fImage = scale_asimage(fgVisual, fImage, to_w, to_h,
                                                    ASA_ASImage, GetImageCompression(),
                                                    GetImageQuality());
            }
         }
         image = fScaledImage->fImage;
      }
   }
   fZoomUpdate = 0;

   if (!image) {
      Error("Paint", "image could not be rendered to display");
      return;
   }
#ifndef WIN32
   Pixmap pxmap = asimage2pixmap(fgVisual, gVirtualX->GetDefaultRootWindow(),
                                 image, 0, kTRUE);
   Int_t wid = gVirtualX->AddWindow(pxmap, to_w, to_h);
   gPad->cd();
   gVirtualX->CopyPixmap(wid, mw, mh);

   gVirtualX->RemoveWindow(wid);
   gVirtualX->DeletePixmap(pxmap);
#else
   // Convert ASImage into DIB: 
   bmi = ASImage2DBI( fgVisual, image, 0, 0, image->width, image->height, &bmbits );
   gPad->cd();
   if(gDrawDIB != 0) {
      gDrawDIB((ULong_t)bmi, (ULong_t)bmbits,
          (int)(gPad->UtoPixel(1.) * gPad->GetLeftMargin() + 0.5),  
          (int)(gPad->VtoPixel(0.) * gPad->GetTopMargin() + 0.5));
      free(bmbits);
      free(bmi);
   }
#endif
   gPad->cd();

   if (grad_im) {
#ifndef WIN32
      // draw color bar
      pxmap = asimage2pixmap(fgVisual, gVirtualX->GetDefaultRootWindow(),
                             grad_im, 0, kTRUE);
      wid = gVirtualX->AddWindow(pxmap, UInt_t(pal_w), pal_h);

      gPad->cd();
      gVirtualX->CopyPixmap(wid, pal_x, pal_y);
      gVirtualX->RemoveWindow(wid);
      gVirtualX->DeletePixmap(pxmap);
#else
      // Convert ASImage into DIB: 
      bmi = ASImage2DBI( fgVisual, grad_im, 0, 0, grad_im->width, grad_im->height, &bmbits );
      gPad->cd();
      if(gDrawDIB != 0) {
         gDrawDIB((ULong_t)bmi, (ULong_t)bmbits, pal_x, pal_y);
         free(bmbits);
         free(bmi);
      }
#endif
      gPad->cd();

      // values of palette
      TGaxis axis;
      Int_t ndiv = 510;
      double min = fMinValue;
      double max = fMaxValue;
      axis.SetLineColor(0);       // draw white ticks
      Double_t pal_Xpos = gPad->AbsPixeltoX(pal_Ax + pal_w);
      axis.PaintAxis(pal_Xpos, gPad->PixeltoY(pal_Ay + pal_h - 1),
                     pal_Xpos, gPad->PixeltoY(pal_Ay),
                     min, max, ndiv, "+LU");
      min = fMinValue;
      max = fMaxValue;
      axis.SetLineColor(1);       // draw black ticks
      axis.PaintAxis(pal_Xpos, gPad->AbsPixeltoY(pal_Ay + pal_h),
                     pal_Xpos, gPad->AbsPixeltoY(pal_Ay + 1),
                     min, max, ndiv, "+L");

   }

   // loop over pxmap and draw image to PostScript
   if (gVirtualPS) {

      // get special color cell to be reused during image printing
      TObjArray *colors = (TObjArray*) gROOT->GetListOfColors();
      TColor *color = 0;
      // Look for color by name
      if ((color = (TColor*)colors->FindObject("Image_PS")) == 0)
         color = new TColor(colors->GetEntries(), 1., 1., 1., "Image_PS");

      gVirtualPS->SetFillColor(color->GetNumber());
      gVirtualPS->SetFillStyle(1001);

      Double_t xconv = (gPad->AbsPixeltoX(to_w) - gPad->AbsPixeltoX(0)) / image->width;
      Double_t yconv = (gPad->AbsPixeltoY(0) - gPad->AbsPixeltoY(to_h)) / image->height;
      Double_t x1 = 0;
      Double_t x2 = 1 * xconv;
      Double_t y2 = 1;
      Double_t y1 = 1 - yconv;
      gVirtualPS->CellArrayBegin(image->width, image->height, x1, x2, y1, y2);

      ASImageDecoder *imdec = start_image_decoding(fgVisual, image, SCL_DO_ALL,
                                                   0, 0, image->width, image->height, 0);
      for (Int_t yt = 0; yt < (Int_t)image->height; yt++) {
         imdec->decode_image_scanline(imdec);
         for (Int_t xt = 0; xt < (Int_t)image->width; xt++)
            gVirtualPS->CellArrayFill(imdec->buffer.red[xt],
                                      imdec->buffer.green[xt],
                                      imdec->buffer.blue[xt]);
      }
      stop_image_decoding(&imdec);
      gVirtualPS->CellArrayEnd();

      // print the color bar
      if (grad_im) {
         xconv = (gPad->AbsPixeltoX(pal_Ax + pal_w) - gPad->AbsPixeltoX(pal_Ax)) / grad_im->width;
         yconv = (gPad->AbsPixeltoY(pal_Ay - pal_h) - gPad->AbsPixeltoY(pal_Ay)) / grad_im->height;
         x1 = gPad->AbsPixeltoX(pal_Ax);
         x2 = x1 + xconv;
         y2 = gPad->AbsPixeltoY(pal_Ay);
         y1 = y2 - yconv;
         gVirtualPS->CellArrayBegin(grad_im->width, grad_im->height,
                                    x1, x2, y1, y2);

         imdec = start_image_decoding(fgVisual, grad_im, SCL_DO_ALL,
                                      0, 0, grad_im->width, grad_im->height, 0);
         for (Int_t yt = 0; yt < (Int_t)grad_im->height; yt++) {
            imdec->decode_image_scanline(imdec);
            for (Int_t xt = 0; xt < (Int_t)grad_im->width; xt++)
               gVirtualPS->CellArrayFill(imdec->buffer.red[xt],
                                         imdec->buffer.green[xt],
                                         imdec->buffer.blue[xt]);
         }
         stop_image_decoding(&imdec);
         gVirtualPS->CellArrayEnd();

         // values of palette
         TGaxis axis;
         Int_t ndiv = 510;
         double min = fMinValue;
         double max = fMaxValue;
         axis.SetLineColor(1);       // draw black ticks
         Double_t pal_Xpos = gPad->AbsPixeltoX(pal_Ax + pal_w);
         axis.PaintAxis(pal_Xpos, gPad->AbsPixeltoY(pal_Ay + pal_h),
                        pal_Xpos, gPad->AbsPixeltoY(pal_Ay + 1),
                        min, max, ndiv, "+L");

      }
   }

   if (grad_im)
      destroy_asimage(&grad_im);
}

//______________________________________________________________________________
Int_t TASImage::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Is the mouse in the image?
                       
   Int_t pxl, pyl, pxt, pyt;

   Int_t px1 = gPad->XtoAbsPixel(0);
   Int_t py1 = gPad->YtoAbsPixel(0);
   Int_t px2 = gPad->XtoAbsPixel(1);
   Int_t py2 = gPad->YtoAbsPixel(1);

   if (px1 < px2) {pxl = px1; pxt = px2;}
   else           {pxl = px2; pxt = px1;}
   if (py1 < py2) {pyl = py1; pyt = py2;}
   else           {pyl = py2; pyt = py1;}

   if ((px > pxl && px < pxt) && (py > pyl && py < pyt))
      return 0;

   return 999999;
}

//______________________________________________________________________________
void TASImage::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute mouse events.

   if (IsEditable()) {
      gPad->ExecuteEvent(event, px, py);
      return;
   }

   gPad->SetCursor(kCross);

   static Int_t stx, sty;
   static Int_t oldx, oldy;

   if (!IsValid()) return;

   if (event == kButton1Motion || event == kButton1Down  ||
       event == kButton1Up) {

      // convert to image pixel on screen
      Int_t imgX = px - gPad->XtoAbsPixel(0);
      Int_t imgY = py - gPad->YtoAbsPixel(1);

      if (imgX < 0)  px = px - imgX;
      if (imgY < 0)  py = py - imgY;

      ASImage *image = fImage;
      if (fScaledImage) image = fScaledImage->fImage;

      if (imgX >= (int)image->width)  px = px - imgX + image->width - 1;
      if (imgY >= (int)image->height) py = py - imgY + image->height - 1;

      switch (event) {

         case kButton1Down:
            gVirtualX->SetLineColor(-1);

            stx = oldx = px;
            sty = oldy = py;
            break;

         case kButton1Motion:
            gVirtualX->DrawBox(oldx, oldy, stx, sty, TVirtualX::kHollow);
            oldx = px;
            oldy = py;
            gVirtualX->DrawBox(oldx, oldy, stx, sty, TVirtualX::kHollow);
            break;

         case kButton1Up:
            // do nothing if zoom area is too small
            if ( TMath::Abs(stx - px) < 5 || TMath::Abs(sty - py) < 5)
               return;

            Double_t xfact = (fScaledImage) ? (Double_t)fScaledImage->fImage->width  / fZoomWidth  : 1;
            Double_t yfact = (fScaledImage) ? (Double_t)fScaledImage->fImage->height / fZoomHeight : 1;

            Int_t imgX1 = stx - gPad->XtoAbsPixel(0);
            Int_t imgY1 = sty - gPad->YtoAbsPixel(1);
            Int_t imgX2 = px  - gPad->XtoAbsPixel(0);
            Int_t imgY2 = py  - gPad->YtoAbsPixel(1);

            imgY1 = image->height - 1 - imgY1;
            imgY2 = image->height - 1 - imgY2;
            imgX1 = (Int_t)(imgX1 / xfact) + fZoomOffX;
            imgY1 = (Int_t)(imgY1 / yfact) + fZoomOffY;
            imgX2 = (Int_t)(imgX2 / xfact) + fZoomOffX;
            imgY2 = (Int_t)(imgY2 / yfact) + fZoomOffY;

            Zoom((imgX1 < imgX2) ? imgX1 : imgX2, (imgY1 < imgY2) ? imgY1 : imgY2,
                 TMath::Abs(imgX1 - imgX2) + 1, TMath::Abs(imgY1 - imgY2) + 1);

            gVirtualX->SetLineColor(-1);
            gPad->Modified(kTRUE);
            gPad->Update();
            break;
      }
   }
}

//______________________________________________________________________________
char *TASImage::GetObjectInfo(Int_t px, Int_t py) const
{
   // Get image pixel coordinates and the pixel value at the mouse pointer.

   static char info[64];
   info[0] = 0;

   if (!IsValid()) return info;

   // convert to image pixel on screen
   px -= gPad->XtoAbsPixel(0);
   py -= gPad->YtoAbsPixel(1);

   // no info if mouse is outside of image
   if (px < 0 || py < 0)  return info;

   ASImage *image = fImage;
   if (fScaledImage) image = fScaledImage->fImage;
   if (px >= (int)image->width || py >= (int)image->height)
      return info;

   py = image->height - 1 - py;
   // convert to original image size and take zooming into account
   if (fScaledImage) {
      px = (Int_t)(px / (Double_t)fScaledImage->fImage->width  * fZoomWidth ) + fZoomOffX;
      py = (Int_t)(py / (Double_t)fScaledImage->fImage->height * fZoomHeight) + fZoomOffY;
   }

   if (fImage->alt.vector) {
      sprintf(info, "x: %d  y: %d   %.5g",
              px, py, fImage->alt.vector[px + py * fImage->width]);
   } else {
      sprintf(info, "x: %d  y: %d", px, py);
   }

   return info;
}

//______________________________________________________________________________
void TASImage::SetPalette(const TImagePalette *palette)
{
   // Set a new palette to an image. Only images that were created with the
   // SetImage() functions can be modified with this function.
   // The previously used palette is destroyed.

   TAttImage::SetPalette(palette);

   if (!InitVisual()) {
      Warning("SetPalette", "Visual not initiated");
      return;
   }

   if (!IsValid()) {
      Warning("SetPalette", "Image not valid");
      return;
   }

   if (fImage->alt.vector == 0)
      return;

   // copy ROOT palette to asImage palette
   const TImagePalette &pal = GetPalette();

   ASVectorPalette asPalette;
   asPalette.npoints = pal.fNumPoints;
   asPalette.channels[0] = new CARD16 [asPalette.npoints];
   asPalette.channels[1] = new CARD16 [asPalette.npoints];
   asPalette.channels[2] = new CARD16 [asPalette.npoints];
   asPalette.channels[3] = new CARD16 [asPalette.npoints];
   memcpy(asPalette.channels[0], pal.fColorBlue,  pal.fNumPoints * sizeof(UShort_t));
   memcpy(asPalette.channels[1], pal.fColorGreen, pal.fNumPoints * sizeof(UShort_t));
   memcpy(asPalette.channels[2], pal.fColorRed,   pal.fNumPoints * sizeof(UShort_t));
   memcpy(asPalette.channels[3], pal.fColorAlpha, pal.fNumPoints * sizeof(UShort_t));

   asPalette.points = new double[asPalette.npoints];
   for (Int_t point = 0; point < Int_t(asPalette.npoints); point++)
      asPalette.points[point] = fMinValue + (fMaxValue - fMinValue) * pal.fPoints[point];

   // use the new palette in this image
   colorize_asimage_vector(fgVisual, fImage, &asPalette, ASA_ASImage, GetImageQuality());

   delete [] asPalette.points;
   for (Int_t col = 0; col < 4; col++)
      delete [] asPalette.channels[col];


   delete fScaledImage;
   fScaledImage = 0;
}

//______________________________________________________________________________
void TASImage::Scale(UInt_t toWidth, UInt_t toHeight)
{
   // Scales the original image. The size of the image on the screen does not
   // change because it is defined by the size of the pad.
   // This function can be used to change the size of an image before writing
   // it into a file. The colors of the new pixels are interpolated.
   // An image created with the SetImage() functions cannot be modified with
   // the function SetPalette() any more after a call of this function!

   if (!IsValid()) {
      Warning("Scale", "Image not initiated");
      return;
   }

   if (!InitVisual()) {
      Warning("Scale", "Visual not initiated");
      return;
   }

   if (toWidth < 1)
       toWidth = 1;
   if (toHeight < 1 )
      toHeight = 1;
   if (toWidth > 30000)
      toWidth = 30000;
   if (toHeight > 30000)
      toHeight = 30000;

   ASImage *img = scale_asimage(fgVisual, fImage, toWidth, toHeight,
                                ASA_ASImage, GetImageCompression(),
                                GetImageQuality());
   if (fImage) destroy_asimage(&fImage);
   fImage = img;
   UnZoom();
   fZoomUpdate = kZoomOps;
}

//______________________________________________________________________________
void TASImage::Tile(UInt_t toWidth, UInt_t toHeight)
{
   // Tiles the original image. 

   if (!IsValid()) {
      Warning("Tile", "Image not initiated");
      return;
   }

   if (!InitVisual()) {
      Warning("Tile", "Visual not initiated");
      return;
   }

   if (toWidth < 1)
       toWidth = 1;
   if (toHeight < 1 )
      toHeight = 1;
   if (toWidth > 30000)
      toWidth = 30000;
   if (toHeight > 30000)
      toHeight = 30000;

   ASImage *img = tile_asimage(fgVisual, fImage, 0, 0, toWidth, toHeight, 0,
                                ASA_ASImage, GetImageCompression(), GetImageQuality());
   if (fImage) destroy_asimage(&fImage);
   fImage = img;
   UnZoom();
   fZoomUpdate = kZoomOps;
}

//______________________________________________________________________________
void TASImage::Zoom(UInt_t offX, UInt_t offY, UInt_t width, UInt_t height)
{
   // The area of an image displayed in a pad is defined by this function.
   // Note: the size on the screen is defined by the size of the pad.
   // The original image is not modified by this function.
   // If width or height is larger than the original image they are reduced to
   // the width and height of the image.
   // If the off values are too large (off + width > image width) than the off
   // values are decreased. For example: offX = image width - width
   // Note: the parameters are always relative to the original image not to the
   // size of an already zoomed image.

   if (!IsValid()) {
      Warning("Zoom", "Image not valid");
      return;
   }
   fZoomUpdate = kZoom;

   fZoomWidth  = (width == 0) ? 1 : ((width > fImage->width) ? fImage->width : width);
   fZoomHeight = (height == 0) ? 1 : ((height > fImage->height) ? fImage->height : height);
   fZoomOffX   = offX;
   if (fZoomOffX + fZoomWidth > fImage->width)
      fZoomOffX = fImage->width - fZoomWidth;
   fZoomOffY   = offY;
   if (fZoomOffY + fZoomHeight > fImage->height)
      fZoomOffY = fImage->height - fZoomHeight;
}

//______________________________________________________________________________
void TASImage::UnZoom()
{
   // Un-zooms the image to original size.
   //
   // UnZoom() - performs undo for Zoom,Crop,Scale actions  

   if (!IsValid()) {
      Warning("UnZoom", "Image not valid");
      return;
   }
   fZoomUpdate = kZoom;
   fZoomOffX   = 0;
   fZoomOffY   = 0;
   fZoomWidth  = fImage->width;
   fZoomHeight = fImage->height;

   delete fScaledImage;
   fScaledImage = 0;
}

//______________________________________________________________________________
void TASImage::Flip(Int_t flip)
{
   // Flip image in place. Flip is either 90, 180, 270, 180 is default.
   // This function manipulates the original image and destroys the
   // scaled and zoomed image which will be recreated at the next call of
   // the Draw function. If the image is zoomed the zoom - coordinates are
   // now relative to the new image.
   // This function cannot be used for images which were created with the
   // SetImage() functions, because the original pixel values would be
   // destroyed.

   if (!IsValid()) {
      Warning("Flip", "Image not valid");
      return;
   }
   if (!InitVisual()) {
      Warning("Flip", "Visual not initiated");
      return;
   }

   if (fImage->alt.vector) {
      Warning("Flip", "flip does not work for data images");
      return;
   }

   Int_t rflip = flip/90;

   UInt_t w = fImage->width;
   UInt_t h = fImage->height;
   if (rflip & 1) {
      w = fImage->height;
      h = fImage->width;
   }

   ASImage *img = flip_asimage(fgVisual, fImage, 0, 0, w, h, rflip,
                               ASA_ASImage, GetImageCompression(),
                               GetImageQuality());
   if (fImage) {
      destroy_asimage(&fImage);
   }

   fImage = img;
   UnZoom();
}

//______________________________________________________________________________
void TASImage::Mirror(Bool_t vert)
{
   // Mirror image in place. If vert is true mirror in vertical axis,
   // horizontal otherwise. Vertical is default.
   // This function manipulates the original image and destroys the
   // scaled and zoomed image which will be recreated at the next call of
   // the Draw function. If the image is zoomed the zoom - coordinates are
   // now relative to the new image.
   // This function cannot be used for images which were created with the
   // SetImage() functions, because the original pixel values would be
   // destroyed.

   if (!IsValid()) {
      Warning("Mirror", "Image not valid");
      return;
   }

   if (!InitVisual()) {
      Warning("Mirrow", "Visual not initiated");
      return;
   }

   if (fImage->alt.vector) {
      Warning("Mirror", "mirror does not work for data images");
      return;
   }

   ASImage *img = mirror_asimage(fgVisual, fImage, 0, 0,
                                 fImage->width, fImage->height, vert,
                                 ASA_ASImage, GetImageCompression(),
                                 GetImageQuality());
   if (fImage) {
      destroy_asimage(&fImage);
   }
   fImage = img;
   UnZoom();
}

//______________________________________________________________________________
UInt_t TASImage::GetWidth() const
{
   // Return width of original image not of the displayed image.
   // (Number of image pixels)

   return fImage ? fImage->width : 0;
}

//______________________________________________________________________________
UInt_t TASImage::GetHeight() const
{
   // Return height of original image not of the displayed image.
   // (Number of image pixels)

   return fImage ? fImage->height : 0;
}

//______________________________________________________________________________
UInt_t TASImage::GetScaledWidth() const
{
   // Return width of the displayed image not of the original image.
   // (Number of screen pixels)

   return fScaledImage ? fScaledImage->fImage->width : GetWidth();
}

//______________________________________________________________________________
UInt_t TASImage::GetScaledHeight() const
{
   // Return height of the displayed image not of the original image.
   // (Number of screen pixels)

   return fScaledImage ? fScaledImage->fImage->height : GetHeight();
}

//______________________________________________________________________________
Bool_t TASImage::InitVisual()
{
   // Static function to initialize the ASVisual.

   if (fgVisual) return kTRUE;
   if (gROOT->IsBatch()) return kFALSE;

   dpy = (Display*) gVirtualX->GetDisplay();

   Int_t screen  = gVirtualX->GetScreen();
   Int_t depth   = gVirtualX->GetDepth();
   Visual *vis   = (Visual*) gVirtualX->GetVisual();
   Colormap cmap = (Colormap) gVirtualX->GetColormap();
#ifndef WIN32
   fgVisual = create_asvisual_for_id(dpy, screen, depth,
                                     XVisualIDFromVisual(vis), cmap, 0);
#else
   fgVisual = create_asvisual( NULL, 0, 0, NULL );
#endif
   return kTRUE;
}

//______________________________________________________________________________
void TASImage::StartPaletteEditor()
{
   // Start palette editor.

   if (!IsValid()) {
      Warning("StartPaletteEditor", "Image not valid");
      return;
   }
   if (fImage->alt.vector == 0) {
      Warning("StartPaletteEditor", "palette can be modified only for data images");
      return;
   }

   // Opens a GUI to edit the color palette
   TAttImage::StartPaletteEditor();
}

//______________________________________________________________________________
Pixmap_t TASImage::GetPixmap()
{
   // returns image pixmap

#ifdef WIN32
   void *bmbits = NULL ;
   BITMAPINFO *bmi = NULL ;
#endif

   Pixmap_t pxmap;

   if (!InitVisual()) {
      Warning("GetPixmap", "Visual not initiated");
      return 0;
   }

   ASImage *img = fScaledImage ? fScaledImage->fImage : fImage;

#ifndef WIN32
   pxmap = (Pixmap_t)asimage2pixmap(fgVisual, gVirtualX->GetDefaultRootWindow(), 
                                    img, 0, kTRUE);
#else
   bmi = ASImage2DBI( fgVisual, img, 0, 0, img->width, img->height, &bmbits );
   if(gDIB2Pixmap != 0) {
       pxmap = gDIB2Pixmap((ULong_t)bmi, (ULong_t)bmbits);
       free(bmbits);
       free(bmi);
   }
#endif
   
   return pxmap;
}

//______________________________________________________________________________
Pixmap_t TASImage::GetMask()
{
   // returns image pixmap 

   Pixmap_t pxmap = 0;

   if (!InitVisual()) {
      Warning("GetMask", "Visual not initiated");
      return 0;
   }

   ASImage *img = fScaledImage ? fScaledImage->fImage : fImage;

#ifndef WIN32
   pxmap = (Pixmap_t)asimage2mask(fgVisual, gVirtualX->GetDefaultRootWindow(), 
                                  img, 0, kTRUE);
#endif
   
   return pxmap;
}

//______________________________________________________________________________
void TASImage::SetImage(Pixmap_t pxm, Pixmap_t mask)
{
   // create image from pixmap

   if (!InitVisual()) {
      Warning("SetImage", "Visual not initiated");
      return;
   }

   if (fImage) {
      destroy_asimage(&fImage);
   }

   delete fScaledImage;
   fScaledImage = 0;

   SetName("unknown");

   Int_t xy;
   UInt_t w, h;

   gVirtualX->GetWindowSize(pxm, xy, xy, w, h);
#ifndef WIN32
   fImage = pixmap2asimage(fgVisual, pxm, 0, 0, w, h, mask ? mask : AllPlanes, 0, 0);
#else
   unsigned char *bits = (gGetBmBits != 0) ? gGetBmBits(pxm, w, h) : 0;
   fImage = bitmap2asimage (bits, w, h, 0);
#endif
}

//______________________________________________________________________________
TArrayL *TASImage::GetPixels(Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // returns 2D array of machine dependent pixel values

   if (!fImage) {
      Warning("GetPixels", "Wrong Image");
      return 0;
   }

   ASImage *img =  fScaledImage ? fScaledImage->fImage : fImage;
   ASImageDecoder *imdec;

   width = !width  ? img->width : width;
   height = !height ? img->height : height;

   if (x < 0) {
      width -= x;
      x = 0 ;
   }
   if (y < 0) {
      height -= y;
      y = 0;
   }

   if ((x >= (int)img->width) || (y >= (int)img->height)) {
      return 0;
   }

   if ((int)(x + width) > (int)img->width) {
      width = img->width - x;
   }

   if ((int)(y + height) > (int)img->height) {
      height = img->height - y;
   }

   if ((imdec = start_image_decoding(NULL, fImage, SCL_DO_ALL, 0, y, 
                                     img->width, height, NULL)) == NULL) {
      Warning("GetPixels", "Failed to create image decoder");
      return 0;
   }
 
   TArrayL *ret = new TArrayL(width * height);
   Int_t r = 0;
   Int_t g = 0;
   Int_t b = 0;
   Long_t p = 0;

   for (UInt_t k = 0; k < height; k++) {
      imdec->decode_image_scanline(imdec);

      for (UInt_t i = 0; i < width; ++i)  {
         if ((r == (Int_t)imdec->buffer.red[i]) &&
             (g == (Int_t)imdec->buffer.green[i]) &&
             (b == (Int_t)imdec->buffer.blue[i])) {
         } else {
            r = (Int_t)imdec->buffer.red[i];
            g = (Int_t)imdec->buffer.green[i];
            b = (Int_t)imdec->buffer.blue[i];
            p = (Long_t)TColor::RGB2Pixel(r, g, b);
         }
         ret->AddAt(p, k*width + i);
      }
   }

   stop_image_decoding(&imdec);
   return ret;
}

//______________________________________________________________________________
TArrayD *TASImage::GetArray(UInt_t w, UInt_t h, TImagePalette *palette)
{
   // Converts an image into 2D array of doubles according to palette.
   // If palette is ZERO a color converted to double value [0, 1] according to formula
   //   Double_t((r << 16) + (g << 8) + b)/0xFFFFFF

   if (!fImage) {
      Warning("GetArray", "Wrong Image");
      return 0;
   }

   ASImageDecoder *imdec;

   w = w ? w : fImage->width;
   h = h ? h : fImage->height;

   if ((fImage->width != w) || (fImage->height != h)) {
      Scale(w, h);
   }

   ASImage *img = fScaledImage ? fScaledImage->fImage : fImage;

   if ((imdec = start_image_decoding(NULL, img, SCL_DO_ALL, 0, 0, 
                                     img->width, 0, NULL)) == NULL) {
      Warning("GetArray", "Failed to create image decoder");
      return 0;
   }

   TArrayD *ret = new TArrayD(w * h);
   CARD32 r = 0;
   CARD32 g = 0;
   CARD32 b = 0;
   Int_t p = 0;
   Double_t v = 0;

   for (UInt_t k = 0; k < h; k++) {
      imdec->decode_image_scanline(imdec);

      for (UInt_t i = 0; i < w; ++i)  {
         if ((r == imdec->buffer.red[i]) &&
             (g == imdec->buffer.green[i]) &&
             (b == imdec->buffer.blue[i])) {
         } else {
            r = imdec->buffer.red[i];
            g = imdec->buffer.green[i];
            b = imdec->buffer.blue[i];
            if (palette) p = palette->FindColor(r, g, b);
         }
         v = palette ? palette->fPoints[p] : Double_t((r << 16) + (g << 8) + b)/0xFFFFFF;
         ret->AddAt(v, (h-k-1)*w + i);
      }
   }

   stop_image_decoding(&imdec);
   return ret;
}

//______________________________________________________________________________
void TASImage::DrawText(Int_t x, Int_t y, const char *text, Int_t size,
                        const char *color, const char *font_name,
                        EText3DType type, const char *fore_file)
{
   // Draw text of size (in pixels for TrueType fonts) 
   // at position (x, y) with color  specified by hex string.
   //   font_name - TrueType font's filename or X font spec or alias.
   //   3D style of text is one of the following:
   //     0 - plain 2D text, 1 - embossed, 2 - sunken, 3 - shade above,
   //     4 - shade below, 5 - embossed thick, 6 - sunken thick.
   //     7 - ouline above, 8 - ouline below, 9 - full ouline.
   //  fore_file specifies foreground texture of text.

   UInt_t width, height ;
   ARGB32 text_color = ARGB32_Black;
   ASImage *fore_im = 0;
   ASImage *text_im = 0;

   if (!InitVisual()) {
      Warning("DrawText", "Visual not initiated");
      return;
   }

   if (!gFontManager) gFontManager = create_font_manager(dpy, 0, 0);
   if (!gFontManager) {
      Warning("DrawText", "cannot create Font Manager");
      return;
   }

   TString fn = font_name;
   fn.Strip();

   if (fn.EndsWith(".ttf") || fn.EndsWith(".TTF")) {
      fn = gSystem->ExpandPathName(fn.Data());
   }

   ASFont *font = get_asfont(gFontManager, fn.Data(), 0, size, ASF_GuessWho);

   if (!font) {
      font = get_asfont(gFontManager, "fixed", 0, size, ASF_GuessWho);
      if (!font) {
         Warning("DrawText", "cannot find a font %s", font_name);
         return;
      }
   }

   if (color) {
      parse_argb_color(color, &text_color);
   }

   get_text_size(text, font, (ASText3DType)type, &width, &height);

   if (fore_file) {
      ASImage *tmp = file2ASImage(fore_file, 0xFFFFFFFF, SCREEN_GAMMA, 0, NULL);
      if (tmp) {
         if ((tmp->width != width) || (tmp->height != height)) { 
            fore_im = tile_asimage(fgVisual, tmp, 0, 0, width, height, 0,
                                   ASA_ASImage, GetImageCompression(), GetImageQuality());
         }
         destroy_asimage(&tmp);
      } else {
         fore_im = tmp;
      }
   }

   text_im = draw_text(text, font, (ASText3DType)type, 0);

   ASImage *rimg = fImage;

   if (fore_im) {
      move_asimage_channel(fore_im, IC_ALPHA, text_im, IC_ALPHA);
      destroy_asimage(&text_im);
   } else {
      fore_im = text_im ;
   } 
 
   release_font(font);

   if (fore_im) {
      ASImage *rendered_im;
      ASImageLayer layers[2];

      init_image_layers(&(layers[0]), 2);
      fore_im->back_color = text_color;
      layers[0].im = rimg;
      layers[0].dst_x = 0;
      layers[0].dst_y = 0;
      layers[0].clip_width = rimg->width;
      layers[0].clip_height = rimg->height;
      layers[0].bevel = 0;
      layers[1].im = fore_im;
      layers[1].dst_x = x;
      layers[1].dst_y = y;
      layers[1].clip_width = fore_im->width;
      layers[1].clip_height = fore_im->height;
     
      rendered_im = merge_layers(fgVisual, &(layers[0]), 2, rimg->width, rimg->height,
                                 ASA_ASImage, GetImageCompression(), GetImageQuality());

      destroy_asimage(&fore_im);
      if (fImage) destroy_asimage(&fImage);
      fImage = rendered_im;
      UnZoom();
   }
}

//______________________________________________________________________________
void TASImage::Merge(const TImage *im, const char *op, Int_t x, Int_t y)
{
   // Merge two images.
   //
   // op is string which specifies overlay operation. Supported operations are:
   //    add            - color addition with saturation
   //    alphablend     - alpha-blending
   //    allanon        - color values averaging
   //    colorize       - hue and saturate bottom image same as top image
   //    darken         - use lowest color value from both images
   //    diff           - use absolute value of the color difference between two images
   //    dissipate      - randomly alpha-blend images
   //    hue            - hue bottom image same as top image
   //    lighten        - use highest color value from both images
   //    overlay        - some wierd image overlaying(see GIMP)
   //    saturate       - saturate bottom image same as top image
   //    screen         - another wierd image overlaying(see GIMP)
   //    sub            - color substraction with saturation
   //    tint           - tinting image with image
   //    value          - value bottom image same as top image

   if (!im) return;

   if (!InitVisual()) {
      Warning("Merge", "Visual not initiated");
      return;
   }

   ASImage *rendered_im;
   ASImageLayer layers[2];

   init_image_layers(&(layers[0]), 2);
   layers[0].im = fImage;
   layers[0].dst_x = 0;
   layers[0].dst_y = 0;
   layers[0].clip_width = fImage->width;
   layers[0].clip_height = fImage->height;
   layers[0].bevel = 0;
   layers[1].im = ((TASImage*)im)->fImage;
   layers[1].dst_x = x;
   layers[1].dst_y = y;
   layers[1].clip_width = im->GetWidth();
   layers[1].clip_height = im->GetHeight();
   layers[1].merge_scanlines = blend_scanlines_name2func(op ? op : "add");

   rendered_im = merge_layers(fgVisual, &(layers[0]), 2, fImage->width, fImage->height,
                              ASA_ASImage, GetImageCompression(), GetImageQuality());

   if (fImage) destroy_asimage(&fImage);
   fImage = rendered_im;
   UnZoom();
}

//______________________________________________________________________________
void TASImage::Blur(Double_t hr, Double_t vr)
{
   // Performs Gaussian blurr of the image (usefull for drop shadows)
   //    hr         - horizontal radius of the blurr
   //    vr         - vertical radius of the blurr

   if (!InitVisual()) {
      Warning("Blur", "Visual not initiated");
      return;
   }

   if (!fImage) {
      fImage = create_asimage(100, 100, 0);

      if (!fImage) {
         Warning("Blur", "Failed to create image");
         return;
      }

      fill_asimage(fgVisual, fImage, 0, 0, fImage->width, fImage->height, ARGB32_White);
   }

   ASImage *rendered_im = blur_asimage_gauss(fgVisual, fImage, hr > 0 ? hr : 3,
                                             vr > 0 ? vr : 3, SCL_DO_ALL,
                                             ASA_ASImage, GetImageCompression(), GetImageQuality());
   if (fImage) {
      destroy_asimage(&fImage);
   }
   fImage = rendered_im;
   UnZoom();
}

//______________________________________________________________________________
TObject *TASImage::Clone(const char *newname) const
{
   // clone image

   if (!InitVisual() || !fImage) {
      Warning("Clone", "Image not initiated");
      return 0;
   }

   TASImage *im = (TASImage*)TImage::Create();

   if (!im) {
      Warning("Clone", "Failed to create image");
      return 0;
   }

   im->SetName(newname);

   im->fImage = clone_asimage(fImage, SCL_DO_ALL);
   im->fMaxValue = fMaxValue;
   im->fMinValue = fMinValue;
   im->fZoomOffX = fZoomOffX;
   im->fZoomOffY = fZoomOffY;
   im->fZoomWidth = fZoomWidth;
   im->fZoomHeight = fZoomHeight;
   im->fZoomUpdate = fZoomUpdate;
   im->fScaledImage = fScaledImage ? (TASImage*)fScaledImage->Clone("") : 0;

   if (fImage->alt.argb32) {
      UInt_t sz = fImage->width * fImage->height;
      im->fImage->alt.argb32 = new ARGB32[sz];
      memcpy(im->fImage->alt.argb32, fImage->alt.argb32, sz * sizeof(ARGB32));
   }

   return im;
}

//______________________________________________________________________________
void TASImage::Vectorize(UInt_t max_colors, UInt_t dither, Int_t opaque_threshold)
{
   // Reduces colordepth of an image and fills vector of "scientific data" [0...1]
   //
   // Colors are reduced by allocating colorcells to most used colors first,
   // and then approximating other colors with those allocated. 
   // max_colors       - maximum size of the colormap.
   // dither           - number of bits to strip off the color data ( 0...7 )
   // opaque_threshold - alpha channel threshold at which pixel should be
   //                    treated as opaque

   if (!InitVisual()) {
      Warning("Vectorize", "Visual not initiated");
      return;
   }

   if (!fImage) {
      fImage = create_asimage(100, 100, 0);

      if (!fImage) {
         Warning("Vectorize", "Failed to create image");
         return;
      }

      fill_asimage(fgVisual, fImage, 0, 0, fImage->width, fImage->height, ARGB32_White);
   }

   ASColormap cmap;
   int *res;
   UInt_t r, g, b;

   dither = dither > 7 ? 7 : dither;

   res = colormap_asimage(fImage, &cmap, max_colors, dither, opaque_threshold);

   Double_t *vec = new Double_t[fImage->height*fImage->width];
   UInt_t v;
   Double_t tmp;
   fMinValue = 2;
   fMaxValue = -1;

   for (UInt_t y = 0; y < fImage->height; y++) {
      for (UInt_t x = 0; x < fImage->width; x++) {
         int i = y*fImage->width + x;
         g = INDEX_SHIFT_GREEN(cmap.entries[res[i]].green);
         b = INDEX_SHIFT_BLUE(cmap.entries[res[i]].blue);
         r = INDEX_SHIFT_RED(cmap.entries[res[i]].red);
         v = MAKE_INDEXED_COLOR24(r,g,b);
         v = (v>>12)&0x0FFF;
         tmp = Double_t(v)/0x0FFF;
         vec[(fImage->height - y - 1)*fImage->width + x] = tmp;
         if (fMinValue > tmp) fMinValue = tmp;
         if (fMaxValue < tmp) fMaxValue = tmp;
      }
   }
   TImagePalette *pal = new TImagePalette(cmap.count);

   for (UInt_t j = 0; j < cmap.count; j++) {
      g = INDEX_SHIFT_GREEN(cmap.entries[j].green);
      b = INDEX_SHIFT_BLUE(cmap.entries[j].blue);
      r = INDEX_SHIFT_RED(cmap.entries[j].red);
      v = MAKE_INDEXED_COLOR24(r,g,b);

      v = (v>>12) & 0x0FFF;
      pal->fPoints[j] = Double_t(v)/0x0FFF;

      pal->fColorRed[j] = cmap.entries[j].red << 8;
      pal->fColorGreen[j] = cmap.entries[j].green << 8;
      pal->fColorBlue[j] = cmap.entries[j].blue << 8;
      pal->fColorAlpha[j] = 0xFF00;
   }

   destroy_colormap(&cmap, kTRUE);

   fPalette = *pal;
   fImage->alt.vector = vec;
   UnZoom();
}

//______________________________________________________________________________
void TASImage::HSV(UInt_t hue, UInt_t radius, Int_t H, Int_t S, Int_t V, 
                   Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // This function will tile original image to specified size with offsets
   // requested, and then it will go though it and adjust hue, saturation and
   // value of those pixels that have specific hue, set by affected_hue/
   // affected_radius parameters. When affected_radius is greater then 180
   // entire image will be adjusted. Note that since grayscale colors have
   // no hue - the will not get adjusted. Only saturation and value will be
   // adjusted in gray pixels.
   // Hue is measured as an angle on a 360 degree circle, The following is
   // relationship of hue values to regular color names :
   // red      - 0
   // yellow   - 60
   // green    - 120
   // cyan     - 180
   // blue     - 240
   // magenta  - 300
   // red      - 360
   //
   // All the hue values in parameters will be adjusted to fall withing 0-360 range.

   // hue - hue in degrees in range 0-360. This allows to limit
   //       impact of color adjustment to affect only limited range of hues.
   //
   // radius - value in degrees to be used in order to
   //          calculate the range of affected hues. Range is determined by
   //          substracting and adding this value from/to affected_hue.
   //
   // H -   value by which to change hues in affected range.
   // S -   value by which to change saturation of the pixels in affected hue range.
   // V -   value by which to change Value(brightness) of pixels in affected hue range.
   //
   // x,y - position on infinite surface tiled with original image, of the 
   //       left-top corner of the area to be used for new image.
   //
   // width, height - size of the area of the original image to be used for new image.
   //                 Default is current width, height of the image.

   if (!InitVisual()) {
      Warning("HSV", "Visual not initiated");
      return;
   }

   if (!fImage) {
      fImage = create_asimage(width ? width : 20, height ? height : 20, 0);

      if (!fImage) {
         Warning("HSV", "Failed to create image");
         return;
      }

      x = 0;
      y = 0;
      fill_asimage(fgVisual, fImage, 0, 0, fImage->width, fImage->height, ARGB32_White);
   }

   width = !width ? fImage->width : width;
   height = !height ? fImage->height : height;

   ASImage *rendered_im = 0;

   if (H || S || V) {
      rendered_im = adjust_asimage_hsv(fgVisual, fImage, x, y, width, height,
                                       hue, radius, H, S, V, ASA_ASImage, 100,
                                       ASIMAGE_QUALITY_TOP);
   }
   if (!rendered_im) {
      Warning("HSV", "Failed to create rendered image");
      return;
   }

   if (fImage) {
      destroy_asimage(&fImage);
   }

   fImage = rendered_im;
   UnZoom();
}

//______________________________________________________________________________
void TASImage::Gradient(UInt_t angle, const char *colors, const char *offsets,
                        Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // Render multipoint gradient inside rectangle of size (width, height) 
   // at position (x,y) within the existing image.
   //
   // angle    Given in degrees.  Default is 0.  This is the
   //          direction of the gradient.  Currently the only supported
   //          values are 0, 45, 90, 135, 180, 225, 270, 315.  0 means left
   //          to right, 90 means top to bottom, etc.
   //
   // colors   Whitespace-separated list of colors.  At least two
   //          colors are required.  Each color in this list will be visited
   //          in turn, at the intervals given by the offsets attribute.
   //
   // offsets  Whitespace-separated list of floating point values
   //          ranging from 0.0 to 1.0.  The colors from the colors attribute
   //          are given these offsets, and the final gradient is rendered
   //          from the combination of the two.  If both colors and offsets
   //          are given but the number of colors and offsets do not match,
   //          the minimum of the two will be used, and the other will be
   //          truncated to match.  If offsets are not given, a smooth
   //          stepping from 0.0 to 1.0 will be used.

   if (!InitVisual()) {
      Warning("Gradient", "Visual not initiated");
      return;
   }

   ASImage *rendered_im = 0;
   ASGradient gradient;

   int reverse = 0, npoints1 = 0, npoints2 = 0;
   char *p;
   char *pb, ch;

   if ((angle > 2 * 180 * 15 / 16) || (angle < 2 * 180 * 1 / 16)) {
      gradient.type = GRADIENT_Left2Right;
   } else if (angle < 2 * 180 * 3 / 16) {
      gradient.type = GRADIENT_TopLeft2BottomRight;
   } else if (angle < 2 * 180 * 5 / 16) {
      gradient.type = GRADIENT_Top2Bottom;
   } else if (angle < 2 * 180 * 7 / 16) {
      gradient.type = GRADIENT_BottomLeft2TopRight; reverse = 1;
   } else if (angle < 2 * 180 * 9 / 16) {
      gradient.type = GRADIENT_Left2Right; reverse = 1;
   } else if (angle < 2 * 180 * 11 / 16) {
      gradient.type = GRADIENT_TopLeft2BottomRight; reverse = 1;
   } else if (angle < 2 * 180 * 13 / 16) {
      gradient.type = GRADIENT_Top2Bottom; reverse = 1;
   } else {
      gradient.type = GRADIENT_BottomLeft2TopRight;
   }

   for (p = (char*)colors; isspace((int)*p); p++);

   for (npoints1 = 0; *p; npoints1++) {
      if (*p) {
         for ( ; *p && !isspace((int)*p); p++);
      }
      for ( ; isspace((int)*p); p++);
   }
   if (offsets) {
      for (p = (char*)offsets; isspace((int)*p); p++);

      for (npoints2 = 0; *p; npoints2++) {
         if (*p) {
            for ( ; *p && !isspace((int)*p); p++);
         }
         for ( ; isspace((int)*p); p++);
      }
   }
   if (npoints1 > 1) {
      int i;
      if (offsets && (npoints1 > npoints2)) npoints1 = npoints2;

      if (!width) {
         width = fImage ? fImage->width : 20;
      }
      if (!height) {
         height = fImage ? fImage->height : 20;
      }

      gradient.color = new ARGB32[npoints1];
      gradient.offset = new double[npoints1];

      for (p = (char*)colors; isspace((int)*p); p++);

      for (npoints1 = 0; *p; ) {
         pb = p;

         if (*p) {
            for ( ; *p && !isspace((int)*p); p++);
         }
         for ( ; isspace((int)*p); p++);

        ch = *p; *p = '\0';
         if (parse_argb_color(pb, gradient.color + npoints1) != pb) {
         npoints1++;
         } else {
            Warning("Gradient", "Failed to parse color [%s] - defaulting to black", pb);
         }
         *p = ch;
      }

      if (offsets) {
         for (p = (char*)offsets; isspace((int)*p); p++);

         for (npoints2 = 0; *p; ) {
            pb = p;

            if (*p) {
               for ( ; *p && !isspace((int)*p); p++);
            }
            ch = *p; *p = '\0';
            gradient.offset[npoints2] = strtod(pb, &pb);

            if (pb == p) npoints2++;
            *p = ch;
            for ( ; isspace((int)*p); p++);
         }
      } else {
         for (npoints2 = 0; npoints2 < npoints1; npoints2++) {
            gradient.offset[npoints2] = (double)npoints2 / (npoints1 - 1);
         }
      }
      gradient.npoints = npoints1;

      if (npoints2 && (gradient.npoints > npoints2)) {
         gradient.npoints = npoints2;
      }
      if (reverse) {
         for (i = 0; i < gradient.npoints/2; i++) {
            int i2 = gradient.npoints - 1 - i;
            ARGB32 c = gradient.color[i];
            double o = gradient.offset[i];
            gradient.color[i] = gradient.color[i2];
            gradient.color[i2] = c;
            gradient.offset[i] = gradient.offset[i2];
            gradient.offset[i2] = o;
         }
         for (i = 0; i < gradient.npoints; i++) {
            gradient.offset[i] = 1.0 - gradient.offset[i];
         }
      }
      rendered_im = make_gradient(fgVisual, &gradient, width, height, SCL_DO_ALL, 
                                  ASA_ASImage, GetImageCompression(), GetImageQuality());

      delete [] gradient.color;
      delete [] gradient.offset;
   }

   if (!rendered_im) {  // error
      Warning("Gradient", "Failed to create gradient image");
      return;
   }

   if (!fImage) {
      fImage = rendered_im;
      return;
   }

   ASImageLayer layers[2];

   init_image_layers(&(layers[0]), 2);
   layers[0].im = fImage;
   layers[0].dst_x = 0;
   layers[0].dst_y = 0;
   layers[0].clip_width = fImage->width;
   layers[0].clip_height = fImage->height;
   layers[0].bevel = 0;
   layers[1].im = rendered_im;
   layers[1].dst_x = x;
   layers[1].dst_y = y;
   layers[1].clip_width = width;
   layers[1].clip_height = height;
   layers[1].merge_scanlines = alphablend_scanlines;

   ASImage *merge_im = merge_layers(fgVisual, &(layers[0]), 2, fImage->width, fImage->height,
                                    ASA_ASImage, GetImageCompression(), GetImageQuality());
   if (!merge_im) {
      Warning("Gradient", "Failed to create merged image");
      return;
   }

   destroy_asimage(&rendered_im);
   if (fImage) destroy_asimage(&fImage);
   fImage = merge_im;
   UnZoom();
}

//______________________________________________________________________________
void TASImage::GetTextSize(UInt_t *width, UInt_t *height, const char *text,
                           Int_t size, const char *font_name, EText3DType type)
{
   // returns width and height of the ttext

   if (!gFontManager) gFontManager = create_font_manager(dpy, 0, 0);

   if (!gFontManager) {
      ::Warning("GetTextSize", "cannot create Font Manager");
      return;
   }

   TString fn = font_name;
   fn.Strip();

   *width = 0;
   *height = 0;

   if (fn.EndsWith(".ttf") || fn.EndsWith(".TTF")) {
      fn = gSystem->ExpandPathName(fn.Data());
   }

   ASFont *font = get_asfont(gFontManager, fn.Data(), 0, size, ASF_GuessWho);

   if (!font) {
      ::Warning("GetTextSize", "cannot find a font %s", font_name);
      return;
   }

   get_text_size(text, font, (ASText3DType)type, width, height);
}

/////////////// auxilary funcs used in TASImage::Bevel method //////////////////
//______________________________________________________________________________
static CARD8 MakeComponentHilite(int cmp)
{
   //

   if (cmp < 51) {
      cmp = 51;
   }
   cmp = (cmp * 12) / 10;

   return (cmp > 255) ? 255 : cmp;
}

//______________________________________________________________________________
static ARGB32 GetHilite(ARGB32 background)
{
   // calculates highlite color

   return ((MakeComponentHilite((background>>24) & 0x000000FF) << 24) & 0xFF000000) |
           ((MakeComponentHilite((background & 0x00FF0000) >> 16) << 16) & 0x00FF0000) |
           ((MakeComponentHilite((background & 0x0000FF00) >> 8) << 8) & 0x0000FF00) |
           ((MakeComponentHilite((background & 0x000000FF))) & 0x000000FF);
}

//______________________________________________________________________________
static ARGB32 GetShadow(ARGB32 background)
{
   // calculates shadow color

   return (background >> 1) & 0x7F7F7F7F;
}

//______________________________________________________________________________
static ARGB32 GetAverage(ARGB32 foreground, ARGB32 background)
{
   //

   CARD16   a, r, g, b;

   a = ARGB32_ALPHA8(foreground) + ARGB32_ALPHA8(background);
   a = (a<<3)/10;
   r = ARGB32_RED8(foreground) + ARGB32_RED8(background);
   r = (r<<3)/10;
   g = ARGB32_GREEN8(foreground) + ARGB32_GREEN8(background);
   g = (g<<3)/10;
   b = ARGB32_BLUE8(foreground) + ARGB32_BLUE8(background);
   b = (b<<3)/10;

   return MAKE_ARGB32(a, r, g, b);
}

//______________________________________________________________________________
void TASImage::Bevel(Int_t x, Int_t y, UInt_t width, UInt_t height,
                     const char *hi_color, const char *lo_color, UShort_t thick,
                     Bool_t reverse)
{
   //  Bevel is used to create 3D effect while drawing buttons, or any other
   // image that needs to be framed. Bevel is drawn using 2 primary colors:
   // one for top and left sides - hi color, and another for bottom and
   // right sides - low color. Bevel can be drawn over exisiting image or
   // as newly created,  as it is shown in code below:
   //
   //  TImage *img = TImage::Create();
   //  img->Bevel(0, 0, 400, 300, "#dddddd", "#000000", 3);
   //

   if (!InitVisual()) {
      Warning("Bevel", "Visual not initiated");
      return;
   }

   ASImageBevel bevel;

   ARGB32 hi, lo;
   parse_argb_color(hi_color, &hi);
   parse_argb_color(lo_color, &lo);

   if (reverse) {
      bevel.lo_color = hi;
      bevel.lolo_color = GetHilite(hi);
      bevel.hi_color = lo;
      bevel.hihi_color = GetShadow(lo);
   } else {
      bevel.hi_color = hi;
      bevel.hihi_color = GetHilite(hi);
      bevel.lo_color = lo;
      bevel.lolo_color = GetShadow(lo);
   }
   bevel.hilo_color = GetAverage(hi, lo); 

   int extra_hilite = 2;
   bevel.left_outline = bevel.top_outline = bevel.right_outline = bevel.bottom_outline = thick;
   bevel.left_inline = bevel.top_inline = bevel.right_inline = bevel.bottom_inline = extra_hilite + 1;

   if (bevel.top_outline > 1) {	
      bevel.top_inline += bevel.top_outline - 1;
   }

   if (bevel.left_outline > 1) {   
      bevel.left_inline += bevel.left_outline - 1;
   }
   
   if (bevel.right_outline > 1) {	
      bevel.right_inline += bevel.right_outline - 1;
   }

   if (bevel.bottom_outline > 1) {	
      bevel.bottom_inline += bevel.bottom_outline - 1;
   }

   ASImage *merge_im;
   ARGB32 fill = ((hi>>24) != 0xff) || ((lo>>24) != 0xff) ? bevel.hilo_color : (bevel.hilo_color | 0xff000000);

   if (!fImage) {
      fImage = create_asimage(width ? width : 20, height ? height : 20, 0);

      if (!fImage) {
         Warning("Bevel", "Failed to create image");
         return;
      }

      x = 0;
      y = 0;
      fill_asimage(fgVisual, fImage, 0, 0, fImage->width, fImage->height, fill);
   }

   width = !width ? fImage->width : width;
   height = !height ? fImage->height : height;

   ASImageLayer layers[2];
   init_image_layers(&(layers[0]), 2);

   layers[0].im = fImage;
   layers[0].dst_x = 0;
   layers[0].dst_y = 0;
   layers[0].clip_width = fImage->width;
   layers[0].clip_height = fImage->height;
   layers[0].bevel = 0;

   UInt_t w = width - (bevel.left_outline + bevel.right_outline);
   UInt_t h = height - (bevel.top_outline + bevel.bottom_outline);
   ASImage *bevel_im = create_asimage(w, h, 0);

   if (!bevel_im) {
      Warning("Bevel", "Failed to create bevel image");
      return;
   }

   layers[1].im = bevel_im;
   fill_asimage(fgVisual, bevel_im, 0, 0, w, h, fill);

   layers[1].dst_x = x;
   layers[1].dst_y = y;
   layers[1].clip_width = width;
   layers[1].clip_height = height;
   layers[1].bevel = &bevel;
   layers[1].merge_scanlines = alphablend_scanlines;

   merge_im = merge_layers(fgVisual, &(layers[0]), 2, fImage->width, fImage->height,
                           ASA_ASImage, GetImageCompression(), GetImageQuality());
   destroy_asimage(&bevel_im);
   
   if (!merge_im) {
      Warning("Bevel", "Failed to image");
      return;
   }

   if (fImage) destroy_asimage(&fImage);
   fImage = merge_im;
}

//______________________________________________________________________________
void TASImage::Pad(const char *col, UInt_t l, UInt_t r, UInt_t t, UInt_t b)
{
   // Enlarges image, padding it with specified color on each side in
   // accordance with requested geometry.

   Int_t x, y;
   UInt_t w, h;

   if (!InitVisual()) {
      Warning("Pad", "Visual not initiated");
      return;
   }

   if (!fImage) {
      fImage = create_asimage(100, 100, 0);

      if (!fImage) {
         Warning("Pad", "Failed to create image");
         return;
      }

      x = 0;
      y = 0;
      fill_asimage(fgVisual, fImage, 0, 0, fImage->width, fImage->height, ARGB32_White);
   }

   ARGB32 color;
   parse_argb_color(col, &color);

   x = l;
   y = t;
   w = l + fImage->width + r;
   h = t + fImage->height + b;

   ASImage *img = pad_asimage(fgVisual, fImage, x, y, w, h, color, 
                              ASA_ASImage, GetImageCompression(), GetImageQuality());

   if (!img) {
      Warning("Pad", "Failed to create output image");
      return;
   }

   if (fImage) destroy_asimage(&fImage);
   fImage = img;
   UnZoom();
   fZoomUpdate = kZoomOps;
}

//______________________________________________________________________________
void TASImage::Crop(Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // Crops an image

   if (!InitVisual()) {
      Warning("Crop", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("Crop", "No image");
      return;
   }
   
   x = x < 0 ? 0 : x;
   y = y < 0 ? 0 : y;

   width = x + width > fImage->width ? fImage->width - x : width;
   height = y + height > fImage->height ? fImage->height - y : height;

   if ((width == fImage->width) && (height == fImage->height)) {
      Warning("Crop", "input size larger than image");
      return;
   }
   ASImageDecoder *imdec = start_image_decoding(fgVisual, fImage, SCL_DO_ALL, 
                                                x, y, width, height, NULL);

   if (!imdec) {
      Warning("Crop", "Failed to start image decoding");
      return;
   }

   ASImage *img = create_asimage(width, height, 0);

   if (!img) {
      Warning("Crop", "Failed to create image");
      return;
   }

   ASImageOutput *imout = start_image_output(fgVisual, img, ASA_ASImage,
                                             GetImageCompression(), GetImageQuality());

   if (!imout) {
      Warning("Crop", "Failed to start image output");
      destroy_asimage(&img);
      return;
   }

#ifdef HAVE_MMX
   mmx_init();
#endif

   for (UInt_t i = 0; i < height; i++) {
      imdec->decode_image_scanline(imdec);
      imout->output_image_scanline(imout, &(imdec->buffer), 1);
   }

   stop_image_decoding(&imdec);
   stop_image_output(&imout);

#ifdef HAVE_MMX
   mmx_off();
#endif
   stop_image_decoding(&imdec);

   if (fImage) destroy_asimage(&fImage);
   fImage = img;
   UnZoom();
   fZoomUpdate = kZoomOps;
}

//______________________________________________________________________________
void TASImage::Append(const TImage *im, const char *option, const char *color )
{
   // Appends image
   //
   // option:
   //       "+" - appends to the right side
   //       "/" - appends to the bottom 

   if (!im) return;

   if (!InitVisual()) {
      Warning("Append", "Visual not initiated");
      return;
   }

   if (!fImage) {
      fImage = ((TASImage*)im)->fImage;
      return;
   }

   TString opt = option;
   opt.Strip();

   UInt_t width = fImage->width;
   UInt_t height = fImage->height;

   if (opt == "+") {
      Pad(color, 0, im->GetWidth(), 0, 0);
      Merge(im, "alphablend", width, 0);
   } else if (opt == "/") {
      Pad(color, 0, 0, 0, im->GetHeight());
      Merge(im, "alphablend", 0, height);
   } else {
      return;
   }

   UnZoom();
}

//______________________________________________________________________________
void TASImage::BeginPaint(Bool_t mode)
{
   // BeginPaint initializes internal array[width x height] of ARGB32 pixel values
   // That provides quick access to image during paint operations.
   // To RLE compress image one needs to call EndPaint method when paintinig is over 

   if (!InitVisual()) {
      Warning("BeginPaint", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("BeginPaint", "no image");
      return;
   }

   fPaintMode = mode;

   if (!fPaintMode || fImage->alt.argb32) {
      return;
   }

   ASImage *img = tile_asimage(fgVisual, fImage, 0, 0, fImage->width, fImage->height,
                               0, ASA_ARGB32, 0, ASIMAGE_QUALITY_DEFAULT);

   if (!img) {
      Warning("BeginPaint", "Failed to create image");
      return;
   }

   destroy_asimage(&fImage);
   fImage = img;
}

//______________________________________________________________________________
void TASImage::EndPaint()
{
   // EndPaint does internal RLE compression of image data

   if (!fImage) {
      Warning("EndPaint", "no image");
      return;
   }

   if (!fImage->alt.argb32) return;

   ASImage *img = tile_asimage(fgVisual, fImage, 0, 0, fImage->width, fImage->height,
                               0, ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT);

   if (!img) {
      Warning("EndPaint", "Failed to create image");
      return;
   }

   fPaintMode = kFALSE;
   destroy_asimage(&fImage);
   fImage = img;
}

//______________________________________________________________________________
UInt_t *TASImage::GetArgbArray()
{
   // Returns a pointer to internal array[width x height] of ARGB32 pixel values
   // This array is directly acessible. That allows to manipulate/change the image

   if (!fImage) {
      Warning("GetArgbArray", "no image");
      return 0;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   ASImage *img = fScaledImage ? fScaledImage->fImage : fImage;

   return (UInt_t *)img->alt.argb32;
}

//______________________________________________________________________________
UInt_t *TASImage::GetScanline(UInt_t y)
{
   // return a pointer to scanline

   if (!fImage) {
      Warning("GetScanline", "no image");
      return 0;
   } 

   ASImage *img = fScaledImage ? fScaledImage->fImage : fImage;
   CARD32 *ret = new CARD32[img->width];

   ASImageDecoder *imdec = start_image_decoding(fgVisual, img, SCL_DO_ALL, 
                                                0, y, img->width, 1, NULL);

   if (!imdec) {
      Warning("GetScanline", "Failed to start image decoding");
      return 0;
   }

#ifdef HAVE_MMX
   mmx_init();
#endif

   imdec->decode_image_scanline(imdec);
   memcpy(imdec->buffer.buffer, ret, img->width*sizeof(CARD32));
   stop_image_decoding(&imdec);

#ifdef HAVE_MMX
   mmx_off();
#endif

   return (UInt_t*)ret;
}

//______________________________________________________________________________
void TASImage::FillRectangleInternal(UInt_t col, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // Fills rectangle of size (width, height) at position (x,y) 
   // within the existing image with specified color.
   //
 
   ARGB32 color = (ARGB32)col;
  
   width = !width ? fImage->width : width;
   height = !height ? fImage->height : height;

   if (x < 0) {
      width += x;
      x = 0;
   }
   if (y < 0) {
      height += y;
      y = 0;
   }

   x = x > (int)fImage->width ? fImage->width : x;
   y = y > (int)fImage->height ? fImage->height : y;

   width = x + width > fImage->width ? fImage->width - x : width;
   height = y + height > fImage->height ? fImage->height - y : height;

   int idx = 0;
   int yy =0;
   int xx = 0;

   if (!fImage->alt.argb32) {
      fill_asimage(fgVisual, fImage, x, y, x+width, height, color); // x+width - because of bug in ASImage
   } else {
      for (UInt_t i = 0; i < height; i++) {
         for (UInt_t j = 0; j < width; j++) {
            yy = y + i;
            xx = x + j;
            if ((yy < 0) || (yy >= (int)fImage->height) || (xx < 0) || (xx >= (int)fImage->width)) continue;
            idx = yy*fImage->width + xx;
            _alphaBlend(&fImage->alt.argb32[idx], &color);
         }
      }
   }
}

//______________________________________________________________________________
void TASImage::FillRectangle(const char *col, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // Fills rectangle of size (width, height) at position (x,y) 
   // within the existing image with specified color.
   //
   // To create new image with Fill method the following code can be used:
   //
   //  TImage *img = TImage::Create();
   //  img->Fill("#FF00FF", 0, 0, 400, 300);

   if (!InitVisual()) {
      Warning("Fill", "Visual not initiated");
      return;
   }

   ARGB32 color = ARGB32_White;

   if (col) {
      parse_argb_color(col, &color);
   }

   if (!fImage) {
      fImage = create_asimage(width ? width : 20, height ? height : 20, 0);
      x = 0;
      y = 0;
   }

   FillRectangleInternal((UInt_t)color, x, y, width, height);
   UnZoom();
}

//______________________________________________________________________________
void TASImage::DrawVLine(UInt_t x, UInt_t y1, UInt_t y2, UInt_t col, UInt_t thick)
{
   // vertical line

   ARGB32 color = (ARGB32)col;

   UInt_t half = 0;

   if (thick > 1) {
      half = thick >> 1;
      if (x > half) {
         x =  x - half;
      } else {
         x = 0;
         thick += (x - half);
      }
   }

   y2 = y2 >= fImage->height ? fImage->height - 1 : y2;
   y1 = y1 >= fImage->height ? fImage->height - 1 : y1;
   x = x + thick >= fImage->width ? fImage->width - thick - 1 : x;

   for (UInt_t y = y1; y <= y2; y++) {
      for (UInt_t w = 0; w < thick; w++) {
         if (x + w < fImage->width) {
            _alphaBlend(&fImage->alt.argb32[y*fImage->width + (x + w)], &color);
         }
      }
   }
}

//______________________________________________________________________________
void TASImage::DrawHLine(UInt_t y, UInt_t x1, UInt_t x2, UInt_t col, UInt_t thick)
{
   // horizontal line

   ARGB32 color = (ARGB32)col;

   UInt_t half = 0;

   if (thick > 1) {
      half = thick >> 1;
      if (y > half) {
         y =  y - half;
      } else {
         y = 0;
         thick += (y - half);
      }
   }

   y = y + thick >= fImage->height ? fImage->height - thick - 1 : y;
   x2 = x2 >= fImage->width ? fImage->width - 1 : x2;
   x1 = x1 >= fImage->width ? fImage->width - 1 : x1;

   for (UInt_t x = x1; x <= x2; x++) {
      for (UInt_t w = 0; w < thick; w++) {
         if (y + w < fImage->height) {
            _alphaBlend(&fImage->alt.argb32[(y + w)*fImage->width + x], &color);
         }
      }
   }
}

//______________________________________________________________________________
void TASImage::DrawLine(UInt_t x1,  UInt_t y1, UInt_t x2, UInt_t y2,
                        const char *col, UInt_t thick)

{
   //

   ARGB32 color;
   parse_argb_color(col, &color);

   DrawLineInternal(x1, y1, x2, y2, (UInt_t)color, thick);
}

//______________________________________________________________________________
void TASImage::DrawLineInternal(UInt_t x1, UInt_t y1, UInt_t x2, UInt_t y2,
                                UInt_t col, UInt_t thick)
{
   // internal drawing

   int dx, dy, d;
   int i1, i2;
   int x, y, xend, yend;
   int xdir, ydir;
   int wid, q;
   int w, wstart;

   if (!InitVisual()) {
      Warning("DrawLine", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("DrawLine", "no image");
      return;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   if (!fImage->alt.argb32) {
      Warning("DrawLine", "Failed to get pixel array");
      return;
   }

   ARGB32 color = (ARGB32)col;

   dx = TMath::Abs(Int_t(x2) - Int_t(x1));
   dy = TMath::Abs(Int_t(y2) - Int_t(y1));

   if (!dx) {
      DrawVLine(x1, y2 > y1 ? y1 : y2, 
                    y2 > y1 ? y2 : y1, color, thick);
      return;
   }

   if (!dy) {
      DrawHLine(y1, x2 > x1 ? x1 : x2, 
                    x2 > x1 ? x2 : x1, color, thick);
      return;
   }

   if (dy <= dx) {
      double ac = TMath::Cos(TMath::ATan2(dy, dx));
      wid = ac != 0 ? int(thick/ac) : 1;
      wid = wid ? wid : 1;

      UInt_t ddy = dy << 1;
      d = ddy - dx;
      i1 = ddy;
      i2 = (dy - dx) << 1;

      if (x1 > x2) {
         x = x2;
         y = y2;
         ydir = -1;
         xend = x1;
      } else {
         x = x1;
         y = y1;
         ydir = 1;
         xend = x2;
      }

      wstart = y - (wid >> 1);

      for (w = wstart; w < wstart + wid; w++) {
         _alphaBlend(&fImage->alt.argb32[w*fImage->width + x], &color);
      }

      q = (y2 - y1) * ydir;

      if (q > 0) {

         while (x < xend) {
            wstart = y - (wid >> 1);

            for (w = wstart; w < wstart + wid; w++) {
               _alphaBlend(&fImage->alt.argb32[w*fImage->width + x], &color);
            }
            x++;
            d += i1;

            if (d >= 0) {
               y++;
               d += i2;
            } 

         }
      } else {

         while (x < xend) {
            wstart = y - (wid >> 1);

            for (w = wstart; w < wstart + wid; w++) {
               _alphaBlend(&fImage->alt.argb32[w*fImage->width + x], &color);
            }
            x++;
            d += i1;
            if (d >= 0) {
               y--;
               d += i2;
            }

         }
      }
   } else {

      double as = TMath::Sin(TMath::ATan2(dy, dx));
      wid = as != 0 ? int(thick/as) : 1;
      wid = wid ? wid : 1;

      UInt_t ddx = dx << 1;

      d = ddx - dy;
      i1 = ddx;
      i2 = (dx - dy) << 1;

      if (y1 > y2) {
         y = y2;
         x = x2;
         yend = y1;
         xdir = -1;
      } else {
         y = y1;
         x = x1;
         yend = y2;
         xdir = 1;
      }

      wstart = x - (wid >> 1);

      for (w = wstart; w < wstart + wid; w++) {
         _alphaBlend(&fImage->alt.argb32[y*fImage->width + w], &color);
      }

      q = (x2 - x1) * xdir;

      if (q > 0) {

         while (y < yend) {
            wstart = x - (wid >> 1);

            for (w = wstart; w < wstart + wid; w++) {
               _alphaBlend(&fImage->alt.argb32[y*fImage->width + w], &color);
            }

            d += i1;
            y++;

            if (d >= 0) {
               x++;
               d += i2;
            }

         }
      } else {
         while (y < yend) {
            wstart = x - (wid >> 1);

            for (w = wstart; w < wstart + wid; w++) {
               _alphaBlend(&fImage->alt.argb32[y*fImage->width + w], &color);
            }

            d += i1;
            y++;

            if (d >= 0) {
               x--;
               d += i2;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TASImage::DrawRectangle(UInt_t x, UInt_t y, UInt_t w, UInt_t h, 
                             const char *col, UInt_t thick)
{
   // draw rectangle

   ARGB32 color;
   parse_argb_color(col, &color);

   DrawHLine(y, x, x + w, (UInt_t)color, thick);
   DrawVLine(x + w, y, y + h, (UInt_t)color, thick);
   DrawHLine(y + h, x, x + w, (UInt_t)color, thick);
   DrawVLine(x, y, y + h, (UInt_t)color, thick);
}

//______________________________________________________________________________
void TASImage::DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, const char *col, 
                       UInt_t thick, Int_t mode)
{
   // draw box

   Int_t x = TMath::Min(x1, x2);
   Int_t y = TMath::Min(y1, y2);
   Int_t w = TMath::Abs(x2 - x1);
   Int_t h = TMath::Abs(y2 - y1);

   switch (mode) {

      case TVirtualX::kHollow:
         DrawRectangle(x, y, w, h, col, thick);
         break;

      case TVirtualX::kFilled:
         FillRectangle(col, x, y, w, h);
         break;

      default:
         break;
   }
}

//______________________________________________________________________________
void TASImage::DrawDashHLine(UInt_t y, UInt_t x1, UInt_t x2, UInt_t nDash,
                             const char *pDash, UInt_t col, UInt_t thick)
{
   // draw dashed horizontal line

   UInt_t iDash = 0;    // index of current dash
   int i = 0;

   ARGB32 color = (ARGB32)col;

   UInt_t half = 0;

   if (thick > 1) {
      half = thick >> 1;
      if (y > half) {
         y =  y - half;
      } else {
         y = 0;
         thick += (y - half);
      }
   }

   y = y + thick >= fImage->height ? fImage->height - thick - 1 : y;
   x2 = x2 >= fImage->width ? fImage->width - 1 : x2;
   x1 = x1 >= fImage->width ? fImage->width - 1 : x1;

   // switch x1, x2 
   UInt_t tmp = x1;
   x1 = x2 < x1 ? x2 : x1;
   x2 = x2 < tmp ? tmp : x2;

   for (UInt_t x = x1; x <= x2; x++) {
      for (UInt_t w = 0; w < thick; w++) {
         if (y + w < fImage->height) {
            if ((iDash%2)==0) {
               _alphaBlend(&fImage->alt.argb32[(y + w)*fImage->width + x], &color);
            }
         }
      }
      i++;

      if (i >= pDash[iDash]) {
         iDash++;
         i = 0;
      }
      if (iDash >= nDash) {
         iDash = 0;
         i = 0;
      }
   }
}

//______________________________________________________________________________
void TASImage::DrawDashVLine(UInt_t x, UInt_t y1, UInt_t y2, UInt_t nDash,
                             const char *pDash, UInt_t col, UInt_t thick)
{
   // draw dashed vertical line

   UInt_t iDash = 0;    // index of current dash
   int i = 0;

   ARGB32 color = (ARGB32)col;

   UInt_t half = 0;

   if (thick > 1) {
      half = thick >> 1;
      if (x > half) {
         x =  x - half;
      } else {
         x = 0;
         thick += (x - half);
      }
   }

   y2 = y2 >= fImage->height ? fImage->height - 1 : y2;
   y1 = y1 >= fImage->height ? fImage->height - 1 : y1;

   // switch x1, x2 
   UInt_t tmp = y1;
   y1 = y2 < y1 ? y2 : y1;
   y2 = y2 < tmp ? tmp : y2;

   x = x + thick >= fImage->width ? fImage->width - thick - 1 : x;

   for (UInt_t y = y1; y <= y2; y++) {
      for (UInt_t w = 0; w < thick; w++) {
         if (x + w < fImage->width) {
            if ((iDash%2)==0) {
               _alphaBlend(&fImage->alt.argb32[y*fImage->width + (x + w)], &color);
            }
         }
      }
      i++;

      if (i >= pDash[iDash]) {
         iDash++;
         i = 0;
      }
      if (iDash >= nDash) {
         iDash = 0;
         i = 0;
      }
   }
}

//______________________________________________________________________________
void TASImage::DrawDashZLine(UInt_t x1, UInt_t y1, UInt_t x2, UInt_t y2,
                             UInt_t nDash, const char *tDash, UInt_t color)
{
   // draw dashed line with 1 pixel width

   int dx, dy, d;
   int i, i1, i2, i3;
   int x, y, xend, yend;
   int xdir, ydir;
   int q;
   UInt_t iDash = 0;    // index of current dash

   dx = TMath::Abs(Int_t(x2) - Int_t(x1));
   dy = TMath::Abs(Int_t(y2) - Int_t(y1));

   char *pDash = new char[nDash];

   if (dy <= dx) {
      double ac = TMath::Cos(TMath::ATan2(dy, dx));

      for (i = 0; i < (int)nDash; i++) {
         pDash[i] = int(tDash[i] * ac);
      }

      UInt_t ddy = dy << 1;
      d = ddy - dx;
      i1 = ddy;
      i2 = (dy - dx) << 1;
      i = 0;

      if (x1 > x2) {
         x = x2;
         y = y2;
         ydir = -1;
         xend = x1;
      } else {
         x = x1;
         y = y1;
         ydir = 1;
         xend = x2;
      }

      _alphaBlend(&fImage->alt.argb32[y*fImage->width + x], (ARGB32*)&color);

      q = (y2 - y1) * ydir;
      i3 = i2 - i1;
      d -= i1;

      if (q > 0) {

         while (x < xend) {

            if ((iDash%2)==0) {
               _alphaBlend(&fImage->alt.argb32[y*fImage->width + x], (ARGB32*)&color);
            }
            d += i1;
            x++;

            if (d >= 0) {
               y++;
               d += i3;
            }

            i++;
            if (i >= pDash[iDash]) {
               iDash++;
               i = 0;
            }
            if (iDash >= nDash) {
               iDash = 0;
               i = 0;
            }
         }
      } else {
      i3 = i2 - i1;
      d -= i1;

         while (x < xend) {

            if ((iDash%2)==0) {
               _alphaBlend(&fImage->alt.argb32[y*fImage->width + x], (ARGB32*)&color);
            }
            d += i1;
            x++;
            if (d >= 0) {
               y--;
               d += i3;
            }

            i++;
            if (i >= pDash[iDash]) {
               iDash++;
               i = 0;
            }
            if (iDash >= nDash) {
               iDash = 0;
               i = 0;
            }
         }
      }
   } else {
      double as = TMath::Sin(TMath::ATan2(dy, dx));

      for (i = 0; i < (int)nDash; i++) {
         pDash[i] = int(tDash[i] * as);
      }

      UInt_t ddx = dx << 1;

      d = ddx - dy;
      i1 = ddx;
      i2 = (dx - dy) << 1;
      i = 0;

      if (y1 > y2) {
         y = y2;
         x = x2;
         yend = y1;
         xdir = (-1);
      } else {
         y = y1;
         x = x1;
         yend = y2;
         xdir = 1;
      }

      _alphaBlend(&fImage->alt.argb32[y*fImage->width + x], (ARGB32*)&color);

      q = (x2 - x1) * xdir;

      if (q > 0) {

         while (y < yend) {

            if ((iDash%2)==0) {
               _alphaBlend(&fImage->alt.argb32[y*fImage->width + x], (ARGB32*)&color);
            }
            d += i1;
            y++;

            if (d >= 0) {
               x++;
               d += i2;
            }

            i++;
            if (i >= pDash[iDash]) {
               iDash++;
               i = 0;
            }
            if (iDash >= nDash) {
               iDash = 0;
               i = 0;
            }
         }
      } else {
         while (y < yend) {

            if ((iDash%2)==0) {
               _alphaBlend(&fImage->alt.argb32[y*fImage->width + x], (ARGB32*)&color);
            }

            d += i1;
            y++;

            if (d >= 0) {
               x--;
               d += i2;
            }

            i++;
            if (i >= pDash[iDash]) {
               iDash++;
               i = 0;
            }
            if (iDash >= nDash) {
               iDash = 0;
               i = 0;
            }
         }
      }
   }
   delete [] pDash; 
}

//______________________________________________________________________________
void TASImage::DrawDashLine(UInt_t x1,  UInt_t y1, UInt_t x2, UInt_t y2, UInt_t nDash,
                            const char *pDash, const char *col, UInt_t thick)

{
   // draw dashed line

   if (!InitVisual()) {
      Warning("DrawDashLine", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("DrawDashLine", "no image");
      return;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   if (!fImage->alt.argb32) {
      Warning("DrawDashLine", "Failed to get pixel array");
      return;
   }

   if ((nDash < 2) || !pDash || (nDash%2)) {
      Warning("DrawDashLine", "Wrong input parameters n=%d %d", nDash, sizeof(pDash)-1);
      return;
   }

   ARGB32 color;
   parse_argb_color(col, &color);

   if (x1 == x2) {
      DrawDashVLine(x1, y1, y2, nDash, pDash, (UInt_t)color, thick);
   } else if (y1 == y2) {
      DrawDashHLine(y1, x1, x2, nDash, pDash, (UInt_t)color, thick);
   } else {
      DrawDashZLine(x1, y1, x2, y2, nDash, pDash, (UInt_t)color);
   }
}

//______________________________________________________________________________
void TASImage::DrawPolyLine(UInt_t nn, TPoint *xy, const char *col, UInt_t thick,
                            TImage::ECoordMode mode)
{
   // draw poly line

   ARGB32 color;
   parse_argb_color(col, &color);

   Int_t x0 = xy[0].GetX();
   Int_t y0 = xy[0].GetY();
   Int_t x = 0;
   Int_t y = 0;

   for (UInt_t i = 1; i < nn; i++) {
      x = (mode == kCoordModePrevious) ? x + xy[i].GetX() : xy[i].GetX();
      y = (mode == kCoordModePrevious) ? y + xy[i].GetY() : xy[i].GetY();

      DrawLineInternal(x0, y0, x, y, (UInt_t)color, thick);
      x0 = x;
      y0 = y;
   }
}

//______________________________________________________________________________
void TASImage::PutPixel(Int_t x, Int_t y, const char *col)
{
   // draw point at specified position

   if (!InitVisual()) {
      Warning("PutPixel", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("PutPixel", "no image");
      return;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   if (!fImage->alt.argb32) {
      Warning("PutPixel", "Failed to get pixel array");
      return;
   }

   ARGB32 color;
   parse_argb_color(col, &color);

   if ((x < 0) || (y < 0) || (x >= (int)fImage->width) || (y >= (int)fImage->height)) {
      Warning("PutPixel", "Out of range width=%d x=%d, height=%d y=%d", 
               fImage->width, x, fImage->height, y);
      return;
   }
   _alphaBlend(&fImage->alt.argb32[y*fImage->width + x], &color);
}

//______________________________________________________________________________
void TASImage::PolyPoint(UInt_t npt, TPoint *ppt, const char *col, TImage::ECoordMode mode)
{
   // draw poly point

   if (!InitVisual()) {
      Warning("PolyPoint", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("PolyPoint", "no image");
      return;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   if (!fImage->alt.argb32) {
      Warning("PolyPoint", "Failed to get pixel array");
      return;
   }

   if (!npt || !ppt) {
      Warning("PolyPoint", "No points specified");
      return;
   }

   TPoint *ipt = 0;
   UInt_t i = 0;
   ARGB32 color;
   parse_argb_color(col, &color);

   //make pointlist origin relative
   if (mode == kCoordModePrevious) {
      ipt = new TPoint[npt];

      for (i = 0; i < npt; i++) {
         ipt[i].fX += ppt[i].fX;
         ipt[i].fY += ppt[i].fY;
      }
   }
   int x, y;

   for (i = 0; i < npt; i++) {
      x = ipt ? ipt[i].fX : ppt[i].fX;
      y = ipt ? ipt[i].fY : ppt[i].fY;

      if ((x < 0) || (y < 0) || (x >= (int)fImage->width) || (y >= (int)fImage->height)) {
         continue;
      }
      _alphaBlend(&fImage->alt.argb32[y*fImage->width + x], &color);
   }

   if (ipt) {
      delete [] ipt;
   }
}

//______________________________________________________________________________
void TASImage::DrawSegments(UInt_t nseg, Segment_t *seg, const char *col, UInt_t thick)
{
   // draw segments

   if (!nseg || !seg) {
      Warning("DrawSegments", "Ivalid data nseg=%d seg=%d", nseg, seg);
      return;
   }

   TPoint pt[2];

   for (UInt_t i = 0; i < nseg; i++) {
      pt[0].fX = seg->fX1;
      pt[1].fX = seg->fX2;
      pt[0].fY = seg->fY1;
      pt[1].fY = seg->fY2;

      DrawPolyLine(2, pt, col, thick, kCoordModeOrigin);
      seg++;
   }
}

//______________________________________________________________________________
void TASImage::FillSpans(UInt_t npt, TPoint *ppt, UInt_t *widths, const char *col,
                         const char *stipple, UInt_t w, UInt_t h)
{
   // fill spans with specified color or/and stipple

   if (!InitVisual()) {
      Warning("FillSpans", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("FillSpans", "no image");
      return;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   if (!fImage->alt.argb32) {
      Warning("FillSpans", "Failed to get pixel array");
      return;
   }

   if (!npt || !ppt || !widths || (stipple && (!w || !h))) {
      Warning("FillSpans", "Invalid input data npt=%d ppt=%x stipple=%x stipple=%x w=%d h=%d",
              npt, ppt, widths, stipple, w, h);
      return;
   }

   ARGB32 color;
   parse_argb_color(col, &color);

   Int_t idx = 0;
   Int_t ii = 0;
   Int_t sz = fImage->width*fImage->height;
   UInt_t x = 0;  
   UInt_t xx = 0;
   UInt_t yy = 0;

   for (UInt_t i = 0; i < npt; i++) {
      for (UInt_t j = 0; j < widths[i]; j++) {
         x = ppt[i].fX + j;
         idx = ppt[i].fY*fImage->width + x;

         if (idx < sz) {
            xx = x%w;
            yy = ppt[i].fY%h;
            ii = yy*w + xx;

            if (!stipple) _alphaBlend(&fImage->alt.argb32[idx], &color);
            else if (stipple[ii/8] & (1 << (ii%8))) _alphaBlend(&fImage->alt.argb32[idx], &color);

         }
      }
   }
}

//______________________________________________________________________________
void TASImage::FillSpans(UInt_t npt, TPoint *ppt, UInt_t *widths, TImage *tile)
{
   // fille spans with tile image

   if (!InitVisual()) {
      Warning("FillSpans", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("FillSpans", "no image");
      return;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   if (!fImage->alt.argb32) {
      Warning("FillSpans", "Failed to get pixel array");
      return;
   }

   if (!npt || !ppt || !widths || !tile) {
      Warning("FillSpans", "Invalid input data npt=%d ppt=%x widths=%x tile=%x",
              npt, ppt, widths, tile);
      return;
   }

   Int_t idx = 0;
   Int_t ii = 0;
   Int_t sz = fImage->width*fImage->height;
   UInt_t x = 0;  
   UInt_t *arr = tile->GetArgbArray();
   UInt_t xx = 0;
   UInt_t yy = 0;

   for (UInt_t i = 0; i < npt; i++) {
      for (UInt_t j = 0; j < widths[i]; j++) {
         x = ppt[i].fX + j;
         idx = ppt[i].fY*fImage->width + x;

         if (idx < sz) {
            xx = x%tile->GetWidth();
            yy = ppt[i].fY%tile->GetHeight();
            ii = yy*tile->GetWidth() + xx;
            _alphaBlend(&fImage->alt.argb32[idx], (ARGB32*)&arr[ii]);
         }
      }
   }
}

//______________________________________________________________________________
void TASImage::CropSpans(UInt_t npt, TPoint *ppt, UInt_t *widths)
{
   //  crop spans

   if (!InitVisual()) {
      Warning("CropSpans", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("CropSpans", "no image");
      return;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   if (!fImage->alt.argb32) {
      Warning("CropSpans", "Failed to get pixel array");
      return;
   }

   if (!npt || !ppt || !widths) {
      Warning("CropSpans", "No points specified npt=%d ppt=%x widths=%x", npt, ppt, widths);
      return;
   }

   int y0 = ppt[0].fY;
   int y1 = ppt[npt-1].fY;
   UInt_t y = 0;
   UInt_t x = 0;
   UInt_t i = 0;
   UInt_t idx = 0;
   UInt_t sz = fImage->width*fImage->height;

   for (y = 0; (int)y < y0; y++) {
      for (x = 0; x < fImage->width; x++) {
         idx = y*fImage->width + x;
         if (idx < sz) fImage->alt.argb32[idx] = 0;
      }
   }

   for (i = 0; i < npt; i++) {
      for (x = 0; (int)x < ppt[i].fX; x++) {
         idx = ppt[i].fY*fImage->width + x;
         if (idx < sz) fImage->alt.argb32[idx] = 0;
      }
      for (x = ppt[i].fX + widths[i] + 1; x < fImage->width; x++) {
         idx = ppt[i].fY*fImage->width + x;
         if (idx < sz) fImage->alt.argb32[idx] = 0;
      }
   }
   for (y = y1; y < fImage->height; y++) {
      for (x = 0; x < fImage->width; x++) {
         idx = y*fImage->width + x;
         if (idx < sz) fImage->alt.argb32[idx] = 0;
      }
   }
}


//______________________________________________________________________________
void TASImage::CopyArea(TImage *dst, Int_t xsrc, Int_t ysrc, UInt_t w,  UInt_t h,
                        Int_t xdst, Int_t ydst, Int_t gfunc, EColorChan)
{
   // Copy source region to the destination image. Copy is done according
   // to specified function:
   //
   // enum EGraphicsFunction {
   //    kGXclear = 0,               // 0
   //    kGXand,                     // src AND dst
   //    kGXandReverse,              // src AND NOT dst
   //    kGXcopy,                    // src (default)
   //    kGXandInverted,             // NOT src AND dst
   //    kGXnoop,                    // dst
   //    kGXxor,                     // src XOR dst
   //    kGXor,                      // src OR dst
   //    kGXnor,                     // NOT src AND NOT dst
   //    kGXequiv,                   // NOT src XOR dst
   //    kGXinvert,                  // NOT dst
   //    kGXorReverse,               // src OR NOT dst
   //    kGXcopyInverted,            // NOT src
   //    kGXorInverted,              // NOT src OR dst
   //    kGXnand,                    // NOT src OR NOT dst
   //    kGXset                      // 1
   // };

   if (!InitVisual()) {
      Warning("CopyArea", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("CopyArea", "no image");
      return;
   }

   ASImage *out = ((TASImage*)dst)->GetImage();

   int x = 0;
   int y = 0;
   int idx = 0;
   int idx2 = 0;
   xsrc = xsrc < 0 ? 0 : xsrc;
   ysrc = ysrc < 0 ? 0 : ysrc;
   xsrc = xsrc > (int)fImage->width ? fImage->width : xsrc;
   ysrc = ysrc > (int)fImage->height ? fImage->height : ysrc;

   w = xsrc + w > fImage->width ? fImage->width - xsrc : w;
   h = ysrc + h > fImage->height ? fImage->height - ysrc : h;

   if (fImage->alt.argb32 && out->alt.argb32) {
      for (y = 0; y < (int)h; y++) {
         for (x = 0; x < (int)w; x++) {
            idx = (ysrc + y)*fImage->width + x + xsrc;
            if ((x + xdst < 0) || (ydst + y < 0) || 
                (x + xdst >= (int)out->width) || (y + ydst >= (int)out->height) ) continue;

            idx2 = (ydst + y)*out->width + x + xdst;

            switch ((EGraphicsFunction)gfunc) {
               case kGXclear:
                  out->alt.argb32[idx2] = 0;
                  break;
               case kGXand:
                  out->alt.argb32[idx2] &= fImage->alt.argb32[idx];
                  break;
               case kGXandReverse:
                  out->alt.argb32[idx2] = fImage->alt.argb32[idx] & (~out->alt.argb32[idx2]);
                  break;
               case kGXandInverted:
                  out->alt.argb32[idx2] &= ~fImage->alt.argb32[idx];
                  break;
               case kGXnoop:
                  break;
               case kGXxor:
                  out->alt.argb32[idx2] ^= fImage->alt.argb32[idx];
                  break;
               case kGXor:
                  out->alt.argb32[idx2] |= fImage->alt.argb32[idx];
                  break;
               case kGXnor:
                  out->alt.argb32[idx2] = (~fImage->alt.argb32[idx]) & (~out->alt.argb32[idx2]);
                  break;
               case kGXequiv:
                  out->alt.argb32[idx2] ^= ~fImage->alt.argb32[idx];
                  break;
               case kGXinvert:
                  out->alt.argb32[idx2] = ~out->alt.argb32[idx2];
                  break;
               case kGXorReverse:
                  out->alt.argb32[idx2] = fImage->alt.argb32[idx] | (~out->alt.argb32[idx2]);
                  break;
               case kGXcopyInverted:
                  out->alt.argb32[idx2] = ~fImage->alt.argb32[idx];
                  break;
               case kGXorInverted:
                  out->alt.argb32[idx2] |= ~fImage->alt.argb32[idx];
                  break;
               case kGXnand:
                  out->alt.argb32[idx2] = (~fImage->alt.argb32[idx]) | (~out->alt.argb32[idx2]);
                  break;
               case kGXset:
                  out->alt.argb32[idx2] = 0xFFFFFFFF;
                  break;
               case kGXcopy:
               default:
                  out->alt.argb32[idx2] = fImage->alt.argb32[idx];
                  break;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TASImage::DrawCellArray(Int_t x1, Int_t y1, Int_t x2, Int_t y2, Int_t nx,
                             Int_t ny, UInt_t *ic)
{
   // Draw a cell array.
   // x1,y1        : left down corner
   // x2,y2        : right up corner
   // nx,ny        : array size
   // ic           : array of ARGB32 colors
   //
   // Draw a cell array. The drawing is done with the pixel presicion
   // if (X2-X1)/NX (or Y) is not a exact pixel number the position of
   // the top rigth corner may be wrong.

   int i, j, ix, iy, w, h;

   ARGB32 color = 0xFFFFFFFF;
   ARGB32 icol;

   w  = TMath::Max((x2-x1)/(nx),1);
   h  = TMath::Max((y1-y2)/(ny),1);
   ix = x1;

   for (i = 0; i < nx; i++) {
      iy = y1 - h;
      for (j = 0; j < ny; j++) {
         icol = (ARGB32)ic[i + (nx*j)];
         if (icol != color) {
            color = icol;
         }
         FillRectangleInternal((UInt_t)color, ix, iy, w, h);
         iy = iy - h;
      }
      ix = ix + w;
   }
}

//______________________________________________________________________________
UInt_t TASImage::AlphaBlend(UInt_t bot, UInt_t top)
{
   // returns alphablended value computed from bottom and top pixel values

   UInt_t ret = bot;

   _alphaBlend((ARGB32*)&ret, (ARGB32*)&top);
   return ret;
}

//////////////////////// polygon filling //////////////////////////////
#define BRESINITPGON(dy, x1, x2, xStart, d, m, m1, incr1, incr2) { \
    int dx;\
\
    if ((dy) != 0) { \
        xStart = (x1); \
        dx = (x2) - xStart; \
        if (dx < 0) { \
            m = dx / (dy); \
            m1 = m - 1; \
            incr1 = -2 * dx + 2 * (dy) * m1; \
            incr2 = -2 * dx + 2 * (dy) * m; \
            d = 2 * m * (dy) - 2 * dx - 2 * (dy); \
        } else { \
            m = dx / (dy); \
            m1 = m + 1; \
            incr1 = 2 * dx - 2 * (dy) * m1; \
            incr2 = 2 * dx - 2 * (dy) * m; \
            d = -2 * m * (dy) + 2 * dx; \
        } \
    } \
}

#define BRESINCRPGON(d, minval, m, m1, incr1, incr2) { \
    if (m1 > 0) { \
        if (d > 0) { \
            minval += m1; \
            d += incr1; \
        } \
        else { \
            minval += m; \
            d += incr2; \
        } \
    } else {\
        if (d >= 0) { \
            minval += m1; \
            d += incr1; \
        } \
        else { \
            minval += m; \
            d += incr2; \
        } \
    } \
}

//______________________________________________________________________________
static int GetPolyYBounds(TPoint *pts, int n, int *by, int *ty)
{
   //

   register TPoint *ptMin;
   int ymin, ymax;
   TPoint *ptsStart = pts;

   ptMin = pts;
   ymin = ymax = (pts++)->fY;

   while (--n > 0) {
      if (pts->fY < ymin) {
         ptMin = pts;
         ymin = pts->fY;
      }
      if (pts->fY > ymax) {
         ymax = pts->fY;
      }
      pts++;
    }

    *by = ymin;
    *ty = ymax;
    return (ptMin - ptsStart);
}

//______________________________________________________________________________
void TASImage::GetPolygonSpans(UInt_t npt, TPoint *ppt, UInt_t *nspans, 
                               TPoint **outPoint, UInt_t **outWidth)
{
   // The code is taken on Xserver/mi/mipolycon.c
   //    "Copyright 1987, 1998  The Open Group"

   int xl = 0;                   // x vals of leftedges
   int xr = 0;                   // x vals of right edges
   int dl = 0;                   // decision variables 
   int dr = 0;                   // decision variables 
   int ml = 0;                   // left edge slope
   int m1l = 0;                  // left edge slope+1
   int mr = 0, m1r = 0;          // right edge slope and slope+1
   int incr1l = 0, incr2l = 0;   // left edge error increments
   int incr1r = 0, incr2r = 0;   // right edge error increments
   int dy;                       // delta y
   int y;                        // current scanline
   int left, right;              // indices to first endpoints
   int i;                        // loop counter
   int nextleft, nextright;      // indices to second endpoints
   TPoint *ptsOut;               // output buffer
   UInt_t *width;                // output buffer
   TPoint *firstPoint;
   UInt_t *firstWidth;
   int imin;                     // index of smallest vertex (in y)
   int ymin;                     // y-extents of polygon
   int ymax;
 
   *nspans = 0;

   if (!InitVisual()) {
      Warning("GetPolygonSpans", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("GetPolygonSpans", "no image");
      return;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   if (!fImage->alt.argb32) {
      Warning("GetPolygonSpans", "Failed to get pixel array");
      return;
   }

   if ((npt < 3) || !ppt) {
      Warning("GetPolygonSpans", "No points specified npt=%d ppt=%x", npt, ppt);
      return;
   }

   //  find leftx, bottomy, rightx, topy, and the index
   //  of bottomy. Also translate the points.

   imin = GetPolyYBounds(ppt, npt, &ymin, &ymax);

   dy = ymax - ymin + 1;
   if ((npt < 3) || (dy < 0)) return;

   ptsOut = firstPoint = new TPoint[dy];
   width = firstWidth = new UInt_t[dy];

   nextleft = nextright = imin;
   y = ppt[nextleft].fY;

   //  loop through all edges of the polygon
   do {
      // add a left edge if we need to
 
      if (ppt[nextleft].fY == y) {
         left = nextleft;

         //  find the next edge, considering the end
         //  conditions of the array.
 
         nextleft++;
         if (nextleft >= (int)npt) {
            nextleft = 0;
         }

         // now compute all of the random information
         // needed to run the iterative algorithm.

         BRESINITPGON(ppt[nextleft].fY - ppt[left].fY,
                      ppt[left].fX, ppt[nextleft].fX,
                      xl, dl, ml, m1l, incr1l, incr2l);
      }

      // add a right edge if we need to
      if (ppt[nextright].fY == y) {
         right = nextright;

         // find the next edge, considering the end
         // conditions of the array.

         nextright--;
         if (nextright < 0) {
            nextright = npt-1;
         }

         //  now compute all of the random information
         //  needed to run the iterative algorithm.

         BRESINITPGON(ppt[nextright].fY - ppt[right].fY,
                      ppt[right].fX, ppt[nextright].fX,
                      xr, dr, mr, m1r, incr1r, incr2r);
      }

      // generate scans to fill while we still have
      //  a right edge as well as a left edge.

      i = min(ppt[nextleft].fY, ppt[nextright].fY) - y;

      // in case we're called with non-convex polygon
      if (i < 0) {
         delete [] firstWidth;
         delete [] firstPoint;
         firstPoint = 0;
         firstWidth = 0;
         return;
      }

      while (i-- > 0)  {
         ptsOut->fY = y;

         // reverse the edges if necessary
         if (xl < xr) {
            *(width++) = xr - xl;
            (ptsOut++)->fX = xl;
         } else {
            *(width++) = xl - xr;
            (ptsOut++)->fX = xr;
         }
         y++;

         // increment down the edges
         BRESINCRPGON(dl, xl, ml, m1l, incr1l, incr2l);
         BRESINCRPGON(dr, xr, mr, m1r, incr1r, incr2r);
      }
   }  while (y != ymax);

   *nspans = UInt_t(ptsOut - firstPoint);
   *outPoint = firstPoint;
   *outWidth = firstWidth;
}

//______________________________________________________________________________
void TASImage::FillPolygon(UInt_t npt, TPoint *ppt, const char *col,
                           const char *stipple, UInt_t w, UInt_t h)
{
   // Fill a convex polygon with background color or bitmap
   // For non convex polygon one must use DrawFillArea method

   UInt_t  nspans = 0;
   TPoint *firstPoint = 0;   // output buffer
   UInt_t *firstWidth = 0;   // output buffer

   GetPolygonSpans(npt, ppt, &nspans, &firstPoint, &firstWidth);

   if (nspans) {
      FillSpans(nspans, firstPoint, firstWidth, col, stipple, w, h);
    
      delete [] firstWidth;
      delete [] firstPoint;
   }
}

//______________________________________________________________________________
void TASImage::FillPolygon(UInt_t npt, TPoint *ppt, TImage *tile)
{
   // Fill a convex polygon with background image.  
   // For non convex polygon one must use DrawFillArea method

   UInt_t  nspans = 0;
   TPoint *firstPoint = 0;   // output buffer
   UInt_t *firstWidth = 0;   // output buffer

   GetPolygonSpans(npt, ppt, &nspans, &firstPoint, &firstWidth);

   if (nspans) {
      FillSpans(nspans, firstPoint, firstWidth, tile);
    
      delete [] firstWidth;
      delete [] firstPoint;
   }
}

//______________________________________________________________________________
void TASImage::CropPolygon(UInt_t npt, TPoint *ppt)
{
   // Crop a convex polygon.

   UInt_t  nspans = 0;
   TPoint *firstPoint = 0;
   UInt_t *firstWidth = 0;

   GetPolygonSpans(npt, ppt, &nspans, &firstPoint, &firstWidth);

   if (nspans) {
      CropSpans(nspans, firstPoint, firstWidth);
    
      delete [] firstWidth;
      delete [] firstPoint;
   }
}

class EdgeTableEntry : public TObject {
public:
   UInt_t fYmax;
   Bool_t fClock;
   int fMinor;    // minor axis
   int fD;        // decision variable
   int fM;        // slope
   int fM1;       // slope+1
   int fIncr1;    // error increment
   int fIncr2;    // error increment
};

//______________________________________________________________________________
void TASImage::GetFillAreaSpans(UInt_t npt, TPoint *ppt, UInt_t * /*nspans*/, 
                                TPoint ** /*firstPoint*/, UInt_t ** /*firstWidth*/)
{
   // fill a polygon (any type convex, non-convex)
   //
   // The code is based on Xserver/mi
   //    "Copyright 1987, 1998  The Open Group"

   if (!InitVisual()) {
      Warning("GetFillAreaSpans", "Visual not initiated");
      return;
   }

   if (!fImage) {
      Warning("GetFillAreaSpans", "no image");
      return;
   }

   if (!fImage->alt.argb32) {
      BeginPaint();
   }

   if (!fImage->alt.argb32) {
      Warning("GetFillAreaSpans", "Failed to get pixel array");
      return;
   }

   if ((npt < 3) || !ppt) {
      Warning("GetFillAreaSpans", "No points specified npt=%d ppt=%x", npt, ppt);
      return;
   }
/*
   TPoint *curr;
   TPoint *pts;
   TPoint *top;
   TPoint *bottom;
   int dy;
   int y; // the current scanline


   while (npt--)  {
      curr = pts++;

      if (prev->fY > curr->fY) {
         bottom = prev;
         top = curr;
            pETEs->flock = kFALSE;
      } else  {
         bottom = curr;
         top = prev;
            pETEs->fClock = kTRUE;
      }

      // don't add horizontal edges to the Edge table.
      if (bottom->fY != top->fY)  {
         pETEs->ymax = bottom->fY - 1;  // -1 so we don't get last scanline

         // initialize integer edge algorithm
         dy = bottom->fY - top->fY;
         BRESINITPGON(dy, top->fX, bottom->fX, pETEs->fMinor, pETEs->fD, 
                      pETEs->fM, bpETEs->fM1, pETEs->fIncr1, pETEs->fIncr2);

         if (!miInsertEdgeInET(ET, pETEs, top->fY, &pSLLBlock, &iSLLBlock)) {
		      miFreeStorage(pSLLBlock->next);
		      return;
	      }

         ET->ymax = max(ET->ymax, prev->fY);
         ET->ymin = min(ET->ymin, prev->fY);
         pETEs++;
      }

      prev = curr;
   }

   pSLL = ET.scanlines.next;

   for (y = ET.ymin; y < ET.ymax; y++)  {
      // Add a new edge to the active edge table when we get to the next edge.

      if (pSLL && (y == pSLL->scanline))  {
         miloadAET(&AET, pSLL->edgelist);
         pSLL = pSLL->next;
      }
      pPrevAET = &AET;
      pAET = AET.next;

      // for each active edge

      while (pAET)  {
         ptsOut->fX = pAET->bres.minor;
		   ptsOut++->fY = y;
         *width++ = pAET->next->bres.minor - pAET->bres.minor;
         nPts++;

         // send out the buffer when its full
         if (nPts == NUMPTSTOBUFFER) {
		      FillSpans(nPts, firstPoint, firstWidth);
            ptsOut = firstPoint;
            width = firstWidth;
            nPts = 0;
         }
   
         if (pAET->ymax == y) { //
            pPrevAET->next = pAET->next;
            pAET = pPrevAET->next;

            if (pAET) {
               pAET->back = pPrevAET;
            }
         } else {
            BRESINCRPGON(pAET->fD, pAET->fMinor, pAET->fM, 
                         pAET->fM1, pAET->fIncr1, pAET->fIncr2);
            pPrevAET = pAET;
            pAET = pAET->next;
         }
      }
    
      AET = AET->next;

      while (AET)  {
         pETEinsert = AET;
         pETEchase = AET;

         while (pETEchase->back->fMinor > AET->fMinor) {
            pETEchase = pETEchase->back;
         }

         AET = AET->next;

         if (pETEchase != pETEinsert) {
            pETEchaseBackTMP = pETEchase->back;
            pETEinsert->back->next = AET;

            if (AET) {
               AET->back = pETEinsert->back;
            }
            pETEinsert->next = pETEchase;
            pETEchase->back->next = pETEinsert;
            pETEchase->back = pETEinsert;
            pETEinsert->back = pETEchaseBackTMP;
         }
      }
   }

   //    Get any spans that we missed by buffering
   FillSpans(nPts, firstPoint, firstWidth, 1);
   DEALLOCATE_LOCAL(pETEs);
   miFreeStorage(SLLBlock.next);
*/
}

//______________________________________________________________________________
void TASImage::DrawFillArea(UInt_t npt, TPoint *ppt, const char *col, 
                           const char *stipple, UInt_t w, UInt_t h)
{
   // fill a polygon (any type convex, non-convex)

   UInt_t  nspans = 0;
   TPoint *firstPoint = 0;   // output buffer
   UInt_t *firstWidth = 0;   // output buffer

   GetFillAreaSpans(npt, ppt, &nspans, &firstPoint, &firstWidth);

   if (nspans) {
      FillSpans(nspans, firstPoint, firstWidth, col, stipple, w, h);
    
      delete [] firstWidth;
      delete [] firstPoint;
   }
}

//______________________________________________________________________________
void TASImage::DrawFillArea(UInt_t npt, TPoint *ppt, TImage *tile)
{
   // fill a polygon (any type convex, non-convex)

   UInt_t  nspans = 0;
   TPoint *firstPoint = 0;   // output buffer
   UInt_t *firstWidth = 0;   // output buffer

   GetFillAreaSpans(npt, ppt, &nspans, &firstPoint, &firstWidth);

   if (nspans) {
      FillSpans(nspans, firstPoint, firstWidth, tile);
    
      delete [] firstWidth;
      delete [] firstPoint;
   }
}


