// @(#)root/asimage:$Name:$:$Id:$
// Author: Reiner Rohlfs   28/11/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include <X11/Xlib.h>
extern "C" {
#   include <afterbase.h>
#   include <afterimage.h>
}

const Int_t kFRS = 5;  // size of frame of image on pad in pixels

ASVisual *TASImage::fgVisual;
Bool_t TASImage::fgInit = kFALSE;


ClassImp(TASImage)

//______________________________________________________________________________
TASImage::TASImage()
{
   // Default image ctor.

   fImage       = 0;
   fScaledImage = 0;

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
   fScaledImage = 0;

   if (img.IsValid()) {
      fImage = clone_asimage(img.fImage, SCL_DO_ALL);
      if (img.fImage->alt.vector) {
         Int_t size = img.fImage->width * img.fImage->height * sizeof(double);
         fImage->alt.vector = (double*)malloc(size);
         memcpy(fImage->alt.vector, img.fImage->alt.vector, size);
      }

      fZoomUpdate = kTRUE;
      fZoomOffX   = img.fZoomOffX;
      fZoomOffY   = img.fZoomOffY;
      fZoomWidth  = img.fZoomWidth;
      fZoomHeight = img.fZoomHeight;
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
      if (fScaledImage) {
         destroy_asimage(&fScaledImage);
         fScaledImage = 0;
      }

      fZoomUpdate = kTRUE;
      fZoomOffX   = img.fZoomOffX;
      fZoomOffY   = img.fZoomOffY;
      fZoomWidth  = img.fZoomWidth;
      fZoomHeight = img.fZoomHeight;
   }
   return *this;
}

//______________________________________________________________________________
TASImage::~TASImage()
{
   // Image dtor, clean up image and visual.

   if (fImage)
      destroy_asimage(&fImage);

   if (fScaledImage)
      destroy_asimage(&fScaledImage);
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

   if (fImage)
      destroy_asimage(&fImage);
   if (fScaledImage) {
      destroy_asimage(&fScaledImage);
      fScaledImage = 0;
   }

   fImage = file2ASImage(file, 0, SCREEN_GAMMA, GetImageCompression(), 0);

   fZoomUpdate = kFALSE;
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

   if (!ASImage2file(fImage, 0, file, atype, &parms))
      Error("WriteImage", "error writing file %s", file);
}

//______________________________________________________________________________
TImage::EImageFileTypes TASImage::GetFileType(const char *ext)
{
   // Return file type depending on specified extension.
   // Protected method.

   TString s(ext);

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

   if (!InitVisual()) return;

   if (fImage)
      destroy_asimage(&fImage);
   if (fScaledImage) {
      destroy_asimage(&fScaledImage);
      fScaledImage = 0;
   }

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

   fZoomUpdate = kFALSE;
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

   SetImage(imageData.GetElements(), width,
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

   if (!InitVisual()) return;

   if (fImage)
      destroy_asimage(&fImage);

   if (fScaledImage) {
      destroy_asimage(&fScaledImage);
      fScaledImage = 0;
   }

   SetName(pad->GetName());

   if (w == 0)
      w = pad->GetWw();
   if (h == 0)
      h = pad->GetWh();

   Int_t wid = (pad == pad->GetCanvas()) ? pad->GetCanvas()->GetCanvasID()
                                         : pad->GetPixmapID();
   Window wd = (Window) gVirtualX->GetWindowID(wid);

   fImage = pixmap2asimage(fgVisual, wd, x, y, w, h, AllPlanes, 0, 0);
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
   TObject::Draw(option);
}

//______________________________________________________________________________
void TASImage::Paint(Option_t *option)
{
   // Paint image in current pad. See Draw() function for drawing options.

   if (!fImage) {
      Error("Paint", "no image set");
      return;
   }

   if (!InitVisual()) return;

   Int_t   tile_x = 0, tile_y = 0;
   ULong_t tile_tint = 0;
   Bool_t  tile = kFALSE;

   TString opt = option;
   opt.ToLower();
   if (opt.Contains("t")) {
      char stint[64];
      if (sscanf(opt.Data()+opt.Index("t"), "t%d,%d,%s", &tile_x, &tile_y,
                 stint) <= 3) {
         tile = kTRUE;
         if (parse_argb_color(stint, &tile_tint) == stint)
            tile_tint = 0;
      } else
         Error("Paint", "tile option error");
   }

   ASImage *image = fImage;

   // Get geometry of pad
   Int_t to_w = TMath::Abs(gPad->XtoPixel(gPad->GetX2()) -
                           gPad->XtoPixel(gPad->GetX1()));
   Int_t to_h = TMath::Abs(gPad->YtoPixel(gPad->GetY2()) -
                           gPad->YtoPixel(gPad->GetY1()));
   Double_t pad_w = to_w;
   Double_t pad_h = to_h;

   // keep a frame of 5 pixels
   to_w -= 2 * kFRS;
   to_h -= 2 * kFRS;
   Int_t pal_w;

   if (fImage->alt.vector) {
      pal_w = Int_t(to_w * 0.2);
      to_w -= pal_w;
   }

   if (to_w < 2 * kFRS + 1 || to_h < 2 * kFRS + 1) {
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

      grad_im = make_gradient(fgVisual, &grad , UInt_t(0.3 * pal_w),
                              to_h - 20, SCL_DO_COLOR,
                              ASA_ASImage, 0, GetImageQuality());
      delete [] grad.color;
      delete [] grad.offset;
   }

   if (tile) {

      fScaledImage = tile_asimage(fgVisual, fImage, tile_x, tile_y,
                                  to_w, to_h, tile_tint, ASA_XImage, 0,
                                  ASIMAGE_QUALITY_TOP);
      image = fScaledImage;

   } else {

      // Scale and zoom image if needed
      if (Int_t(fImage->width) != to_w || Int_t(fImage->height) != to_h ||
          fImage->width != fZoomWidth || fImage->height != fZoomHeight) {
         if (fScaledImage) {
            if (Int_t(fScaledImage->width) != to_w ||
                Int_t(fScaledImage->height) != to_h ||
                fZoomUpdate) {
               destroy_asimage(&fScaledImage);
               fScaledImage = 0;
            }
         }
         if (!fScaledImage) {
            if (fImage->width != fZoomWidth || fImage->height != fZoomHeight) {
                // zoom and scale image
                ASImage *tmpImage = tile_asimage(fgVisual, fImage, fZoomOffX,
                                  fImage->height - fZoomHeight - fZoomOffY,
                                  fZoomWidth, fZoomHeight, 0, ASA_ASImage, GetImageCompression(),
                                  GetImageQuality());
               fScaledImage = scale_asimage(fgVisual, tmpImage, to_w, to_h,
                                            ASA_ASImage, GetImageCompression(),
                                            GetImageQuality());
               destroy_asimage(&tmpImage);
            } else {
               // scale image, no zooming
               fScaledImage = scale_asimage(fgVisual, fImage, to_w, to_h,
                                   /* ASA_XImage */  ASA_ASImage, GetImageCompression(),
                                            GetImageQuality());
            }
         }
         image = fScaledImage;
      }
   }
   fZoomUpdate = kFALSE;

   if (!image) {
      Error("Paint", "image could not be rendered to display");
      return;
   }

   Display *dpy = (Display*)gVirtualX->GetDisplay();
   Pixmap pxmap = asimage2pixmap(fgVisual, DefaultRootWindow(dpy), image,
                                 0, kTRUE);
   Int_t wid = gVirtualX->AddWindow(pxmap, to_w, to_h);
   gPad->cd();
   gVirtualX->CopyPixmap(wid, kFRS, kFRS);
   gVirtualX->RemoveWindow(wid);

   gPad->cd();

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

      Double_t xconv = gPad->PixeltoX(to_w) / image->width;
      Double_t yconv = TMath::Abs(gPad->PixeltoY(to_h)) / image->height;
      Double_t x1 = kFRS * xconv;
      Double_t x2 = (kFRS + 1) * xconv;
      Double_t y2 = 1 - kFRS * yconv;
      Double_t y1 = 1 - (kFRS + 1) * yconv;
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
         xconv = gPad->PixeltoX(Int_t(0.3 * pal_w)) / grad_im->width;
         yconv = TMath::Abs(gPad->PixeltoY(to_h - 20)) / grad_im->height;
         x1 = (to_w + 0.2 * pal_w) * xconv;
         x2 = ((to_w + 0.2 * pal_w) + 1) * xconv;
         y2 = 1 - (kFRS + 20) * yconv;
         y1 = 1 - (kFRS + 21) * yconv;
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
      }
   }

   if (grad_im) {
      // draw color bar
      pxmap = asimage2pixmap(fgVisual, DefaultRootWindow(dpy), grad_im,
                             0, kTRUE);
      wid = gVirtualX->AddWindow(pxmap, UInt_t(0.3 * pal_w), to_h - 20);

      gPad->cd();
      gVirtualX->CopyPixmap(wid, Int_t(to_w + 0.2 * pal_w), kFRS + 20);
      gVirtualX->RemoveWindow(wid);

      gPad->cd();

      // values of palette
      TGaxis axis;
      Int_t ndiv = 510;
      double min = fMinValue;
      double max = fMaxValue;
      axis.SetLineColor(0);       // draw white ticks
      axis.PaintAxis((to_w + 0.5 * pal_w) / pad_w, (pad_h - to_h - kFRS - 1) / pad_h,
                     (to_w + 0.5 * pal_w) / pad_w, (pad_h - kFRS - 21) / pad_h,
                     min, max, ndiv, "+LU");
      min = fMinValue;
      max = fMaxValue;
      axis.SetLineColor(1);       // draw black ticks
      axis.PaintAxis((to_w + 0.5 * pal_w) / pad_w, (pad_h - to_h - kFRS) / pad_h,
                     (to_w + 0.5 * pal_w) / pad_w, (pad_h - kFRS - 20) / pad_h,
                     min, max, ndiv, "+L");

   }

   if (grad_im)
      destroy_asimage(&grad_im);
}

//______________________________________________________________________________
Int_t TASImage::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Is the mouse in the image?

   Int_t pxl, pyl, pxt, pyt;
   Int_t px1 = gPad->XtoAbsPixel(gPad->GetX1());
   Int_t py1 = gPad->YtoAbsPixel(gPad->GetY1());
   Int_t px2 = gPad->XtoAbsPixel(gPad->GetX2());
   Int_t py2 = gPad->YtoAbsPixel(gPad->GetY2());
   if (px1 < px2) {pxl = px1; pxt = px2;}
   else           {pxl = px2; pxt = px1;}
   if (py1 < py2) {pyl = py1; pyt = py2;}
   else           {pyl = py2; pyt = py1;}

   // Are we inside the image leave 5 (kFRS) pixels on all sides to
   // be able to grab the pad
   if ((px > pxl+kFRS && px < pxt-kFRS) && (py > pyl+kFRS && py < pyt-kFRS))
      return 0;

   return 999999;
}

//______________________________________________________________________________
void TASImage::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute mouse events.

   gPad->SetCursor(kCross);

   static Int_t stx, sty;
   static Int_t oldx, oldy;

   if (!IsValid()) return;

   if (event ==kButton1Motion || event == kButton1Down  ||
       event == kButton1Up) {

      // convert to image pixel on screen
      Int_t imgX = (Int_t)(gPad->AbsPixeltoX(px) * gPad->XtoPixel(1) + 0.5) - kFRS;
      Int_t imgY = (Int_t)((1 - gPad->AbsPixeltoY(py)) * gPad->YtoPixel(0) + 0.5) - kFRS;

      if (imgX < 0)  px = px - imgX;
      if (imgY < 0)  py = py - imgY;

      ASImage *image = fImage;
      if (fScaledImage) image = fScaledImage;

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

            Double_t xfact = (fScaledImage) ? (Double_t)fScaledImage->width  / fZoomWidth  : 1;
            Double_t yfact = (fScaledImage) ? (Double_t)fScaledImage->height / fZoomHeight : 1;

            Int_t imgX1 = (Int_t)(gPad->AbsPixeltoX(stx) * gPad->XtoPixel(1) + 0.5) - kFRS;
            Int_t imgY1 = (Int_t)((1 - gPad->AbsPixeltoY(sty)) * gPad->YtoPixel(0) + 0.5) - kFRS;
            Int_t imgX2 = (Int_t)(gPad->AbsPixeltoX(px)  * gPad->XtoPixel(1) + 0.5) - kFRS;
            Int_t imgY2 = (Int_t)((1 - gPad->AbsPixeltoY(py))  * gPad->YtoPixel(0) + 0.5) - kFRS;
            imgY1 = image->height - 1 - imgY1;
            imgY2 = image->height - 1 - imgY2;
            imgX1 = (Int_t)(imgX1 / xfact) + fZoomOffX;
            imgY1 = (Int_t)(imgY1 / yfact) + fZoomOffY;
            imgX2 = (Int_t)(imgX2 / xfact) + fZoomOffX;
            imgY2 = (Int_t)(imgY2 / yfact) + fZoomOffY;

            Zoom((imgX1 < imgX2) ? imgX1 : imgX2, (imgY1 < imgY2) ? imgY1 : imgY2,
                 abs(imgX1 - imgX2) + 1, abs(imgY1 - imgY2) + 1);

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
   px = (Int_t)(gPad->AbsPixeltoX(px) * gPad->XtoPixel(1) + 0.5) - kFRS;
   py = (Int_t)((1 - gPad->AbsPixeltoY(py)) * gPad->YtoPixel(0) + 0.5) - kFRS;

   // no info if mouse is outside of image
   if (px < 0 || py < 0)  return info;

   ASImage *image = fImage;
   if (fScaledImage) image = fScaledImage;
   if (px >= (int)image->width || py >= (int)image->height)
      return info;

   py = image->height - 1 - py;
   // convert to original image size and take zooming into account
   if (fScaledImage) {
      px = (Int_t)(px / (Double_t)fScaledImage->width  * fZoomWidth ) + fZoomOffX;
      py = (Int_t)(py / (Double_t)fScaledImage->height * fZoomHeight) + fZoomOffY;
   }

   if (fImage->alt.vector)
      sprintf(info, "x: %d  y: %d   %.5g",
              px, py, fImage->alt.vector[px + py * fImage->width]);
   else
      sprintf(info, "x: %d  y: %d", px, py);

   return info;
}

//______________________________________________________________________________
void TASImage::SetPalette(const TImagePalette *palette)
{
   // Set a new palette to an image. Only images that were created with the
   // SetImage() functions can be modified with this function.
   // The previously used palette is destroyed.

   TAttImage::SetPalette(palette);

   if (!InitVisual())
      return;

   if (!IsValid())
      return;

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

   if (fScaledImage) {
      destroy_asimage(&fScaledImage);
      fScaledImage = 0;
   }
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

   if (!IsValid())
      return;

   if (!InitVisual())
      return;

   if (toWidth < 1) toWidth = 1;
   if (toHeight < 1 ) toHeight = 1;

   ASImage *img = scale_asimage(fgVisual, fImage, toWidth, toHeight,
                                ASA_ASImage, GetImageCompression(),
                                GetImageQuality());
   if (fImage)
      destroy_asimage(&fImage);

   if (fScaledImage) {
      destroy_asimage(&fScaledImage);
      fScaledImage = 0;
   }

   fImage = img;
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

   if (!IsValid())
      return;

   fZoomUpdate = kTRUE;

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

   if (!IsValid())
      return;

   fZoomUpdate = kTRUE;
   fZoomOffX   = 0;
   fZoomOffY   = 0;
   fZoomWidth  = fImage->width;
   fZoomHeight = fImage->height;
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

   if (!IsValid())
      return;

   if (!InitVisual())
      return;

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
                               ASA_ASImage, 0, ASIMAGE_QUALITY_TOP);
   if (fImage)
      destroy_asimage(&fImage);

   if (fScaledImage) {
      destroy_asimage(&fScaledImage);
      fScaledImage = 0;
   }

   fImage = img;
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

   if (!IsValid())
      return;

   if (!InitVisual())
      return;

   if (fImage->alt.vector) {
      Warning("Mirror", "mirror does not work for data images");
      return;
   }

   ASImage *img = mirror_asimage(fgVisual, fImage, 0, 0,
                                 fImage->width, fImage->height, vert,
                                 ASA_ASImage, 0, ASIMAGE_QUALITY_TOP);
   if (fImage)
      destroy_asimage(&fImage);

   if (fScaledImage) {
      destroy_asimage(&fScaledImage);
      fScaledImage = 0;
   }

   fImage = img;
}

//______________________________________________________________________________
UInt_t TASImage::GetWidth() const
{
   // Return width of original image not of the displayed image.

   if (fImage)
      return fImage->width;
   return 0;
}

//______________________________________________________________________________
UInt_t TASImage::GetHeight() const
{
   // Return height of original image not of the displayed image.

   if (fImage)
      return fImage->height;
   return 0;
}

//______________________________________________________________________________
Bool_t TASImage::InitVisual()
{
   // Static function to initialize the ASVisual.

   if (fgVisual) return kTRUE;
   if (gROOT->IsBatch()) return kFALSE;

   Display *dpy  = (Display*)gVirtualX->GetDisplay();
   Int_t screen  = gVirtualX->GetScreen();
   Int_t depth   = gVirtualX->GetDepth();
   Visual *vis   = (Visual*) gVirtualX->GetVisual();
   Colormap cmap = (Colormap) gVirtualX->GetColormap();
   fgVisual = create_asvisual_for_id(dpy, screen, depth,
                                     XVisualIDFromVisual(vis), cmap, 0);
   return kTRUE;
}

//______________________________________________________________________________
void TASImage::StartPaletteEditor()
{
   // Start palette editor.

   if (!IsValid())
      return;

   if (fImage->alt.vector == 0) {
      Warning("StartPaletteEditor", "palette can be modified only for data images");
      return;
   }

   // Opens a GUI to edit the color palette
   TAttImage::StartPaletteEditor();
}


