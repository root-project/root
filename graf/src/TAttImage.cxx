// @(#)root/graf:$Name:  $:$Id: TAttImage.cxx,v 1.1 2002/08/09 13:56:00 rdm Exp $
// Author: Reiner Rohlfs   24/03/02

/*************************************************************************
 * Copyright (C) 2001-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TAttImage                                                           //
//                                                                      //
//  Image attributes are:                                               //
//    Image Quality (see EImageQuality for the list of qualities)       //
//    Compression defines the compression rate of the color data in the //
//                internal image structure. Speed and memory depends    //
//                on this rate, but not the image display itself        //
//                0: no compression;  100: max compression              //
//    Radio Flag: kTRUE  the x/y radio of the displayed image is always //
//                       identical to the original image                //
//                kFALSE the x and y size of the displayed image depends//
//                       on the size of the pad                         //
//    Palette:    Defines the conversion from a pixel value to the      //
//                screen color                                          //
//                                                                      //
//  This class is used (in general by secondary inheritance)            //
//  by some other classes (image display).                              //
//                                                                      //
//                                                                      //
//  TImagePalette                                                       //
//                                                                      //
//  A class to define a conversion from pixel values to pixel color.    //
//  A Palette is defined by some anchor points. Each anchor point has   //
//  a value between 0 and 1 and a color. An image has to be normalized  //
//  and the values between the anchor points are interpolated.          //
//  All member variables are public and can be directly manipulated.    //
//  In most cases the default operator will be used to create a         //
//  TImagePalette. In this case the member arrays have to be allocated  //
//  by an application and will be deleted in the destructor of this     //
//  class.                                                              //
//                                                                      //
//                                                                      //
//  TPaletteEditor                                                      //
//                                                                      //
//  This class provides a way to edit the palette via a GUI.            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAttImage.h"
#include "TROOT.h"
#include "TPluginManager.h"
#include "Riostream.h"


// definition of a default palette
const Int_t kNUM_DEFAULT_COLORS = 12;
static UShort_t alphaDefault[kNUM_DEFAULT_COLORS] = {
   0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
   0xffff, 0xffff, 0xffff, 0xffff
};

static UShort_t redDefault[kNUM_DEFAULT_COLORS] = {
   0x0000, 0x0000, 0x7000, 0x0000, 0x0000, 0x0000, 0xffff, 0xffff,
   0x7000, 0x8000, 0xffff, 0xffff
};

static UShort_t greenDefault[kNUM_DEFAULT_COLORS] = {
   0x0000, 0x0000, 0x0000, 0x0000, 0xffff, 0xffff, 0xffff, 0x0000,
   0x0000, 0x8000, 0xffff, 0xffff
};

static UShort_t blueDefault[kNUM_DEFAULT_COLORS] = {
   0x0000, 0x0000, 0x7000, 0xffff, 0xffff, 0x0000, 0x0000, 0x0000,
   0x0000, 0xa000, 0xffff, 0xffff
};


ClassImp(TPaletteEditor)
ClassImp(TAttImage)
ClassImp(TImagePalette)

//______________________________________________________________________________
TPaletteEditor::TPaletteEditor(TAttImage *attImage, UInt_t, UInt_t)
{
   // Constructor.

   fAttImage = attImage;
}

//______________________________________________________________________________
void TPaletteEditor::CloseWindow()
{
   // Closes the window and deletes itself.

   fAttImage->EditorClosed();
}


//______________________________________________________________________________
TImagePalette::TImagePalette()
{
   // Default constructor, sets all pointers to 0.

   fNumPoints     = 0;
   fPoints        = 0;
   fColorRed      = 0;
   fColorGreen    = 0;
   fColorBlue     = 0;
   fColorAlpha    = 0;
}

//______________________________________________________________________________
TImagePalette::TImagePalette(UInt_t numPoints)
{
   // Constructor for a palette with numPoints anchor points.
   // It allocates the memory but does not set any colors.

   fNumPoints  = numPoints;
   fPoints     = new Double_t[fNumPoints];
   fColorRed   = new UShort_t[fNumPoints];
   fColorGreen = new UShort_t[fNumPoints];
   fColorBlue  = new UShort_t[fNumPoints];
   fColorAlpha = new UShort_t[fNumPoints];
}

//______________________________________________________________________________
TImagePalette::TImagePalette(const TImagePalette &palette)
{
   // Copy constructor.

   fNumPoints = palette.fNumPoints;

   fPoints = new Double_t[fNumPoints];
   memcpy(fPoints, palette.fPoints, fNumPoints * sizeof(Double_t));

   fColorRed   = new UShort_t[fNumPoints];
   fColorGreen = new UShort_t[fNumPoints];
   fColorBlue  = new UShort_t[fNumPoints];
   fColorAlpha = new UShort_t[fNumPoints];
   memcpy(fColorRed,   palette.fColorRed,   fNumPoints * sizeof(UShort_t));
   memcpy(fColorGreen, palette.fColorGreen, fNumPoints * sizeof(UShort_t));
   memcpy(fColorBlue,  palette.fColorBlue,  fNumPoints * sizeof(UShort_t));
   memcpy(fColorAlpha, palette.fColorAlpha, fNumPoints * sizeof(UShort_t));
}

//______________________________________________________________________________
TImagePalette::~TImagePalette()
{
   // Destructor.

   delete [] fPoints;
   delete [] fColorRed;
   delete [] fColorGreen;
   delete [] fColorBlue;
   delete [] fColorAlpha;
}

//______________________________________________________________________________
TImagePalette &TImagePalette::operator=(const TImagePalette &palette)
{
   // Assignment operator.

   if (this != &palette) {
      fNumPoints = palette.fNumPoints;

      delete [] fPoints;
      fPoints = new Double_t[fNumPoints];
      memcpy(fPoints, palette.fPoints, fNumPoints * sizeof(Double_t));

      delete [] fColorRed;
      fColorRed = new UShort_t[fNumPoints];
      memcpy(fColorRed, palette.fColorRed, fNumPoints * sizeof(UShort_t));

      delete [] fColorGreen;
      fColorGreen = new UShort_t[fNumPoints];
      memcpy(fColorGreen, palette.fColorGreen, fNumPoints * sizeof(UShort_t));

      delete [] fColorBlue;
      fColorBlue = new UShort_t[fNumPoints];
      memcpy(fColorBlue, palette.fColorBlue, fNumPoints * sizeof(UShort_t));

      delete [] fColorAlpha;
      fColorAlpha = new UShort_t[fNumPoints];
      memcpy(fColorAlpha, palette.fColorAlpha, fNumPoints * sizeof(UShort_t));
   }

   return *this;
}



//______________________________________________________________________________
TAttImage::TAttImage()
{
   // TAttImage default constructor.
   // Calls ResetAttImage to set the attributes to a default state.

   ResetAttImage();
   fPaletteEditor = 0;
}

//______________________________________________________________________________
TAttImage::TAttImage(EImageQuality lquality, UInt_t lcompression,
                     Bool_t constRatio)
{
   // TAttImage normal constructor.
   // Image attributes are taken from the argument list
   //    qualtity     : must be one of EImageQuality (kImgDefault is same as
   //                   kImgGood in the current implementation)
   //    lcompression : defines the compression rate of the color data in the
   //                   image. Speed and memory depends on this rate, but not
   //                   the image display itself
   //                   0: no compression;  100: max compression
   //    constRatio   : keeps the aspect ratio of the image constant on the
   //                   screen (in pixel units)

   ResetAttImage();

   fImageQuality = lquality;
   fImageCompression = (lcompression > 100) ? 100 : lcompression;
   fConstRatio = constRatio;
   fPaletteEditor = 0;
}

//______________________________________________________________________________
TAttImage::~TAttImage()
{
   // TAttImage destructor.

   delete fPaletteEditor;
}

//______________________________________________________________________________
void TAttImage::Copy(TAttImage &attimage) const
{
   // Copy this image attributes to a new attimage.

   attimage.fImageQuality     = fImageQuality;
   attimage.fImageCompression = fImageCompression;
   attimage.fConstRatio       = fConstRatio;
   attimage.fPalette          = fPalette;
}

//______________________________________________________________________________
void TAttImage::ResetAttImage(Option_t *)
{
   // Reset this image attributes to default values.
   // Default values are:
   //    quality:     kImgPoor, (no smoothing while the image is zoomed)
   //    compression: 0 (no compression)
   //    constRatio:  kTRUE
   //    palette:     a default rainbow palette

   fImageQuality      = kImgPoor;
   fImageCompression  = 0;
   fConstRatio        = kTRUE;

   // set the default palette
   delete [] fPalette.fPoints;
   delete [] fPalette.fColorRed;
   delete [] fPalette.fColorGreen;
   delete [] fPalette.fColorBlue;
   delete [] fPalette.fColorAlpha;

   fPalette.fNumPoints = kNUM_DEFAULT_COLORS;

   fPalette.fColorRed    = new UShort_t [kNUM_DEFAULT_COLORS];
   fPalette.fColorGreen  = new UShort_t [kNUM_DEFAULT_COLORS];
   fPalette.fColorBlue   = new UShort_t [kNUM_DEFAULT_COLORS];
   fPalette.fColorAlpha  = new UShort_t [kNUM_DEFAULT_COLORS];
   fPalette.fPoints      = new Double_t [kNUM_DEFAULT_COLORS];

   memcpy(fPalette.fColorRed,   redDefault,   kNUM_DEFAULT_COLORS * sizeof(UShort_t));
   memcpy(fPalette.fColorGreen, greenDefault, kNUM_DEFAULT_COLORS * sizeof(UShort_t));
   memcpy(fPalette.fColorBlue,  blueDefault,  kNUM_DEFAULT_COLORS * sizeof(UShort_t));
   memcpy(fPalette.fColorAlpha, alphaDefault, kNUM_DEFAULT_COLORS * sizeof(UShort_t));

   for (Int_t point = 0; point < kNUM_DEFAULT_COLORS - 2; point++)
      fPalette.fPoints[point + 1]  =  (double)point / (kNUM_DEFAULT_COLORS - 3);
   fPalette.fPoints[0] = 0;
   fPalette.fPoints[kNUM_DEFAULT_COLORS - 1] = 1;
}

//______________________________________________________________________________
void TAttImage::SaveImageAttributes(ofstream &out, const char *name,
                                    EImageQuality qualdef,
                                    UInt_t comprdef, Bool_t constRatiodef)
{
   // Save image attributes as C++ statement(s) on output stream, but
   // not the palette.

   if (fImageQuality != qualdef) {
      out<<"   "<<name<<"->SetImageQuality("<<fImageQuality<<");"<<endl;
   }
   if (fImageCompression != comprdef) {
      out<<"   "<<name<<"->SetImageCompression("<<fImageCompression<<");"<<endl;
   }
   if (fConstRatio != constRatiodef) {
      out<<"   "<<name<<"->SetConstRatio("<<fConstRatio<<");"<<endl;
   }
}

//______________________________________________________________________________
void TAttImage::SetConstRatio(Bool_t constRatio)
{
   // Set (constRatio = kTRUE) or unset (constRadio = kFALSE) the ratio flag.
   // The aspect ratio of the image on the screen is constant if the ratio
   // flag is set. That means one image pixel is allways a square on the screen
   // independent of the pad size and of the size of the zoomed area.

   fConstRatio = constRatio;
}

//______________________________________________________________________________
void TAttImage::SetPalette(const TImagePalette *palette)
{
   // Set a new palette for the image. If palette == 0 a default
   // rainbow color palette is used.

   if (palette)
      fPalette = *palette;
   else {
      // set default palette

      delete [] fPalette.fPoints;
      delete [] fPalette.fColorRed;
      delete [] fPalette.fColorGreen;
      delete [] fPalette.fColorBlue;
      delete [] fPalette.fColorAlpha;

      fPalette.fNumPoints = kNUM_DEFAULT_COLORS;

      fPalette.fColorRed    = new UShort_t [kNUM_DEFAULT_COLORS];
      fPalette.fColorGreen  = new UShort_t [kNUM_DEFAULT_COLORS];
      fPalette.fColorBlue   = new UShort_t [kNUM_DEFAULT_COLORS];
      fPalette.fColorAlpha  = new UShort_t [kNUM_DEFAULT_COLORS];
      fPalette.fPoints      = new Double_t [kNUM_DEFAULT_COLORS];

      memcpy(fPalette.fColorRed,   redDefault,   kNUM_DEFAULT_COLORS * sizeof(UShort_t));
      memcpy(fPalette.fColorGreen, greenDefault, kNUM_DEFAULT_COLORS * sizeof(UShort_t));
      memcpy(fPalette.fColorBlue,  blueDefault,  kNUM_DEFAULT_COLORS * sizeof(UShort_t));
      memcpy(fPalette.fColorAlpha, alphaDefault, kNUM_DEFAULT_COLORS * sizeof(UShort_t));

      for (Int_t point = 0; point < kNUM_DEFAULT_COLORS - 2; point++)
         fPalette.fPoints[point + 1]  =  (double)point / (kNUM_DEFAULT_COLORS - 3);
      fPalette.fPoints[0] = 0;
      fPalette.fPoints[kNUM_DEFAULT_COLORS - 1] = 1;
   }
}

//______________________________________________________________________________
void TAttImage::StartPaletteEditor()
{
   // Opens a GUI to edit the color palette.

   if (fPaletteEditor == 0) {
      TPluginHandler *h;

      if ((h = gROOT->GetPluginManager()->FindHandler("TPaletteEditor"))) {
         if (h->LoadPlugin() == -1)
            return;
         fPaletteEditor = (TPaletteEditor *) h->ExecPlugin(3, this, 80, 25);
      }
   }
}
