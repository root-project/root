// @(#)root/graf:$Id$
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
//  We provide few predifined palettes:                                 //
//                                                                      //
//    o gHistImagePalette - palette used in TH2::Draw("col")            //
//                                                                      //
//    o gWebImagePalette                                                //
//       The web palette is a set of 216 colors that will not dither or //
//       shift on PCs or Macs. Browsers use this built-in palette when  //
//       they need to render colors on monitors with only 256 colors    //
//       (also called 8-bit color monitors).                            //
//       The 6x6x6 web palette provides very quick color index lookup   //
//       and can be used for good quality convertion of images into     //
//       2-D histograms.                                                //
//                                                                      //
//    o  TImagePalette(Int_t ncolors, Int_t *colors)                    //
//        if ncolors <= 0 a default palette (see below) of 50 colors    //
//        is defined.                                                   //
//                                                                      //   
//        if ncolors == 1 && colors == 0, then                          //
//        a Pretty Palette with a Spectrum Violet->Red is created.      //
//                                                                      //
//        if ncolors > 50 and colors=0, the DeepSea palette is used.    //
//         (see TStyle::CreateGradientColorTable for more details)      //
//                                                                      //
//        if ncolors > 0 and colors = 0, the default palette is used    //
//        with a maximum of ncolors.                                    //
//                                                                      //
// The default palette defines:                                         //
//   index 0->9   : grey colors from light to dark grey                 //
//   index 10->19 : "brown" colors                                      //
//   index 20->29 : "blueish" colors                                    //
//   index 30->39 : "redish" colors                                     //
//   index 40->49 : basic colors                                        //
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
#include "TColor.h"
#include "TMath.h"


ClassImp(TPaletteEditor)
ClassImp(TAttImage)
ClassImp(TImagePalette)


// definition of a default palette
const Int_t kNUM_DEFAULT_COLORS = 12;
static UShort_t gAlphaDefault[kNUM_DEFAULT_COLORS] = {
   0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
   0xffff, 0xffff, 0xffff, 0xffff
};

static UShort_t gRedDefault[kNUM_DEFAULT_COLORS] = {
   0x0000, 0x0000, 0x7000, 0x0000, 0x0000, 0x0000, 0xffff, 0xffff,
   0x7000, 0x8000, 0xffff, 0xffff
};

static UShort_t gGreenDefault[kNUM_DEFAULT_COLORS] = {
   0x0000, 0x0000, 0x0000, 0x0000, 0xffff, 0xffff, 0xffff, 0x0000,
   0x0000, 0x8000, 0xffff, 0xffff
};

static UShort_t gBlueDefault[kNUM_DEFAULT_COLORS] = {
   0x0000, 0x0000, 0x7000, 0xffff, 0xffff, 0x0000, 0x0000, 0x0000,
   0x0000, 0xa000, 0xffff, 0xffff
};


//////////////////////////// Web Palette ////////////////////////////////////
static UShort_t gWebBase[6] = { 0, 51, 102, 153, 204, 255 };

class TWebPalette : public TImagePalette {

private:
   Int_t fCLUT[6][6][6];   // Color LookUp Table

public:
   TWebPalette() : TImagePalette() {
      int i = 0;
      fNumPoints = 216;
      fPoints = new Double_t[216];
      fColorRed = new UShort_t[216];
      fColorBlue = new UShort_t[216];
      fColorGreen = new UShort_t[216];
      fColorAlpha = new UShort_t[216];

      for (i = 0; i < 214; i++) {
         fPoints[i + 1]  =  (double)i/213;
      }
      fPoints[0] = 0;
      fPoints[215] = 1;

      i = 0;
      for (int r = 0; r < 6; r++) {
         for (int g = 0; g < 6; g++) {
            for (int b = 0; b < 6; b++) {
               fColorRed[i] = gWebBase[r] << 8;
               fColorGreen[i] = gWebBase[g] << 8;
               fColorBlue[i] = gWebBase[b] << 8;
               fColorAlpha[i] = 0xffff;
               fCLUT[r][g][b] = i;
               i++;
            }
         }
      }
   }

   Int_t FindColor(UShort_t r, UShort_t g, UShort_t b) {
      Int_t ri = TMath:: BinarySearch(6, (const Short_t*)gWebBase, (Short_t)r);
      Int_t gi = TMath:: BinarySearch(6, (const Short_t*)gWebBase, (Short_t)g);
      Int_t bi = TMath:: BinarySearch(6, (const Short_t*)gWebBase, (Short_t)b);
      return fCLUT[ri][gi][bi];
   }

   Int_t *GetRootColors() {
      static Int_t *gRootColors = 0;
      if (gRootColors) return gRootColors;

      gRootColors = new Int_t[216];

      int i = 0;
      for (int r = 0; r < 6; r++) {
         for (int g = 0; g < 6; g++) {
            for (int b = 0; b < 6; b++) {
               gRootColors[i] = TColor::GetColor(gWebBase[r], gWebBase[g], gWebBase[b]);
               i++;
            }
         }
      }
      return gRootColors;
   }
};

TImagePalette *gWebImagePalette = new TWebPalette();


////////////////////////////// Hist Palette ////////////////////////////////////
static Double_t gDefHistP[50] =  {
      0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,
      0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,
      0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,
      0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98 };

static UShort_t gDefHistR[50] = {
      242,229,204,178,153,127,102,76,192,204,204,193,186,178,183,173,155,135,
      175,132,89,137,130,173,122, 117,104,109,124,127,170,89,211,221,188,198,
      191,170,165,147,206,211,255,0,255,255,0,0,53,0 };

static UShort_t gDefHistG[50] = {
      242,229,204,178,153,127,102,76,182,198,198,191,181,165,163,153,142,102,
      206,193,211,168,158,188,142,137,130,122,153,127,165,84,206,186,158,153,
      130,142,119,104,94,89,0,255,0,255,0,255,53,0 };

static UShort_t gDefHistB[50] = {
      242,229,204,178,153,127,102,76,172,170,170,168,163,150,155,140,130,86,
      198,163,84,160,140,198,153,145,150,132,209,155,191,216,135,135,130,124,
      119,147,122,112,96,84,0,255,255,0,255,0,53,0 };

static UShort_t gDefHistA[50] = {
      255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
      255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
      255,255,255,255,255,255,255,255,255,255,255,255,255,255 };

static Int_t gDefHistRoot[50] = {
      19,18,17,16,15,14,13,12,11,20,21,22,23,24,25,26,27,28,29,30, 8,
      31,32,33,34,35,36,37,38,39,40, 9, 41,42,43,44,45,47,48,49,46,50, 2,
      7, 6, 5, 4, 3, 112,1};


class TDefHistImagePalette : public TImagePalette {

public:
   TDefHistImagePalette() : TImagePalette() {
      fNumPoints = 50;
      fPoints = gDefHistP;
      fColorRed = gDefHistR;
      fColorGreen = gDefHistG;
      fColorBlue = gDefHistB;
      fColorAlpha = gDefHistA;

      for (int i = 0; i<50; i++) {
         fColorRed[i] = fColorRed[i] << 8;
         fColorGreen[i] = fColorGreen[i] << 8;
         fColorBlue[i] = fColorBlue[i] << 8;
         fColorAlpha[i] = fColorAlpha[i] << 8;
      }
   }

   Int_t *GetRootColors() { return gDefHistRoot; }
};

TImagePalette *gHistImagePalette = new TDefHistImagePalette();


///////////////////////////////////////////////////////////////////////////////
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
TImagePalette::TImagePalette(const TImagePalette &palette) : TObject(palette)
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
TImagePalette::TImagePalette(Int_t ncolors, Int_t *colors)
{
   // Creates palette in the same way as TStyle::SetPalette

   fNumPoints  = 0;
   fPoints     = 0;
   fColorRed   = 0;
   fColorGreen = 0;
   fColorBlue  = 0;
   fColorAlpha = 0;

   Int_t i;
   static Int_t palette[50] = {19,18,17,16,15,14,13,12,11,20,
                        21,22,23,24,25,26,27,28,29,30, 8,
                        31,32,33,34,35,36,37,38,39,40, 9,
                        41,42,43,44,45,47,48,49,46,50, 2,
                         7, 6, 5, 4, 3, 112,1};
   TColor *col = 0;
   Float_t step = 0;
   // set default palette (pad type)
   if (ncolors <= 0) {
      ncolors = 50;
      fNumPoints  = ncolors;
      step = 1./fNumPoints;
      fPoints     = new Double_t[fNumPoints];
      fColorRed   = new UShort_t[fNumPoints];
      fColorGreen = new UShort_t[fNumPoints];
      fColorBlue  = new UShort_t[fNumPoints];
      fColorAlpha = new UShort_t[fNumPoints];
      for (i=0;i<ncolors;i++) {
         col = gROOT->GetColor(palette[i]);
         fPoints[i] = i*step;
         if (col) {
            fColorRed[i]   = UShort_t(col->GetRed()*255)  << 8;
            fColorGreen[i] = UShort_t(col->GetGreen()*255) << 8;
            fColorBlue[i]  = UShort_t(col->GetBlue()*255) << 8;
         }
         fColorAlpha[i] = 65280;
      }
      return;
   }

   // set Pretty Palette Spectrum Violet->Red
   if (ncolors == 1 && colors == 0) {
      ncolors = 50;
      fNumPoints  = ncolors;
      step = 1./fNumPoints;
      fPoints     = new Double_t[fNumPoints];
      fColorRed   = new UShort_t[fNumPoints];
      fColorGreen = new UShort_t[fNumPoints];
      fColorBlue  = new UShort_t[fNumPoints];
      fColorAlpha = new UShort_t[fNumPoints];

      // 0 point is white
      fPoints[0] = 0;
      fColorRed[0] = 255 << 8;
      fColorGreen[0] = 255 << 8;
      fColorBlue[0] = 255 << 8;
      fColorAlpha[0] = 0;

      for (i=1;i<ncolors;i++) {
         col = gROOT->GetColor(51+i);
         fPoints[i] = i*step;
         if (col) {
            fColorRed[i]   = UShort_t(col->GetRed()*255) << 8;
            fColorGreen[i] = UShort_t(col->GetGreen()*255) << 8;
            fColorBlue[i]  = UShort_t(col->GetBlue()*255) << 8;
         }
         fColorAlpha[i] = 65280;
      }
      return;
   }

   // set DeepSea palette
   if (colors == 0 && ncolors > 50) {
      static const Int_t nRGBs = 5;
      static Float_t stops[nRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
      static Float_t red[nRGBs] = { 0.00, 0.09, 0.18, 0.09, 0.00 };
      static Float_t green[nRGBs] = { 0.01, 0.02, 0.39, 0.68, 0.97 };
      static Float_t blue[nRGBs] = { 0.17, 0.39, 0.62, 0.79, 0.97 };
      fNumPoints = nRGBs;
      fPoints     = new Double_t[fNumPoints];
      fColorRed   = new UShort_t[fNumPoints];
      fColorGreen = new UShort_t[fNumPoints];
      fColorBlue  = new UShort_t[fNumPoints];
      fColorAlpha = new UShort_t[fNumPoints];
      for (i=0;i<(int)fNumPoints;i++) {
         fPoints[i] = stops[i];
         fColorRed[i] = UShort_t(red[i]*255) << 8;
         fColorGreen[i] = UShort_t(green[i]*255) << 8;
         fColorBlue[i] = UShort_t(blue[i]*255) << 8;
         fColorAlpha[i] = 65280;   
      }
      return;
   }

   // set user defined palette
   if (colors)  {
      fNumPoints  = ncolors;
      step = 1./fNumPoints;
      fPoints     = new Double_t[fNumPoints];
      fColorRed   = new UShort_t[fNumPoints];
      fColorGreen = new UShort_t[fNumPoints];
      fColorBlue  = new UShort_t[fNumPoints];
      fColorAlpha = new UShort_t[fNumPoints];
      for (i=0;i<ncolors;i++) {
         fPoints[i] = i*step;
         col = gROOT->GetColor(colors[i]);
         if (col) {
            fColorRed[i] = UShort_t(col->GetRed()*255) << 8;
            fColorGreen[i] = UShort_t(col->GetGreen()*255) << 8;
            fColorBlue[i] = UShort_t(col->GetBlue()*255) << 8;
            fColorAlpha[i] = 65280;
         } else {
            fColorRed[i] = 0;
            fColorGreen[i] = 0;
            fColorBlue[i] = 0;
            fColorAlpha[i] = 0;
         }
      }
   }
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
Int_t TImagePalette::FindColor(UShort_t r, UShort_t g, UShort_t b)
{
   // returns an index of the closest color

   Int_t ret = 0;
   UInt_t d = 10000;
   UInt_t min = 10000;

   for (UInt_t i = 0; i < fNumPoints; i++) {
      d = TMath::Abs(r - ((fColorRed[i] & 0xff00) >> 8)) +
          TMath::Abs(g - ((fColorGreen[i] & 0xff00) >> 8)) +
          TMath::Abs(b - ((fColorBlue[i] & 0xff00) >> 8));
      if (d < min) {
         min = d;
         ret = i;
      }
   }
   return ret;
}

//______________________________________________________________________________
Int_t *TImagePalette::GetRootColors()
{
   // Returns a list of ROOT colors. Could be used to set histogram palette.
   // See also http://root.cern.ch/root/htmldoc/TStyle.html#TStyle:SetPalette

   static Int_t *gRootColors = 0;
   if (gRootColors) return gRootColors;

   gRootColors = new Int_t[fNumPoints];

   for (UInt_t i = 0; i < fNumPoints; i++) {
      gRootColors[i] = TColor::GetColor(fColorRed[i], fColorGreen[i], fColorBlue[i]);
   }
   return gRootColors;
}


//______________________________________________________________________________
TAttImage::TAttImage()
{
   // TAttImage default constructor.
   // Calls ResetAttImage to set the attributes to a default state.

   ResetAttImage();
   fPaletteEditor = 0;
   fPaletteEnabled = kTRUE;
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
   fPaletteEnabled = kTRUE;
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

   memcpy(fPalette.fColorRed,   gRedDefault,   kNUM_DEFAULT_COLORS * sizeof(UShort_t));
   memcpy(fPalette.fColorGreen, gGreenDefault, kNUM_DEFAULT_COLORS * sizeof(UShort_t));
   memcpy(fPalette.fColorBlue,  gBlueDefault,  kNUM_DEFAULT_COLORS * sizeof(UShort_t));
   memcpy(fPalette.fColorAlpha, gAlphaDefault, kNUM_DEFAULT_COLORS * sizeof(UShort_t));

   for (Int_t point = 0; point < kNUM_DEFAULT_COLORS - 2; point++)
      fPalette.fPoints[point + 1]  =  (double)point / (kNUM_DEFAULT_COLORS - 3);
   fPalette.fPoints[0] = 0;
   fPalette.fPoints[kNUM_DEFAULT_COLORS - 1] = 1;
}

//______________________________________________________________________________
void TAttImage::SaveImageAttributes(ostream &out, const char *name,
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

      memcpy(fPalette.fColorRed,   gRedDefault,   kNUM_DEFAULT_COLORS * sizeof(UShort_t));
      memcpy(fPalette.fColorGreen, gGreenDefault, kNUM_DEFAULT_COLORS * sizeof(UShort_t));
      memcpy(fPalette.fColorBlue,  gBlueDefault,  kNUM_DEFAULT_COLORS * sizeof(UShort_t));
      memcpy(fPalette.fColorAlpha, gAlphaDefault, kNUM_DEFAULT_COLORS * sizeof(UShort_t));

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
