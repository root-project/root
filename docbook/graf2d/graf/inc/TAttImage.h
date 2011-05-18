// @(#)root/graf:$Id$
// Author: Reiner Rohlfs   24/03/02

/*************************************************************************
 * Copyright (C) 2001-2002, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttImage
#define ROOT_TAttImage


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

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif

class TAttImage;


class TPaletteEditor {

protected:
   TAttImage    *fAttImage;    // image attributes to be edited

public:
   TPaletteEditor(TAttImage *attImage, UInt_t w, UInt_t h);
   virtual ~TPaletteEditor() { }

   virtual void CloseWindow();

   ClassDef(TPaletteEditor, 0)  // Base class for palette editor
};



class TImagePalette : public TObject {

public:
   UInt_t      fNumPoints;   // number of anchor points
   Double_t   *fPoints;      // [fNumPoints] value of each anchor point [0..1]
   UShort_t   *fColorRed;    // [fNumPoints] red color at each anchor point
   UShort_t   *fColorGreen;  // [fNumPoints] green color at each anchor point
   UShort_t   *fColorBlue;   // [fNumPoints] blue color at each anchor point
   UShort_t   *fColorAlpha;  // [fNumPoints] alpha at each anchor point

   TImagePalette();
   TImagePalette(const TImagePalette &palette);
   TImagePalette(UInt_t numPoints);
   TImagePalette(Int_t ncolors, Int_t *colors);
   virtual ~TImagePalette();
   virtual Int_t FindColor(UShort_t r, UShort_t g, UShort_t b);
   virtual Int_t *GetRootColors();

   TImagePalette &operator=(const TImagePalette &palette);

   ClassDef(TImagePalette,1)  // Color Palette for value -> color conversion
};



class TAttImage {

public:
   // Defines level of output quality/speed ratio
   enum EImageQuality {
      kImgDefault = -1,
      kImgPoor    = 0,
      kImgFast    = 1,
      kImgGood    = 2,
      kImgBest    = 3
   };

protected:
   EImageQuality    fImageQuality;       // *OPTION={GetMethod="GetImageQuality";SetMethod="SetImageQuality";Items=(kImgDefault="Default",kImgPoor="Poor",kImgFast="Fast",kImgGood="Good",kImgBest="Best")}*
   UInt_t           fImageCompression;   // compression [0 .. 100] 0: no compression
   Bool_t           fConstRatio;         // keep aspect ratio of image on the screen
   TImagePalette    fPalette;            // color palette for value -> color conversion
   TPaletteEditor  *fPaletteEditor;      //! GUI to edit the color palette
   Bool_t           fPaletteEnabled;     //! kTRUE - palette is drawn on the image

public:
   TAttImage();
   TAttImage(EImageQuality lquality, UInt_t lcompression, Bool_t constRatio);
   virtual ~TAttImage();

   void             Copy(TAttImage &attline) const;
   Bool_t           GetConstRatio() const { return fConstRatio; }
   UInt_t           GetImageCompression() const { return fImageCompression; }
   EImageQuality    GetImageQuality() const { return fImageQuality; }
   virtual const TImagePalette &GetPalette() const { return fPalette; }

   virtual void     ResetAttImage(Option_t *option="");
   virtual void     SaveImageAttributes(ostream &out, const char *name,
                                        EImageQuality qualdef = kImgDefault,
                                        UInt_t comprdef = 0,
                                        Bool_t constRatiodef = kTRUE);
   virtual void     SetConstRatio(Bool_t constRatio = kTRUE); // *TOGGLE*
   virtual void     SetPaletteEnabled(Bool_t on = kTRUE) {  fPaletteEnabled = on; }
   virtual void     SetImageCompression(UInt_t lcompression)
                     { fImageCompression = (lcompression > 100) ? 100 : lcompression; } // *MENU*
   virtual void     SetImageQuality(EImageQuality lquality)
                     { fImageQuality = lquality;} // *SUBMENU*
   virtual void     SetPalette(const TImagePalette *palette);
   virtual void     StartPaletteEditor(); // *MENU*
   virtual void     EditorClosed() { fPaletteEditor = 0; }
   Bool_t           IsPaletteEnabled() const { return fPaletteEnabled; }

   ClassDef(TAttImage,1)  //Image attributes
};


R__EXTERN TImagePalette  *gHistImagePalette;    // palette used in TH2::Draw("col")
R__EXTERN TImagePalette  *gWebImagePalette;     // 6x6x6 colors web palette



#endif
